"""
Statistical analysis of the FM x MIL c-index table for the reviewer response.

Answers three requests:
  1. Confidence intervals on every c-index
  2. Fold-level variability (per-fold c-index, mean +/- SD)
  3. Statistical comparison of the leading models (paired tests)

Two c-index conventions exist in this repo; both are reported:
  POOLED  - concatenate the 10 disjoint test folds, one c-index over all
            patients. CI by patient-level bootstrap. This is what
            overall_combined_cindex_matrix.csv contains.
  FOLDWISE- c-index within each fold, then mean +/- SD across folds.
            CI = t-based on the 10 fold values. This is fold_results.csv.

Usage:  python cindex_stats.py
"""

import os, glob, itertools
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy import stats

BASE = os.environ.get('ADG_RESULTS', 'results/tcga_gbmlgg_survival') + os.sep
OUT  = os.path.join(BASE, 'stats') + os.sep

TIME, EVENT, RISK = 'survival', 'censorship', 'risk'
N_BOOT = 2000
SEED   = 0

MIL_ORDER = ['mean_mil', 'max_mil', 'att_mil', 'dsmil', 'trans_mil', 'rrtmil', 'mamba_mil']
MIL_NAME  = {'mean_mil':'MeanMIL', 'max_mil':'MaxMIL', 'att_mil':'ABMIL', 'dsmil':'DSMIL',
             'trans_mil':'TransMIL', 'rrtmil':'RRTMIL', 'mamba_mil':'MambaMIL'}

FM_ORDER = ['imagenet_s1','lunit_s1','gigapath_s1','ctranspath_s1','hibou_s1','hoptimus1_s1',
            'virchow_s1','conch_v15_s1','uni_v2_s1','chief_s1','prism_s1','titan_s1']
FM_NAME  = {'imagenet_s1':'ResNet50','lunit_s1':'Lunit','gigapath_s1':'GigaPath',
            'ctranspath_s1':'CTransPath','hibou_s1':'Hibou','hoptimus1_s1':'H-Optimus1',
            'virchow_s1':'Virchow2','conch_v15_s1':'CONCHv1.5','uni_v2_s1':'UNI2',
            'chief_s1':'CHIEF','prism_s1':'PRISM','titan_s1':'TITAN'}


def load_folds(mil, fm):
    """Return list of per-fold DataFrames (risk, survival, event), fold order fixed."""
    d = os.path.join(BASE, mil, fm)
    files = sorted(glob.glob(os.path.join(d, 'split_*_results.pkl')),
                   key=lambda f: int(os.path.basename(f).split('_')[1]))
    folds = []
    for f in files:
        df = pd.DataFrame(pd.read_pickle(f)).T
        if not {RISK, TIME, EVENT} <= set(df.columns):
            continue
        r = df[RISK].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x)
        out = pd.DataFrame({
            'risk':  r.astype(float).values,
            'time':  df[TIME].astype(float).values,
            'event': 1 - df[EVENT].astype(float).values,   # pkl: 1=censored
        }, index=df.index)
        folds.append(out.replace([np.inf, -np.inf], np.nan).dropna())
    return folds


def cidx(d):
    if len(d) < 2 or d['event'].sum() == 0:
        return np.nan
    return concordance_index(d['time'], -d['risk'], d['event'])


def boot_ci(d, n=N_BOOT, seed=SEED):
    """Patient-level bootstrap CI for the pooled c-index."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(d))
    vals = []
    for _ in range(n):
        s = d.iloc[rng.choice(idx, len(idx), replace=True)]
        v = cidx(s)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def t_ci(x):
    """t-based 95% CI on the mean of the per-fold c-indices."""
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) < 2:
        return np.nan, np.nan
    se = stats.sem(x)
    h = se * stats.t.ppf(0.975, len(x) - 1)
    return x.mean() - h, x.mean() + h


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, per_fold, pooled_store = [], [], {}

    for fm in FM_ORDER:
        for mil in MIL_ORDER:
            folds = load_folds(mil, fm)
            if not folds:
                continue
            pooled = pd.concat(folds, ignore_index=True)
            p_c    = cidx(pooled)
            p_lo, p_hi = boot_ci(pooled)

            fvals  = [cidx(f) for f in folds]
            f_mean = np.nanmean(fvals)
            f_sd   = np.nanstd(fvals, ddof=1)
            f_lo, f_hi = t_ci(fvals)

            pooled_store[(fm, mil)] = (pooled, fvals)

            rows.append(dict(
                FM=FM_NAME.get(fm, fm), MIL=MIL_NAME.get(mil, mil), n=len(pooled),
                pooled_cindex=p_c, pooled_lo=p_lo, pooled_hi=p_hi,
                fold_mean=f_mean, fold_sd=f_sd, fold_lo=f_lo, fold_hi=f_hi,
            ))
            for i, v in enumerate(fvals):
                per_fold.append(dict(FM=FM_NAME.get(fm, fm), MIL=MIL_NAME.get(mil, mil),
                                     fold=i, c_index=v))
            print(f"{FM_NAME.get(fm,fm):<12} {MIL_NAME.get(mil,mil):<9} "
                  f"pooled {p_c:.3f} [{p_lo:.3f}-{p_hi:.3f}]   "
                  f"folds {f_mean:.3f} +/- {f_sd:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT + 'cindex_with_ci.csv', index=False)
    pd.DataFrame(per_fold).to_csv(OUT + 'per_fold_cindex.csv', index=False)

    # ── Formatted table: c-index [95% CI] ────────────────────────────────────
    for tag, c, lo, hi in [('pooled', 'pooled_cindex', 'pooled_lo', 'pooled_hi'),
                           ('foldwise', 'fold_mean', 'fold_lo', 'fold_hi')]:
        t = df.copy()
        t['cell'] = (t[c].map('{:.3f}'.format) + ' [' + t[lo].map('{:.3f}'.format)
                     + '-' + t[hi].map('{:.3f}'.format) + ']')
        piv = t.pivot(index='FM', columns='MIL', values='cell')
        piv = piv.reindex(index=[FM_NAME[f] for f in FM_ORDER if FM_NAME[f] in piv.index],
                          columns=[MIL_NAME[m] for m in MIL_ORDER if MIL_NAME[m] in piv.columns])
        piv.to_csv(OUT + f'table_{tag}_with_ci.csv')

    # ── Statistical comparison of the leading configurations ────────────────
    # Rank by pooled c-index; compare the top configurations against each other.
    top = df.sort_values('pooled_cindex', ascending=False).head(8)
    print("\nTop configurations by pooled c-index:")
    for _, r in top.iterrows():
        print(f"  {r.FM:<12} {r.MIL:<9} {r.pooled_cindex:.4f} "
              f"[{r.pooled_lo:.3f}-{r.pooled_hi:.3f}]")

    name2key = {(FM_NAME.get(f, f), MIL_NAME.get(m, m)): (f, m)
                for f in FM_ORDER for m in MIL_ORDER}
    comps = []
    pairs = list(itertools.combinations(
        [(r.FM, r.MIL) for _, r in top.iterrows()], 2))

    rng = np.random.default_rng(SEED)
    for a, b in pairs:
        ka, kb = name2key.get(a), name2key.get(b)
        if ka not in pooled_store or kb not in pooled_store:
            continue
        pa, fa = pooled_store[ka]
        pb, fb = pooled_store[kb]

        # (i) paired t-test / Wilcoxon over the 10 folds
        fa_, fb_ = np.array(fa), np.array(fb)
        ok = np.isfinite(fa_) & np.isfinite(fb_)
        t_p = stats.ttest_rel(fa_[ok], fb_[ok]).pvalue if ok.sum() > 1 else np.nan
        try:
            w_p = stats.wilcoxon(fa_[ok], fb_[ok]).pvalue if ok.sum() > 1 else np.nan
        except ValueError:
            w_p = np.nan

        # (ii) paired patient-level bootstrap on the shared patients
        shared = pa.index.intersection(pb.index) if pa.index.equals(pb.index) else None
        if shared is None:
            A = pa.reset_index(drop=True); B = pb.reset_index(drop=True)
            n = min(len(A), len(B)); A, B = A.iloc[:n], B.iloc[:n]
        else:
            A, B = pa.loc[shared], pb.loc[shared]
        diffs = []
        idx = np.arange(len(A))
        for _ in range(N_BOOT):
            s = rng.choice(idx, len(idx), replace=True)
            va, vb = cidx(A.iloc[s]), cidx(B.iloc[s])
            if np.isfinite(va) and np.isfinite(vb):
                diffs.append(va - vb)
        diffs = np.array(diffs)
        d_lo, d_hi = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)) if len(diffs) else (np.nan, np.nan)
        boot_p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) if len(diffs) else np.nan

        comps.append(dict(
            model_A=f"{a[0]}+{a[1]}", model_B=f"{b[0]}+{b[1]}",
            c_A=cidx(pa), c_B=cidx(pb), delta=cidx(pa) - cidx(pb),
            delta_lo=d_lo, delta_hi=d_hi, boot_p=boot_p,
            paired_t_p=t_p, wilcoxon_p=w_p,
        ))

    cdf = pd.DataFrame(comps)
    if len(cdf):
        # Holm correction across the comparisons
        order = np.argsort(cdf['boot_p'].values)
        m = len(cdf); adj = np.empty(m); run = 0.0
        for rank, i in enumerate(order):
            val = (m - rank) * cdf['boot_p'].values[i]
            run = max(run, val); adj[i] = min(run, 1.0)
        cdf['boot_p_holm'] = adj
        cdf.to_csv(OUT + 'leading_model_comparisons.csv', index=False)

        print("\nPairwise comparison of leading models "
              "(delta = A - B, paired patient bootstrap):")
        for _, r in cdf.iterrows():
            sig = '*' if r.boot_p_holm < 0.05 else 'ns'
            print(f"  {r.model_A:<22} vs {r.model_B:<22} "
                  f"d={r.delta:+.4f} [{r.delta_lo:+.3f},{r.delta_hi:+.3f}] "
                  f"p={r.boot_p:.3f} holm={r.boot_p_holm:.3f} {sig}")

    print(f"\nWrote outputs to {OUT}")


if __name__ == '__main__':
    main()
