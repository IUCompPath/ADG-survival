"""
Generate patient-level k-fold cross-validation splits for survival MIL.

Only patients that have BOTH a survival label and at least one feature file
are kept, so the folds match the cohort that will actually be trained on.
Splitting is at the patient level: every slide from a patient stays in the
same fold, which prevents the same patient appearing in train and test.

Folds are stratified on the event indicator so each fold carries a similar
number of deaths.

Usage:
    python tools/make_splits.py \
        --csv      dataset_csv/gbmlgg_processed.csv \
        --features /path/to/pt_files \
        --out      splits/tcga_gbmlgg_survival_gigapath_5fold \
        --k 5 --seed 1
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def feature_patients(feature_dir, prefix='TCGA-'):
    """Patient barcodes (first 3 hyphen fields) that have a .pt feature file."""
    pats = set()
    for p in glob.glob(os.path.join(feature_dir, f'{prefix}*.pt')):
        stem = os.path.basename(p)[:-3]
        pats.add('-'.join(stem.split('-')[:3]))
    return pats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--case-col', default='case_id')
    ap.add_argument('--event-col', default='censorship',
                    help='1 = censored in this schema; event = 1 - censorship')
    ap.add_argument('--histology-csv', default=None,
                    help='CSV with case_id + label (0=oligo, 1=astro, 2=GBM). '
                         'When given, folds are stratified on subtype x event '
                         'jointly, not on event alone.')
    ap.add_argument('--label-col', default='label')
    ap.add_argument('--subtypes', type=int, nargs='+', default=None,
                    help='restrict the cohort to these subtype labels '
                         '(0=oligodendroglioma, 1=astrocytoma, 2=glioblastoma). '
                         'With one value the subtype term drops out of the '
                         'stratification, leaving event-only stratification.')
    ap.add_argument('--val-frac', type=float, default=0.0,
                    help='fraction of each training fold held out for validation; '
                         '0 reproduces the existing setup where val == test')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    have_feats = feature_patients(args.features)

    if args.histology_csv:
        hist = (pd.read_csv(args.histology_csv)
                  .drop_duplicates(subset=args.case_col)[[args.case_col, args.label_col]])
        df = df.merge(hist, on=args.case_col, how='left')

    # Subtype restriction must happen BEFORE the per-patient frame is built,
    # otherwise the filter has no effect on the cohort that gets split.
    if args.subtypes:
        if args.label_col not in df.columns:
            raise SystemExit('--subtypes needs --histology-csv for the label column')
        before = df[args.case_col].nunique()
        df = df[df[args.label_col].isin(args.subtypes)]
        print(f"subtype filter {args.subtypes}: "
              f"{before} -> {df[args.case_col].nunique()} patients")

    # One row per patient, restricted to patients with features.
    pat = (df.drop_duplicates(subset=args.case_col)
             .loc[lambda d: d[args.case_col].isin(have_feats)]
             .reset_index(drop=True))
    pat['event'] = 1 - pat[args.event_col].astype(float)

    print(f"patients in csv           : {df[args.case_col].nunique()}")
    print(f"patients with feature file: {len(have_feats)}")
    print(f"cohort for splitting      : {len(pat)}  "
          f"(events={int(pat.event.sum())}, censored={int((1 - pat.event).sum())})")

    if len(pat) == 0:
        raise SystemExit("No overlap between the label CSV and the feature directory.")

    os.makedirs(args.out, exist_ok=True)
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    rng = np.random.default_rng(args.seed)

    cases = pat[args.case_col].values

    # Subtype and event rate are strongly coupled here (GBM ~65% events,
    # oligodendroglioma ~9%), so stratifying on either alone unbalances the
    # other. Stratify on the joint subtype x event class instead.
    single_subtype = args.subtypes is not None and len(args.subtypes) == 1
    if args.histology_csv and args.label_col in pat.columns and not single_subtype:
        missing = pat[args.label_col].isna().sum()
        if missing:
            print(f"  warning: {missing} patients lack a subtype label; "
                  f"they are pooled into class '-1'")
        subtype = pat[args.label_col].fillna(-1).astype(int)
        strat = (subtype.astype(str) + '_' + pat['event'].astype(int).astype(str)).values
        print("\nstratifying on subtype x event:")
        names = {0: 'Oligodendroglioma', 1: 'Astrocytoma', 2: 'Glioblastoma', -1: 'Unlabelled'}
        for st in sorted(subtype.unique()):
            sub = pat[subtype == st]
            print(f"  {names.get(st, st):<20} n={len(sub):>3}  "
                  f"events={int(sub.event.sum()):>3}  "
                  f"censored={int((1 - sub.event).sum()):>3}")
        counts = pd.Series(strat).value_counts()
        if (counts < args.k).any():
            raise SystemExit(
                f"Stratum smaller than k={args.k}: "
                f"{counts[counts < args.k].to_dict()} - lower --k or drop --histology-csv.")
    else:
        subtype = None
        strat = pat['event'].values

    for fold, (tr_idx, te_idx) in enumerate(skf.split(cases, strat)):
        train = list(cases[tr_idx])
        test = list(cases[te_idx])

        if args.val_frac > 0:
            n_val = max(1, int(round(args.val_frac * len(train))))
            perm = rng.permutation(len(train))
            val = [train[i] for i in perm[:n_val]]
            train = [train[i] for i in perm[n_val:]]
        else:
            val = test                      # matches the existing convention

        n = max(len(train), len(val), len(test))
        out = pd.DataFrame({
            'train': train + [np.nan] * (n - len(train)),
            'val':   val + [np.nan] * (n - len(val)),
            'test':  test + [np.nan] * (n - len(test)),
        })
        path = os.path.join(args.out, f'splits_{fold}.csv')
        out.to_csv(path)

        idx = pat.set_index(args.case_col)
        ev_te = idx.loc[test, 'event'].sum()
        line = (f"  fold {fold}: train={len(train)} val={len(val)} test={len(test)} "
                f"(test events={int(ev_te)}")
        if subtype is not None:
            st_te = idx.loc[test, args.label_col].fillna(-1).astype(int).value_counts()
            line += ", test subtypes " + "/".join(
                f"{int(k)}:{int(st_te.get(k, 0))}" for k in sorted(st_te.index))
        print(line + ")")

    print(f"\nWrote {args.k} splits to {args.out}")


if __name__ == '__main__':
    main()
