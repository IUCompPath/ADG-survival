"""
Attention faithfulness by patch flipping, for survival MIL.

Protocol follows xMIL (Hense & Jamshidi Idaji et al., NeurIPS 2024),
https://github.com/bifold-pathomics/xMIL, adapted from classification to
survival: the tracked quantity is the model's risk score (and the cohort
c-index) rather than a target-class probability.

Three orderings are compared, which is what makes the result interpretable:

    morf    most relevant first  - drop highest-attention patches first
    morl    most relevant last   - drop lowest-attention patches first
    random  random baseline      - drop patches in random order

If attention is faithful, `morf` degrades the c-index fastest and `random`
sits between the two. Running `morf` alone cannot distinguish attention
quality from the effect of simply having fewer patches.

Usage:
    python analysis/patch_flipping.py \
        --features  features/uni_v2/pt_files \
        --ckpt-dir  results/tcga_gbmlgg_survival/mamba_mil/uni_v2_s1 \
        --survival  results/tcga_gbmlgg_survival/mamba_mil/uni_v2_s1/combined_results.csv \
        --in-dim    1536 \
        --out       results/faithfulness \
        --max-slides 5          # smoke test
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from lifelines.utils import concordance_index

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.MambaMIL import MambaMIL
from models.ABMIL import DAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(model_type, in_dim, **kw):
    if model_type == 'att_mil':
        return DAttention(in_dim, kw.get('n_classes', 4), dropout=0.0,
                          act='relu', survival=True)
    if model_type == 'mamba_mil':
        return MambaMIL(in_dim=in_dim, n_classes=kw.get('n_classes', 4),
                        dropout=0.0, act='gelu', survival=True,
                        layer=kw.get('layer', 2), rate=kw.get('rate', 5),
                        type=kw.get('mtype', 'SRMamba'))
    raise ValueError(f'unsupported --model-type {model_type}')


def load_models(ckpt_dir, in_dim, n_folds=10, model_type='mamba_mil', **kw):
    """Load all fold checkpoints once. in_dim must match training
    (UNI2/GigaPath = 1536, UNI/ResNet50 = 1024) or the state dict will not fit."""
    models = []
    for fold in range(n_folds):
        path = os.path.join(ckpt_dir, f's_{fold}_checkpoint.pt')
        if not os.path.exists(path):
            print(f"  missing {path}, skipping fold {fold}")
            continue
        model = build_model(model_type, in_dim, **kw).to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    if not models:
        raise RuntimeError(f"No checkpoints loaded from {ckpt_dir}")
    return models


@torch.no_grad()
def forward(model, feats):
    """Return (risk, attention) for one bag. feats: [1, n_patches, in_dim].

    DAttention returns None for attention when survival=True, so for ABMIL the
    attention logits are recomputed from the same feature/attention layers the
    forward pass uses."""
    # DAttention uses torch.mm and expects [n_patches, in_dim];
    # MambaMIL expects a batch dimension.
    is_abmil = hasattr(model, 'attention') and hasattr(model, 'feature')
    x = feats.squeeze(0) if (is_abmil and feats.dim() == 3) else feats

    hazards, S, Y_hat, A_raw, _ = model(x)
    risk = -torch.sum(S, dim=1).item()

    if A_raw is None and is_abmil:
        A_raw = model.attention(model.feature(x)).transpose(-1, -2)

    attn = A_raw.squeeze().detach().cpu().numpy()
    return risk, np.atleast_1d(attn)


# ── Survival labels ───────────────────────────────────────────────────────────

def load_survival(path):
    """Accept this repo's combined_results.csv schema and normalise it.
    Note censorship is 1 = censored, so event = 1 - censorship."""
    df = pd.read_csv(path)
    rename = {}
    for a, b in [('slide_id', 'slide'), ('slide_name', 'slide'),
                 ('survival', 'time'), ('survival_months', 'time'),
                 ('censorship', 'censorship'), ('event', 'event')]:
        if a in df.columns:
            rename[a] = b
    df = df.rename(columns=rename)
    if 'event' not in df.columns:
        df['event'] = 1 - df['censorship'].astype(float)
    df['time'] = df['time'].astype(float).abs()
    if 'risk' in df.columns:
        df['risk'] = (df['risk'].astype(str).str.strip('[]').astype(float))
    return df[['slide', 'time', 'event']].drop_duplicates(subset='slide')


# ── Flipping ──────────────────────────────────────────────────────────────────

def oof_assignment(ckpt_dir, n_folds):
    """Map case_id -> fold index for which that patient was held out.

    Each checkpoint saw ~80% of the cohort during training, so scoring a slide
    with all folds and averaging mixes in-sample predictions into the result.
    Attention and risk are therefore taken only from the fold that held the
    patient out, matching how the reported cross-validated c-index is computed.
    """
    import csv
    mapping = {}
    for fold in range(n_folds):
        path = os.path.join(ckpt_dir, f'splits_{fold}.csv')
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        if 'test' not in df.columns:
            return None
        for case in df['test'].dropna().astype(str):
            mapping[case] = fold
    return mapping


def rank(attn, order, rng):
    """Index order in which patches are REMOVED."""
    if order == 'random':
        idx = np.arange(len(attn)); rng.shuffle(idx); return idx
    if order == 'morf':
        return np.argsort(-attn)          # highest attention removed first
    if order == 'morl':
        return np.argsort(attn)           # lowest attention removed first
    raise ValueError(order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True, help='dir of per-slide .pt features')
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--survival', required=True)
    ap.add_argument('--in-dim', type=int, required=True,
                    help='feature dim used at training time (UNI2=1536)')
    ap.add_argument('--out', default='results/faithfulness')
    ap.add_argument('--steps', type=int, default=5,
                    help='removal step size in %% (default 5 -> 0,5,...,50)')
    ap.add_argument('--percents', type=float, nargs='+', default=None,
                    help='explicit removal percentages, e.g. 0 1 2 5 10 20 50. '
                         'Overrides --steps/--max-percent.')
    ap.add_argument('--full-curve', action='store_true',
                    help="xMIL's '1%%-of-all' protocol: 1%% steps from 0 to 100, "
                         'every patient, down to an empty bag')
    ap.add_argument('--n-random', type=int, default=1,
                    help='number of random orderings to average for the random '
                         'baseline; a single draw is noisy when effects are small')
    ap.add_argument('--max-percent', type=int, default=50)
    ap.add_argument('--orders', nargs='+', default=['morf', 'morl', 'random'])
    ap.add_argument('--max-slides', type=int, default=None,
                    help='random subset of this many PATIENTS (smoke test only)')
    ap.add_argument('--n-folds', type=int, default=10)
    ap.add_argument('--model-type', choices=['att_mil', 'mamba_mil'],
                    default='mamba_mil', help='must match the checkpoints')
    ap.add_argument('--all-folds', action='store_true',
                    help='average every fold per slide instead of using only the '
                         'held-out fold; reports in-sample performance')
    ap.add_argument('--cache', choices=['none', 'cpu', 'gpu'], default='none',
                    help="preload every patient bag into RAM ('cpu') or GPU memory "
                         "('gpu') before the sweep. The full cohort is ~72 GB in "
                         "fp32, so 'gpu' needs --half on a 48 GB card.")
    ap.add_argument('--half', action='store_true',
                    help='store cached features as fp16 (halves memory; features '
                         'are cast back to fp32 at the model boundary)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    surv = load_survival(args.survival)
    surv_map = surv.set_index('slide')[['time', 'event']].to_dict('index')

    print(f"Loading {args.n_folds} {args.model_type} checkpoints (in_dim={args.in_dim}) ...")
    models = load_models(args.ckpt_dir, args.in_dim, args.n_folds,
                         model_type=args.model_type)
    print(f"  loaded {len(models)} fold models on {DEVICE}")

    oof = None if args.all_folds else oof_assignment(args.ckpt_dir, args.n_folds)
    if oof:
        print(f"  out-of-fold scoring: {len(oof)} patients mapped to their held-out fold")
    elif args.all_folds:
        print("  --all-folds: averaging every fold (includes in-sample predictions)")
    else:
        print("  WARNING: splits_*.csv not found next to the checkpoints; "
              "falling back to averaging all folds (in-sample contamination)")

    def to_case(stem):
        """Feature files are full slide barcodes
        (TCGA-02-0001-01Z-00-DX1.<uuid>); the survival table is keyed by the
        12-character patient barcode (TCGA-02-0001). Try progressively
        shorter forms so either convention works."""
        for cand in (stem, stem.split('.')[0], '-'.join(stem.split('-')[:3])):
            if cand in surv_map:
                return cand
        return None

    # The model is trained on PATIENT-level bags: dataset_survival.py
    # concatenates every slide belonging to a patient before the forward pass.
    # Scoring slides individually would both mismatch that input and count a
    # multi-slide patient several times against one survival label.
    by_case = {}
    for f in sorted(glob.glob(os.path.join(args.features, 'TCGA-*.pt'))):
        case = to_case(os.path.basename(f)[:-3])
        if case is not None:
            by_case.setdefault(case, []).append(f)

    cases = sorted(by_case)
    if args.max_slides and args.max_slides < len(cases):
        # Random subset, not the alphabetically first N: TCGA barcodes encode
        # tissue source site, so a head slice is biased toward a few institutions.
        cases = sorted(rng.choice(cases, args.max_slides, replace=False).tolist())
    print(f"  {len(cases)} patients "
          f"({sum(len(by_case[c]) for c in cases)} slides) with survival labels")
    if not cases:
        raise SystemExit('No patients matched the survival table - check slide id formatting.')

    if args.full_curve:
        # xMIL default strategy '1%-of-all': np.arange(1, 101, 1), plus the
        # unperturbed baseline at 0.
        percents = list(range(0, 101))
    elif args.percents:
        percents = sorted(set(args.percents))
    else:
        percents = list(range(0, args.max_percent + 1, args.steps))
    records = {o: {p: [] for p in percents} for o in args.orders}

    # Optional preload. Each patient's bag is read once either way, so caching
    # only helps when the same bag is revisited - it is kept for runs that sweep
    # the same cohort repeatedly, and to take the filesystem out of the loop.
    cache = {}
    if args.cache != 'none':
        dev = DEVICE if args.cache == 'gpu' else torch.device('cpu')
        dtype = torch.float16 if args.half else torch.float32
        total = 0
        for case in tqdm(cases, desc=f'preload->{args.cache}'):
            bags = []
            for f in by_case[case]:
                x = torch.load(f, map_location='cpu')
                if isinstance(x, dict):
                    x = x.get('features', next(iter(x.values())))
                x = x.float()
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                bags.append(x)
            bag = torch.cat(bags, dim=0).to(dtype).to(dev)
            cache[case] = bag
            total += bag.element_size() * bag.nelement()
        print(f'  cached {len(cache)} patient bags on {args.cache} '
              f'({total / 1e9:.1f} GB, {dtype})')

    for case in tqdm(cases, desc='patients'):
        meta = surv_map[case]

        if case in cache:
            feats = cache[case].to(DEVICE).float().unsqueeze(0)
        else:
            bags = []
            for f in by_case[case]:
                x = torch.load(f, map_location=DEVICE)
                if isinstance(x, dict):
                    x = x.get('features', next(iter(x.values())))
                x = x.float()
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                bags.append(x)
            feats = torch.cat(bags, dim=0).to(DEVICE).unsqueeze(0)
        n_patches = feats.shape[1]

        # Use the fold that held this patient out, so both the attention
        # ranking and the risk score are out-of-sample.
        if oof is not None and case in oof and oof[case] < len(models):
            fold_models = [models[oof[case]]]
        else:
            fold_models = models

        _, base_attn = forward(fold_models[0], feats)
        if len(base_attn) != n_patches:
            base_attn = np.resize(base_attn, n_patches)

        for order in args.orders:
            n_draws = args.n_random if order == 'random' else 1
            orderings = [rank(base_attn, order, rng) for _ in range(n_draws)]
            for p in percents:
                n_rm = int(p / 100 * n_patches)
                vals = []
                for removal in orderings:
                    keep = np.sort(removal[n_rm:])
                    if keep.size == 0:
                        # Bag fully removed. xMIL scores an all-zero bag here
                        # rather than dropping the patient, so the curve stays
                        # defined on the same cohort at 100%.
                        sub = torch.zeros_like(feats[:, :1, :])
                    else:
                        sub = feats[:, keep, :]
                    vals.extend(forward(m, sub)[0] for m in fold_models)
                if vals:
                    records[order][p].append((float(np.mean(vals)),
                                              meta['time'], meta['event']))

    rows = []
    for order in args.orders:
        for p in percents:
            vals = records[order][p]
            if len(vals) < 2:
                continue
            r, t, e = zip(*vals)
            if np.sum(e) == 0:
                continue
            rows.append(dict(order=order, percent_removed=p, n_slides=len(vals),
                             c_index=concordance_index(t, -np.array(r), e)))

    out = pd.DataFrame(rows)
    path = os.path.join(args.out, 'patch_flipping_cindex.csv')
    out.to_csv(path, index=False)
    print('\n' + out.to_string(index=False))
    print(f"\nWrote {path}")

    if not out.empty:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for order, g in out.groupby('order'):
                ax.plot(g.percent_removed, g.c_index, marker='o', label=order)
            ax.set_xlabel('Top-% patches removed')
            ax.set_ylabel('C-index')
            ax.set_title('Attention faithfulness (patch flipping)')
            ax.grid(alpha=0.4); ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, 'patch_flipping_curve.png'), dpi=300)
            print(f"Wrote {os.path.join(args.out, 'patch_flipping_curve.png')}")
        except Exception as exc:
            print(f"(plot skipped: {exc})")


if __name__ == '__main__':
    main()
