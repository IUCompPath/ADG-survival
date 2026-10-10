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
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from lifelines.utils import concordance_index

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.MambaMIL import MambaMIL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────

def load_models(ckpt_dir, in_dim, n_folds=10, **kw):
    """Load all fold checkpoints once. in_dim must match training
    (UNI2 = 1536, UNI/ResNet50 = 1024) or the state dict will not fit."""
    models = []
    for fold in range(n_folds):
        path = os.path.join(ckpt_dir, f's_{fold}_checkpoint.pt')
        if not os.path.exists(path):
            print(f"  missing {path}, skipping fold {fold}")
            continue
        model = MambaMIL(in_dim=in_dim, n_classes=kw.get('n_classes', 4),
                         dropout=0.0, act='gelu', survival=True,
                         layer=kw.get('layer', 2), rate=kw.get('rate', 5),
                         type=kw.get('mtype', 'SRMamba')).to(DEVICE)
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
    """Return (risk, attention) for one bag. feats: [1, n_patches, in_dim]."""
    hazards, S, Y_hat, A_raw, _ = model(feats)
    risk = -torch.sum(S, dim=1).item()
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
    ap.add_argument('--max-percent', type=int, default=50)
    ap.add_argument('--orders', nargs='+', default=['morf', 'morl', 'random'])
    ap.add_argument('--max-slides', type=int, default=None,
                    help='limit slides (use for a smoke test)')
    ap.add_argument('--n-folds', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    surv = load_survival(args.survival)
    surv_map = surv.set_index('slide')[['time', 'event']].to_dict('index')

    print(f"Loading {args.n_folds} checkpoints (in_dim={args.in_dim}) ...")
    models = load_models(args.ckpt_dir, args.in_dim, args.n_folds)
    print(f"  loaded {len(models)} models on {DEVICE}")

    def to_case(stem):
        """Feature files are full slide barcodes
        (TCGA-02-0001-01Z-00-DX1.<uuid>); the survival table is keyed by the
        12-character patient barcode (TCGA-02-0001). Try progressively
        shorter forms so either convention works."""
        for cand in (stem, stem.split('.')[0], '-'.join(stem.split('-')[:3])):
            if cand in surv_map:
                return cand
        return None

    pt_files = sorted(f for f in os.listdir(args.features) if f.endswith('.pt'))
    pt_files = [f for f in pt_files if to_case(f[:-3]) is not None]
    if args.max_slides:
        pt_files = pt_files[:args.max_slides]
    print(f"  {len(pt_files)} slides with survival labels")
    if not pt_files:
        raise SystemExit("No slides matched the survival table - check slide id formatting.")

    percents = list(range(0, args.max_percent + 1, args.steps))
    records = {o: {p: [] for p in percents} for o in args.orders}

    for pt in tqdm(pt_files, desc='slides'):
        case = to_case(pt[:-3])
        if case is None:
            continue
        meta = surv_map[case]

        feats = torch.load(os.path.join(args.features, pt), map_location=DEVICE)
        if isinstance(feats, dict):
            feats = feats.get('features', next(iter(feats.values())))
        feats = feats.float()
        if feats.ndim == 2:
            feats = feats.unsqueeze(0)
        feats = feats.to(DEVICE)
        n_patches = feats.shape[1]

        # Patch ranking comes from fold 0, as in the original experiment.
        _, base_attn = forward(models[0], feats)
        if len(base_attn) != n_patches:          # guard against pooled attention
            base_attn = np.resize(base_attn, n_patches)

        for order in args.orders:
            removal = rank(base_attn, order, rng)
            for p in percents:
                n_rm = int(p / 100 * n_patches)
                keep = np.sort(removal[n_rm:])
                if keep.size == 0:
                    continue
                sub = feats[:, keep, :]
                risks = [forward(m, sub)[0] for m in models]
                records[order][p].append((float(np.mean(risks)),
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
