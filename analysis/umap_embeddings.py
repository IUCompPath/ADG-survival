"""
UMAP projection of slide-level embeddings, coloured by risk group and WHO 2021 type.

Each slide's patch features are pooled to a single vector, projected to 2-D with
UMAP, and plotted twice: once by predicted risk group (low/high) and once by
WHO 2021 subtype. Comparing the two panels shows whether the embedding separates
risk independently of subtype, or whether the apparent risk separation is mostly
subtype structure.

Two pooling functions are available:
  mean    unweighted mean over patches - the raw feature geometry
  gated   gated-attention pooling, so the plot reflects what the MIL model attends to

Usage:
    python analysis/umap_embeddings.py \
        --features  /path/to/pt_files \
        --results   results/.../combined_results.csv \
        --histology dataset_csv/df_2021_histology_labels.csv \
        --in-dim    1536 \
        --out       results/umap
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SUBTYPE = {0: 'Oligodendroglioma', 1: 'Astrocytoma', 2: 'Glioblastoma'}


class AttnNetGated(nn.Module):
    """Gated attention (Ilse et al. 2018), as used by the ABMIL aggregator."""

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        a = [nn.Linear(L, D), nn.Tanh()]
        b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            a.append(nn.Dropout(0.25))
            b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*a)
        self.attention_b = nn.Sequential(*b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        return self.attention_c(self.attention_a(x) * self.attention_b(x))


def pool_mean(x, _=None):
    return x.mean(dim=0)


def pool_gated(x, attn_net):
    with torch.no_grad():
        w = torch.softmax(attn_net(x).squeeze(-1), dim=0)
    return (w.unsqueeze(-1) * x).sum(dim=0)


def case_of(stem):
    """Patient barcode from a slide filename (TCGA-XX-YYYY-...)."""
    return '-'.join(stem.split('-')[:3])


def load_labels(results_csv, histology_csv):
    """Return {case_id: (risk_group, subtype)} from the model output + WHO labels."""
    r = pd.read_csv(results_csv)
    col = 'slide_id' if 'slide_id' in r.columns else r.columns[0]
    r['case_id'] = r[col].astype(str).map(case_of)
    r['risk'] = (r['risk'].astype(str).str.strip('[]').astype(float)
                 if 'risk' in r.columns else np.nan)
    r = r.drop_duplicates(subset='case_id')
    med = r['risk'].median()
    # Lower score = higher risk, matching the KM/stratification scripts.
    r['risk_group'] = np.where(r['risk'] <= med, 'High risk', 'Low risk')

    h = pd.read_csv(histology_csv).drop_duplicates(subset='case_id')
    m = r.merge(h[['case_id', 'label']], on='case_id', how='left')
    return {row.case_id: (row.risk_group, SUBTYPE.get(row.label, 'Unknown'))
            for row in m.itertuples()}


def embed_slides(feature_dir, labels, pooling, in_dim, max_slides=None):
    files = sorted(glob.glob(os.path.join(feature_dir, 'TCGA-*.pt')))
    files = [f for f in files if case_of(os.path.basename(f)[:-3]) in labels]
    if max_slides:
        files = files[:max_slides]
    if not files:
        raise SystemExit('No feature files matched the label table.')

    attn = None
    if pooling == 'gated':
        attn = AttnNetGated(L=in_dim, D=256).to(DEVICE).eval()
        print('  gated attention is randomly initialised: it shows the pooling '
              'geometry, not a trained model\'s attention')

    fn = pool_gated if pooling == 'gated' else pool_mean
    vecs, rows = [], []
    for i, f in enumerate(files, 1):
        x = torch.load(f, map_location=DEVICE)
        if isinstance(x, dict):
            x = x.get('features', next(iter(x.values())))
        x = x.float().to(DEVICE)
        vecs.append(fn(x, attn).cpu().numpy())
        case = case_of(os.path.basename(f)[:-3])
        rows.append((case,) + labels[case])
        if i % 100 == 0:
            print(f'  {i}/{len(files)} slides')

    meta = pd.DataFrame(rows, columns=['case_id', 'risk_group', 'subtype'])
    return np.vstack(vecs), meta


def scatter(ax, emb, series, title, palette):
    for name, colour in palette.items():
        m = (series == name).values
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.7,
                       c=colour, label=f'{name} (n={int(m.sum())})')
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=9, markerscale=1.6, loc='best', frameon=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--results', required=True,
                    help='combined_results.csv with per-patient risk scores')
    ap.add_argument('--histology', required=True)
    ap.add_argument('--in-dim', type=int, required=True)
    ap.add_argument('--out', default='results/umap')
    ap.add_argument('--pooling', choices=['mean', 'gated'], default='mean')
    ap.add_argument('--max-slides', type=int, default=None)
    ap.add_argument('--n-neighbors', type=int, default=15)
    ap.add_argument('--min-dist', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    try:
        import umap
    except ImportError:
        raise SystemExit('pip install umap-learn')

    os.makedirs(args.out, exist_ok=True)

    labels = load_labels(args.results, args.histology)
    print(f'{len(labels)} patients with risk group and subtype')

    X, meta = embed_slides(args.features, labels, args.pooling,
                           args.in_dim, args.max_slides)
    print(f'embedded {X.shape[0]} slides, dim {X.shape[1]}')

    emb = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                    random_state=args.seed).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    scatter(axes[0], emb, meta.risk_group, 'By predicted risk group',
            {'Low risk': 'tab:blue', 'High risk': 'tab:red'})
    scatter(axes[1], emb, meta.subtype, 'By WHO 2021 type',
            {'Oligodendroglioma': 'tab:blue', 'Astrocytoma': 'tab:green',
             'Glioblastoma': 'tab:red', 'Unknown': 'lightgrey'})
    fig.suptitle(f'UMAP of slide embeddings ({args.pooling} pooling)', fontsize=15)
    fig.tight_layout()

    png = os.path.join(args.out, f'umap_{args.pooling}.png')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(png.replace('.png', '.tiff'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    out = meta.copy()
    out['umap_1'], out['umap_2'] = emb[:, 0], emb[:, 1]
    out.to_csv(os.path.join(args.out, f'umap_{args.pooling}_coords.csv'), index=False)
    print(f'wrote {png} (+ .tiff, + coords CSV)')


if __name__ == '__main__':
    main()
