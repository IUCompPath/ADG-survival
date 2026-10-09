"""
Recreate the KM curves for the ADG survival paper with larger fonts,
in particular the at-risk / censored table printed below each plot.

Cohorts : all, oligo (label 0), astro (label 1), gbm (label 2)
Splits  : low vs high (median), tertile, quartile
Output  : results/new/tcga_gbmlgg_survival/  (PNG + TIFF)

Usage:  python plot_km_curves.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.environ.get('ADG_ROOT', '.')
RESULT_CSV = os.environ.get('ADG_RESULT_CSV',
    f'{BASE_DIR}/results/tcga_gbmlgg_survival/mamba_mil/uni_v2_s1/combined_results.csv')
HIST_CSV   = f'{BASE_DIR}/dataset_csv/df_2021_histology_labels.csv'
SAVE_DIR   = os.environ.get('ADG_FIG_DIR',
    f'{BASE_DIR}/results/tcga_gbmlgg_survival/km_curves/') 
PNG_DIR    = f'{SAVE_DIR}png/'
TIFF_DIR   = f'{SAVE_DIR}tiff/'

time_col, event_col, risk_col = 'survival', 'censorship', 'risk'

# ── Font sizes (bumped up; at-risk table was the main complaint) ──────────────

FS_AXIS_LABEL = 22    # "Time (months)" / "Survival Probability"
FS_TICKS      = 20
FS_LEGEND     = 20
FS_PVAL       = 15    # p-value annotation, inside the plot, no box
FS_AT_RISK    = 18    # was 12 in the original figures

FIGSIZE = (10, 9)

LABEL_META = {
    0: dict(name='Oligodendroglioma', short='oligo'),
    1: dict(name='Astrocytoma',       short='astro'),
    2: dict(name='Glioblastoma',      short='gbm'),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_risk(df):
    df = df.copy()
    df[risk_col] = (df[risk_col].astype(str)
                    .str.replace('[', '', regex=False)
                    .str.replace(']', '', regex=False)
                    .astype(float))
    df[time_col] = df[time_col].abs()
    return df

def fmt_p(p):
    if p < 0.0001:  return "p < 0.0001"
    if p < 0.001:   return f"p = {p:.4f}"
    return f"p = {p:.3f}"

def fit_km(group, label):
    kmf = KaplanMeierFitter()
    kmf.fit(group[time_col], event_observed=group[event_col], label=label)
    return kmf

def lr_p(g1, g2):
    return logrank_test(g1[time_col], g2[time_col],
                        event_observed_A=g1[event_col],
                        event_observed_B=g2[event_col]).p_value

def get_hr(df, group_col):
    cph = CoxPHFitter()
    cox_df = df[[time_col, event_col, group_col]].dropna()
    cph.fit(cox_df, duration_col=time_col, event_col=event_col)
    hr    = np.exp(cph.params_[group_col])
    ci_lo = np.exp(cph.confidence_intervals_.loc[group_col].iloc[0])
    ci_hi = np.exp(cph.confidence_intervals_.loc[group_col].iloc[1])
    return hr, ci_lo, ci_hi, cph.summary.loc[group_col, 'p']

def style_ax(ax):
    ax.set_xlabel("")   # drawn manually in finish(), below the at-risk table
    ax.set_ylabel("Survival Probability", fontsize=FS_AXIS_LABEL)
    ax.tick_params(labelsize=FS_TICKS)
    ax.legend(fontsize=FS_LEGEND, loc='upper right')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.4)

def pval_box(ax, text, x=None, y=None):
    """Stash the p-value text; finish() places it inside the axes, unboxed,
    in whichever corner the curves leave empty."""
    ax._pval_text = text

def place_pvalues(ax, text):
    """Draw `text` inside the axes with no bounding box, in the candidate
    corner whose region contains the fewest plotted points."""
    lines = text.split('\n')
    # Rough extent of the text block in axes fraction, from the font size.
    bbox_ax = ax.get_window_extent()
    w = max(len(l) for l in lines) * FS_PVAL * 0.60 / bbox_ax.width
    h = len(lines) * FS_PVAL * 1.45 / bbox_ax.height

    # (x, y, ha, va) anchors: lower-left, lower-right, upper-left, upper-right.
    pad = 0.03
    candidates = [
        (pad,     pad,     'left',  'bottom'),
        (1 - pad, pad,     'right', 'bottom'),
        (pad,     1 - pad, 'left',  'top'),
        (1 - pad, 1 - pad, 'right', 'top'),
    ]

    # Curve vertices in axes coordinates.
    pts = []
    for line in ax.get_lines():
        xd, yd = line.get_xdata(), line.get_ydata()
        if len(xd) == 0:
            continue
        disp = ax.transData.transform(np.column_stack([xd, yd]))
        pts.append(ax.transAxes.inverted().transform(disp))
    pts = np.vstack(pts) if pts else np.empty((0, 2))

    # The legend also occupies space - treat it as a blocked region.
    blocked = []
    leg = ax.get_legend()
    if leg is not None:
        lb = leg.get_window_extent()
        lb = ax.transAxes.inverted().transform([[lb.x0, lb.y0], [lb.x1, lb.y1]])
        blocked.append(lb)

    def cost(x, y, ha, va):
        x0 = x if ha == 'left' else x - w
        y0 = y if va == 'bottom' else y - h
        x1, y1 = x0 + w, y0 + h
        n = np.sum((pts[:, 0] >= x0 - 0.02) & (pts[:, 0] <= x1 + 0.02) &
                   (pts[:, 1] >= y0 - 0.02) & (pts[:, 1] <= y1 + 0.02)) if len(pts) else 0
        for (bx0, by0), (bx1, by1) in blocked:      # heavy penalty for legend
            if x0 < bx1 and x1 > bx0 and y0 < by1 and y1 > by0:
                n += 10000
        return n

    x, y, ha, va = min(candidates, key=lambda c: cost(*c))
    ax.text(x, y, text, transform=ax.transAxes, fontsize=FS_PVAL,
            ha=ha, va=va, linespacing=1.45)

def finish(fig, ax, kmfs, stem):
    """Add the enlarged at-risk table + p-value box and save PNG + TIFF."""
    # p-values go inside the plot, unboxed, in the emptiest corner.
    txt = getattr(ax, '_pval_text', None)
    if txt:
        fig.canvas.draw()
        place_pvalues(ax, txt)

    add_at_risk_counts(*kmfs, ax=ax, fontsize=FS_AT_RISK,
                       rows_to_show=['At risk', 'Censored'])
    plt.tight_layout()

    # Reserve a strip under the at-risk table for the x-label and p-value box.
    # bbox_inches='tight' crops to content, so figure-fraction placement below
    # y=0 is stable and never collides with the curves, legend, or counts.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv  = fig.transFigure.inverted()

    # Bottom of everything currently drawn (axes + at-risk block).
    y_low = min(inv.transform(a.get_tightbbox(rend))[0][1] for a in fig.axes)

    ax_bb = ax.get_position()

    fig.text(ax_bb.x0 + ax_bb.width / 2, y_low - 0.025, "Time (months)",
             ha='center', va='top', fontsize=FS_AXIS_LABEL)


    fig.savefig(f"{PNG_DIR}{stem}.png",  dpi=300, bbox_inches='tight')
    fig.savefig(f"{TIFF_DIR}{stem}.tiff", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {stem}.png / .tiff")

# ── Risk grouping ─────────────────────────────────────────────────────────────
# NOTE: risk score is inverted here (higher score -> better survival), matching
# the original figures: the lowest-score bin is labelled "High Risk".

def split_groups(df):
    """Return dicts of the median / tertile / quartile subgroups."""
    median_risk = df[risk_col].median()
    t33, t66    = df[risk_col].quantile([1/3, 2/3])
    q25, q50, q75 = df[risk_col].quantile([0.25, 0.50, 0.75])

    two = {
        'Low Risk' : df[df[risk_col] >  median_risk],
        'High Risk': df[df[risk_col] <= median_risk],
    }
    three = {
        'Low Risk'   : df[df[risk_col] >  t66],
        'Medium Risk': df[(df[risk_col] > t33) & (df[risk_col] <= t66)],
        'High Risk'  : df[df[risk_col] <= t33],
    }
    four = {
        'Q1 - Low Risk'      : df[df[risk_col] >  q75],
        'Q2 - Medium Risk'   : df[(df[risk_col] > q50) & (df[risk_col] <= q75)],
        'Q3 - High Risk'     : df[(df[risk_col] > q25) & (df[risk_col] <= q50)],
        'Q4 - Very High Risk': df[df[risk_col] <= q25],
    }
    return two, three, four, median_risk

# ── One cohort → 3 figures ────────────────────────────────────────────────────

def plot_cohort(df, short, name):
    print(f"\n{'='*60}\n  {name}   n = {len(df)}\n{'='*60}")
    if df.empty:
        print("  empty — skipping"); return

    two, three, four, median_risk = split_groups(df)

    g_lo, g_hi = two['Low Risk'], two['High Risk']
    t1, t2, t3 = three['Low Risk'], three['Medium Risk'], three['High Risk']
    q1, q2, q3, q4 = (four['Q1 - Low Risk'], four['Q2 - Medium Risk'],
                      four['Q3 - High Risk'], four['Q4 - Very High Risk'])

    print(f"  Low/High : {len(g_lo)} / {len(g_hi)}")
    print(f"  Tertiles : {len(t1)} / {len(t2)} / {len(t3)}")
    print(f"  Quartiles: {len(q1)} / {len(q2)} / {len(q3)} / {len(q4)}")

    # ── PLOT 1 — Low vs High (median split) ──────────────────────────────────
    km_lo, km_hi = fit_km(g_lo, 'Low Risk'), fit_km(g_hi, 'High Risk')
    p_lh = lr_p(g_lo, g_hi)

    try:
        d = df.copy()
        d['grp2_code'] = (d[risk_col] <= median_risk).astype(int)   # 1 = high risk
        hr, lo, hi, p_cox = get_hr(d, 'grp2_code')
        hr_text = fmt_p(p_cox)
        print(f"  Cox HR: {hr:.2f} [{lo:.2f}-{hi:.2f}], {fmt_p(p_cox)}")
    except Exception as e:
        hr_text = fmt_p(p_lh)
        print(f"  Cox failed ({e}) — using log-rank p")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    km_lo.plot_survival_function(ax=ax, ci_show=False, color='blue')
    km_hi.plot_survival_function(ax=ax, ci_show=False, color='red')
    pval_box(ax, hr_text, y=0.22)
    style_ax(ax)
    finish(fig, ax, (km_lo, km_hi),
           'low_vs_high_risk_survival' if short == 'all' else f'{short}_low_vs_high_survival')

    # ── PLOT 2 — Tertile ─────────────────────────────────────────────────────
    km_t1, km_t2, km_t3 = (fit_km(t1, 'Low Risk'), fit_km(t2, 'Medium Risk'),
                           fit_km(t3, 'High Risk'))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    km_t1.plot_survival_function(ax=ax, ci_show=False, color='blue')
    km_t2.plot_survival_function(ax=ax, ci_show=False, color='green')
    km_t3.plot_survival_function(ax=ax, ci_show=False, color='red')
    pval_box(ax,
             f"Low vs Med:  {fmt_p(lr_p(t1, t2))}\n"
             f"Med vs High: {fmt_p(lr_p(t2, t3))}\n"
             f"Low vs High: {fmt_p(lr_p(t1, t3))}", y=0.34)
    style_ax(ax)
    finish(fig, ax, (km_t1, km_t2, km_t3),
           'tertile_risk_group_survival' if short == 'all' else f'{short}_tertile_survival')

    # ── PLOT 3 — Quartile ────────────────────────────────────────────────────
    km_q1, km_q2, km_q3, km_q4 = (fit_km(q1, 'Q1 - Low Risk'),
                                  fit_km(q2, 'Q2 - Medium Risk'),
                                  fit_km(q3, 'Q3 - High Risk'),
                                  fit_km(q4, 'Q4 - Very High Risk'))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    km_q1.plot_survival_function(ax=ax, ci_show=False, color='blue')
    km_q2.plot_survival_function(ax=ax, ci_show=False, color='green')
    km_q3.plot_survival_function(ax=ax, ci_show=False, color='orange')
    km_q4.plot_survival_function(ax=ax, ci_show=False, color='red')
    pval_box(ax,
             f"Q1 vs Q2: {fmt_p(lr_p(q1, q2))}\n"
             f"Q2 vs Q3: {fmt_p(lr_p(q2, q3))}\n"
             f"Q3 vs Q4: {fmt_p(lr_p(q3, q4))}\n"
             f"Q1 vs Q4: {fmt_p(lr_p(q1, q4))}", y=0.40)
    style_ax(ax)
    finish(fig, ax, (km_q1, km_q2, km_q3, km_q4),
           'quartile_risk_group_survival' if short == 'all' else f'{short}_quartile_survival')

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PNG_DIR, exist_ok=True)
    os.makedirs(TIFF_DIR, exist_ok=True)

    combined_df = clean_risk(pd.read_csv(RESULT_CSV))

    # Whole cohort
    plot_cohort(combined_df, 'all', 'All gliomas')

    # Per-subtype: attach the 2021 WHO histology labels
    hist = pd.read_csv(HIST_CSV).drop_duplicates(subset='case_id', keep='first')
    merged_df = pd.merge(combined_df, hist[['case_id', 'label']],
                         left_on='slide_id', right_on='case_id', how='inner')
    merged_df = clean_risk(merged_df)

    for label, meta in LABEL_META.items():
        plot_cohort(merged_df[merged_df['label'] == label].copy(),
                    meta['short'], meta['name'])

    print(f"\nDone — 12 figures written to {SAVE_DIR} (png/ and tiff/)")

if __name__ == '__main__':
    main()
