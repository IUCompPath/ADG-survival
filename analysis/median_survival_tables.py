"""
Median overall survival by risk group, with 95% CIs and log-rank p-values.

Cleaned from the exploratory `try_sutv.py` in the original working repo.
Produces the median-OS tables (low/high, tertile, quartile) for the paper.

Usage:
    python analysis/median_survival_tables.py \
        --results results/tcga_gbmlgg_survival/mamba_mil/uni_v2_s1/combined_results.csv \
        --out     results/tables
"""

import os
import argparse
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

TIME, EVENT, RISK = 'survival', 'censorship', 'risk'


def clean_risk(df):
    """Strip bracket formatting from pickled risk scores; make times positive."""
    df = df.copy()
    df[RISK] = (df[RISK].astype(str)
                .str.replace('[', '', regex=False)
                .str.replace(']', '', regex=False)
                .astype(float))
    df[TIME] = df[TIME].astype(float).abs()
    return df


def median_os(group):
    """Return (median, ci_lower, ci_upper) in months; NaN where not reached."""
    kmf = KaplanMeierFitter()
    kmf.fit(group[TIME], event_observed=group[EVENT])
    med = kmf.median_survival_time_
    try:
        ci = kmf.confidence_interval_survival_function_

        def cross(series):
            below = series[series <= 0.5]
            return below.index[0] if not below.empty else np.nan

        lo = cross(ci[ci.columns[0]])
        hi = cross(ci[ci.columns[1]])
    except Exception:
        lo, hi = np.nan, np.nan
    return med, lo, hi


def fmt_median(med, lo, hi, decimals=1):
    def f(x):
        return f"{x:.{decimals}f}" if not np.isnan(x) else "NR"
    return f"{f(med)} ({f(lo)}-{f(hi)})"


def lr_p(g1, g2):
    return logrank_test(g1[TIME], g2[TIME],
                        event_observed_A=g1[EVENT],
                        event_observed_B=g2[EVENT]).p_value


def fmt_p(p):
    if p < 0.0001:
        exp = int(np.floor(np.log10(p)))
        return f"{p / 10 ** exp:.2f}e{exp}"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}"


def split_groups(df):
    """Median / tertile / quartile groups. Lower score = higher risk."""
    med = df[RISK].median()
    t33, t66 = df[RISK].quantile([1 / 3, 2 / 3])
    q25, q50, q75 = df[RISK].quantile([0.25, 0.50, 0.75])

    two = {'Low Risk': df[df[RISK] > med],
           'High Risk': df[df[RISK] <= med]}
    three = {'Low Risk': df[df[RISK] > t66],
             'Medium Risk': df[(df[RISK] > t33) & (df[RISK] <= t66)],
             'High Risk': df[df[RISK] <= t33]}
    four = {'Q1 - Low Risk': df[df[RISK] > q75],
            'Q2 - Medium Risk': df[(df[RISK] > q50) & (df[RISK] <= q75)],
            'Q3 - High Risk': df[(df[RISK] > q25) & (df[RISK] <= q50)],
            'Q4 - Very High Risk': df[df[RISK] <= q25]}
    return two, three, four


def build_table(groups, name):
    rows = []
    for label, g in groups.items():
        if g.empty:
            continue
        med, lo, hi = median_os(g)
        rows.append(dict(stratification=name, group=label, n=len(g),
                         events=int(g[EVENT].sum()),
                         median_os=fmt_median(med, lo, hi)))
    keys = list(groups)
    ref = groups[keys[0]]
    for label in keys[1:]:
        if not groups[label].empty and not ref.empty:
            rows.append(dict(stratification=name,
                             group=f"{keys[0]} vs {label}",
                             n='', events='',
                             median_os=f"log-rank p = {fmt_p(lr_p(ref, groups[label]))}"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True,
                    help='combined_results.csv with risk/survival/censorship')
    ap.add_argument('--out', default='results/tables')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = clean_risk(pd.read_csv(args.results))
    print(f"Loaded {len(df)} patients from {args.results}")

    two, three, four = split_groups(df)
    rows = (build_table(two, 'Median split')
            + build_table(three, 'Tertile')
            + build_table(four, 'Quartile'))

    out = pd.DataFrame(rows)
    path = os.path.join(args.out, 'median_survival_by_risk_group.csv')
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {path}")


if __name__ == '__main__':
    main()
