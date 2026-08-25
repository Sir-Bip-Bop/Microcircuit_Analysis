#!/usr/bin/env python3
"""
plot_ratio_vs_rate.py

Plots the measured f_low / f0 ratio as a function of background (input)
drive rate, one line per population, with a reference line at 0.5 (the
value predicted if f_low is a clean 2:1 subharmonic of f0).

Input: the summary CSV produced by analyze_subharmonic.py
(default name: subharmonic_summary.csv), which must contain the columns
    rate, pop_idx, pop_name, flow_over_f0_mean, flow_over_f0_sem

Usage
-----
    python plot_ratio_vs_rate.py --summary-csv subharmonic_summary.csv --out ratio_vs_rate.png

    # only excitatory populations, and only every 2nd rate (to de-clutter):
    python plot_ratio_vs_rate.py --pop-type exc --rate-stride 2
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

POP_ORDER = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]


def rate_to_numeric(s):
    """Extract a numeric drive value from a rate label like '9' or '9spks'."""
    m = re.search(r"[-+]?\d*\.?\d+", str(s))
    return float(m.group()) if m else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-csv", default="subharmonic_summary.csv")
    ap.add_argument("--out", default="ratio_vs_rate.png")
    ap.add_argument("--ref-line", type=float, default=0.5,
                     help="reference ratio to draw as a dashed horizontal line (default: %(default)s)")
    ap.add_argument("--pop-type", choices=["all", "exc", "inh"], default="all",
                     help="restrict to excitatory ('exc', names ending in E), "
                          "inhibitory ('inh', names ending in I), or 'all' (default: %(default)s)")
    ap.add_argument("--rate-stride", type=int, default=1,
                     help="plot every Nth rate (by sorted numeric order) to reduce clutter (default: %(default)s)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    df["rate_numeric"] = df["rate"].apply(rate_to_numeric)
    df = df.sort_values("rate_numeric")

    if args.pop_type == "exc":
        df = df[df["pop_name"].str.endswith("E")]
    elif args.pop_type == "inh":
        df = df[df["pop_name"].str.endswith("I")]

    if args.rate_stride > 1:
        kept_rates = sorted(df["rate_numeric"].unique())[::args.rate_stride]
        df = df[df["rate_numeric"].isin(kept_rates)]

    pops_present = [p for p in POP_ORDER if p in df["pop_name"].unique()]
    pops_present += [p for p in df["pop_name"].unique() if p not in pops_present]

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, pop in enumerate(pops_present):
        sub = df[df["pop_name"] == pop].sort_values("rate_numeric")
        if sub.empty:
            continue
        color = cmap(i % 10)
        yerr = sub["flow_over_f0_sem"] if "flow_over_f0_sem" in sub.columns else None
        ax.errorbar(
            sub["rate_numeric"], sub["flow_over_f0_mean"], yerr=yerr,
            marker="o", markersize=5, linewidth=1.5, capsize=3,
            color=color, label=pop,
        )

    ax.axhline(args.ref_line, linestyle="--", color="grey", linewidth=1,
               label=f"ratio = {args.ref_line:g}")

    ax.set_xlabel("Background rate (spikes/s)")
    ax.set_ylabel(r"$f_{low} / f_0$")
    ax.set_title(r"$f_{low}/f_0$ ratio vs. background drive")
    ax.legend(loc="best", fontsize=8, ncol=2, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()