#!/usr/bin/env python3
"""
pathway_split.py

Motivation
----------
The transfer entropy matrix contains directed pairwise interactions, but the
poster's framing is about a signal being received, processed and transmitted
through the column. That requires asking not how strongly populations interact
in general, but whether interaction along the canonical feedforward route
behaves differently from interaction along the feedback route as the drive
varies.

This uses data already computed: no new simulation, no new estimation. It
partitions the existing 8x8 effective TE matrix at each drive into named
pathways and follows each across the sweep.

A caution worth carrying into any caption. The delay profiles show that the
strong, well-localised transfer in this circuit is the within-layer excitatory-
inhibitory loop in L4, not the inter-laminar pathway. The feedforward and
feedback pairs sit in the weaker part of the matrix, so a flat result here is a
real possibility and would mean the drive dependence lives in local E-I
interaction rather than in laminar propagation. That is a finding, not a failed
panel, and should be reported as such rather than dropped.

What it measures
----------------
For each drive, the mean effective TE over the pairs in each pathway, with the
standard error across pairs, plus the feedforward-to-feedback ratio. Pathways
are given as explicit pair lists so the choice is visible and arguable rather
than buried.

Default pathways, following the canonical cortical route:

  feedforward   L4E->L23E, L23E->L5E, L5E->L6E
  feedback      L6E->L4E, L5E->L23E, L23E->L4E
  local E-I     L4E->L4I, L4I->L4E, L23E->L23I, L23I->L23E,
                L5E->L5I, L5I->L5E, L6E->L6I, L6I->L6E

The local E-I set is included as a reference: it is where the strong transfer
actually is, so plotting it alongside shows whether any pathway effect is
distinguishable from the dominant local one, or merely a diluted copy of it.

It also reports, per pathway and drive, the delay at which each constituent pair
peaks. If the feedforward latencies accumulate along the route, that is
propagation; if they do not, the pathway sum is not describing a travelling
signal and should not be described as one.

Usage
-----
    python pathway_split.py --out-dir ./te_out --tag full \\
        --min-drive 7.8 --out pathway_split.png

    # custom pathways
    python pathway_split.py --out-dir ./te_out --tag full \\
        --pathway "feedforward=L4E->L23E,L23E->L5E" \\
        --pathway "feedback=L5E->L23E,L23E->L4E"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

DEFAULT_PATHWAYS = [
    ("feedforward", ["L4E->L23E", "L23E->L5E", "L5E->L6E"]),
    ("feedback", ["L6E->L4E", "L5E->L23E", "L23E->L4E"]),
    ("local E-I", ["L4E->L4I", "L4I->L4E", "L23E->L23I", "L23I->L23E",
                   "L5E->L5I", "L5I->L5E", "L6E->L6I", "L6I->L6E"]),
]

PATHWAY_COLOURS = {"feedforward": "#0072B2", "feedback": "#D55E00",
                   "local E-I": "#009E73"}


def drive_value(d):
    m = re.search(r"[\d.]+", str(d))
    return float(m.group()) if m else np.nan


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="./te_out")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--min-drive", type=float, default=None)
    ap.add_argument("--pathway", action="append", default=None,
                    help='name=SRC->TGT,SRC->TGT  (repeatable)')
    ap.add_argument("--out", default="pathway_split.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(args.out_dir, "meta.json")) as fh:
        meta = json.load(fh)
    pops = meta["pops"]
    bin_ms = meta.get("bin_ms", 1.0)

    rows = [r for r in np.load(os.path.join(args.out_dir,
                                            f"te_{args.tag}.npy"),
                               allow_pickle=True)
            if np.isfinite(r.get("te_eff", np.nan))]
    if args.min_drive is not None:
        rows = [r for r in rows if drive_value(r["drive"]) >= args.min_drive]
    if not rows:
        raise SystemExit("no usable rows")

    if args.pathway:
        pathways = []
        for spec in args.pathway:
            name, _, plist = spec.partition("=")
            pathways.append((name.strip(),
                             [p.strip() for p in plist.split(",") if p.strip()]))
    else:
        pathways = DEFAULT_PATHWAYS

    drives = sorted({r["drive"] for r in rows}, key=drive_value)
    xs = np.array([drive_value(d) for d in drives])

    def key_of(r):
        return f"{pops[r['source']]}->{pops[r['target']]}"

    index = {}
    for r in rows:
        index.setdefault(r["drive"], {})[key_of(r)] = r

    missing = set()
    for _, plist in pathways:
        for p in plist:
            if p not in index[drives[0]]:
                missing.add(p)
    if missing:
        raise SystemExit(f"pairs not found in the results: {sorted(missing)}")

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [1.5, 1]})

    curves = {}
    for name, plist in pathways:
        m, s = [], []
        for d in drives:
            v = np.array([index[d][p]["te_eff"] for p in plist])
            m.append(v.mean())
            s.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)
        m, s = np.array(m), np.array(s)
        curves[name] = m
        col = PATHWAY_COLOURS.get(name)
        ax.errorbar(xs, m, yerr=s, marker="o", markersize=4, lw=1.6,
                    capsize=2.5, color=col, label=f"{name} (n={len(plist)})")

        print(f"\n{name}: {', '.join(plist)}")
        for d, mi in zip(drives, m):
            lat = ", ".join(
                f"{p.split('->')[0]}\u2192{p.split('->')[1]} "
                f"{index[d][p]['u'] * bin_ms:.0f}ms" for p in plist)
            print(f"  {drive_value(d):6.2f}  mean TE {mi:+.5f}   peak delays: {lat}")

    ax.set_ylabel("mean effective TE (nats)")
    ax.legend(fontsize=8, frameon=False)
    ax.axhline(0, color="0.85", lw=0.8)

    if "feedforward" in curves and "feedback" in curves:
        denom = np.where(np.abs(curves["feedback"]) > 1e-9,
                         curves["feedback"], np.nan)
        ratio = curves["feedforward"] / denom
        axr.plot(xs, ratio, marker="s", markersize=4, lw=1.6, color="k")
        axr.axhline(1, color="0.7", ls=":", lw=1.0)
        axr.set_ylabel("feedforward / feedback")
        print("\nfeedforward / feedback ratio:")
        for d, v in zip(drives, ratio):
            print(f"  {drive_value(d):6.2f}  {v:+.2f}")
    else:
        axr.axis("off")
    axr.set_xlabel("background rate (spikes/s)")

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)