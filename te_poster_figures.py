#!/usr/bin/env python3
"""
te_poster_figures.py

Motivation
----------
The default plot stage of te_drive.py emits four exploratory figures. Two of
them do not survive scrutiny well enough to display: the connectivity matrices
are unreadable at poster size and say nothing the delay profile does not, and
the structure correlation is a positive control that belongs in a sentence
(Spearman rho = 0.29 +/- 0.13 across drives over the connected pairs, no
systematic dependence on drive).

This produces the two that do. It also reworks the summary curve. The default
version sums the effective TE over pairs that pass a significance threshold,
which makes it depend on three arbitrary choices at once: the alpha, the number
of surrogates that sets the p floor, and the delay window. Plotting the mean
effective TE over all ordered pairs removes all three dependencies and shows the
same shape, so the same claim is made with less to defend.

What it measures
----------------
Figure 1, two panels:

  A   (top row) effective TE against source-target delay at several drives,
      cropped to short latency. The full scan ran past one complete cycle of
      the population rhythm with no structure beyond the crop, which is a
      control worth stating in the caption since it cannot be seen on a cropped
      axis. Share the y axis (--share-y) unless there is a reason not to: with
      independent scales a panel whose peak is twice as large looks the same
      size as the others, which understates the very effect the panel exists to
      show. Fix the pair set across drives (--pairs) so the reader follows the
      same lines from panel to panel; selecting the strongest pairs
      independently at each drive gives three different legends and nothing to
      compare.
  B   (bottom row) mean effective TE across all ordered pairs against
      background rate, with the across-pair standard error. No threshold
      anywhere.

Figure 2: active information storage per population against background rate.
This is the one measure here that needs no surrogate, no threshold and no delay
window, so it is shown on its own rather than beside quantities that do.

A caveat the caption should carry: A and B are measured at delays below 20 ms,
within bursts, while the population rhythm has a period of roughly 150 ms. These
panels are not about that rhythm.

Usage
-----
    python te_poster_figures.py --out-dir ./te_out --tag full \\
        --min-drive 7.8 --crop-ms 20 \\
        --profile-drives 08.05 12.12 19.24 \\
        --out-prefix poster
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

LAYER_COLOURS = {"L23": "#0072B2", "L4": "#D55E00", "L5": "#009E73",
                 "L6": "#CC79A7"}


def pop_style(pop):
    colour = LAYER_COLOURS[pop[:-1]]
    if pop.endswith("E"):
        return dict(color=colour, linestyle="-", marker="o")
    return dict(color=colour, linestyle="--", marker="s")


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
    ap.add_argument("--crop-ms", type=float, default=20.0)
    ap.add_argument("--profile-drives", nargs="+", default=None,
                    help="drives shown in panel A; defaults to low, peak, high")
    ap.add_argument("--n-profile-pairs", type=int, default=5)
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="fixed pairs shown at every drive, as SRC->TGT "
                         "(e.g. L4I->L4E). Defaults to the strongest pairs at "
                         "the peak drive, which keeps one legend for all "
                         "panels.")
    ap.add_argument("--share-y", action="store_true",
                    help="common y axis across the panel A sub-panels")
    ap.add_argument("--out-prefix", default="poster")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(args.out_dir, "meta.json")) as fh:
        meta = json.load(fh)
    pops = meta["pops"]
    bin_ms = meta.get("bin_ms", 1.0)

    rows = list(np.load(os.path.join(args.out_dir, f"te_{args.tag}.npy"),
                        allow_pickle=True))
    rows = [r for r in rows if np.isfinite(r.get("te_eff", np.nan))]
    if args.min_drive is not None:
        rows = [r for r in rows if drive_value(r["drive"]) >= args.min_drive]
    if not rows:
        raise SystemExit("no usable rows after filtering")

    embed = np.load(os.path.join(args.out_dir, "embed.npy"), allow_pickle=True)
    ais = {(r["drive"], r["pop"]): r["ais"] for r in embed}

    drives = sorted({r["drive"] for r in rows}, key=drive_value)
    xs = np.array([drive_value(d) for d in drives])

    # ---------------- figure 1 ----------------
    if args.profile_drives:
        show = [d for d in args.profile_drives if d in drives]
        if len(show) != len(args.profile_drives):
            missing = set(args.profile_drives) - set(show)
            raise SystemExit(f"drives not present in tag '{args.tag}': "
                             f"{sorted(missing)}")
    else:
        show = [drives[0], drives[int(np.argmax(
            [np.mean([r["te_eff"] for r in rows if r["drive"] == d])
             for d in drives]))], drives[-1]]

    n_show = len(show)

    # Fix one pair set across all drives so the panels are comparable and one
    # legend serves them all.
    def key_of(r):
        return f"{pops[r['source']]}->{pops[r['target']]}"

    if args.pairs:
        pair_keys = list(args.pairs)
        available = {key_of(r) for r in rows}
        unknown = [p for p in pair_keys if p not in available]
        if unknown:
            raise SystemExit(f"unknown pairs: {unknown}")
    else:
        anchor = show[int(np.argmax([
            np.mean([r["te_eff"] for r in rows if r["drive"] == d])
            for d in show]))]
        pair_keys = [key_of(r) for r in
                     sorted([r for r in rows if r["drive"] == anchor],
                            key=lambda r: -r["te_eff"])[:args.n_profile_pairs]]
        print(f"pair set taken from drive {anchor}: {', '.join(pair_keys)}")

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pair_colour = {p: colours[i % len(colours)] for i, p in enumerate(pair_keys)}

    fig = plt.figure(figsize=(4.0 * n_show, 7.0))
    gs = fig.add_gridspec(2, n_show, height_ratios=[1.0, 0.95],
                          hspace=0.42, wspace=0.28)

    prof_axes = []
    for k, d in enumerate(show):
        ax = fig.add_subplot(gs[0, k], sharey=prof_axes[0]
                             if (args.share_y and prof_axes) else None)
        prof_axes.append(ax)
        by_key = {key_of(r): r for r in rows if r["drive"] == d}
        for p in pair_keys:
            r = by_key.get(p)
            if r is None:
                continue
            u = np.asarray(r["u_list"]) * bin_ms
            y = np.asarray(r["profile"]) - np.asarray(r["profile_surr"])
            m = u <= args.crop_ms
            ax.plot(u[m], y[m], lw=1.6, color=pair_colour[p],
                    label=p.replace("->", "\u2192"))
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_xlim(0, args.crop_ms)
        ax.set_xlabel("source-target delay (ms)")
        if k == 0:
            ax.set_ylabel("effective TE (nats)")
        elif args.share_y:
            ax.tick_params(labelleft=False)
        ax.set_title(f"{drive_value(d):.1f} spikes/s", fontsize=10)
    prof_axes[0].legend(fontsize=7.5, frameon=False, loc="best")
    prof_axes[0].text(-0.20, 1.08, "A", transform=prof_axes[0].transAxes,
                      fontsize=15, fontweight="bold")

    # panel B: mean over ALL ordered pairs, no threshold, full bottom row
    axb = fig.add_subplot(gs[1, :])
    means, sems, npair = [], [], []
    for d in drives:
        v = np.array([r["te_eff"] for r in rows if r["drive"] == d])
        means.append(v.mean())
        sems.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)
        npair.append(v.size)
    means, sems = np.array(means), np.array(sems)
    axb.errorbar(xs, means, yerr=sems, marker="o", color="k",
                 lw=1.6, markersize=5, capsize=3)
    axb.set_xlabel("background rate (spikes/s)")
    axb.set_ylabel("mean effective TE (nats)")
    axb.text(-0.065, 1.06, "B", transform=axb.transAxes,
             fontsize=15, fontweight="bold")

    i_pk = int(np.argmax(means))
    rest = np.delete(means, i_pk)
    print(f"peak: {xs[i_pk]:.2f} spikes/s, mean TE {means[i_pk]:.4f} nats")
    print(f"other drives: {rest.mean():.4f} +/- {rest.std(ddof=1):.4f} nats")
    print(f"peak stands {(means[i_pk] - rest.mean()) / rest.std(ddof=1):.1f} "
          f"SD above the rest")
    print(f"pairs per drive: {min(npair)} to {max(npair)}")

    f1 = f"{args.out_prefix}_te.png"
    fig.savefig(f1, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {f1}")

    # ---------------- figure 2 ----------------
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.4))
    for pop in pops:
        y = [ais.get((d, pop), np.nan) for d in drives]
        ax2.plot(xs, y, label=pop, markersize=4, lw=1.5, **pop_style(pop))
    ax2.set_xlabel("background rate (spikes/s)")
    ax2.set_ylabel("active information storage (nats)")
    ax2.legend(ncol=2, fontsize=8, frameon=False)
    f2 = f"{args.out_prefix}_ais.png"
    fig2.tight_layout()
    fig2.savefig(f2, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {f2}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)