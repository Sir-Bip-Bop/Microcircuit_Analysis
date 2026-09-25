#!/usr/bin/env python3
"""
state_check.py

Motivation
----------
Per-trial firing rates in the background-rate sweep reveal that below roughly
8.5 spikes/s the network is bistable: some seeds sustain activity while others
collapse to silence, with L23E showing exactly zero spikes over 60 s in several
trials. Averaging any quantity across trials in that region mixes two
dynamical regimes and reports a number describing neither. Error bars computed
as SEM across trials compound the problem, presenting a bimodal mixture as
scatter about a single value.

This quantifies the effect: where the bistability starts, what fraction of
trials survive at each drive, and whether the collapse is network-wide or
confined to particular populations.

What it measures
----------------
Per drive and trial, the mean rate of every population. Then:

  fraction active   proportion of trials above the activity threshold
  rate spread       across-trial min and max for the reference population
  per-population    whether silence is network-wide or layer-specific

Writes a boolean mask (n_drives, n_trials) that downstream analysis can use to
restrict statistics to active trials, together with the per-trial rates so the
threshold can be revisited without reloading the traces.

The threshold is a choice, not a fact. It defaults to a rate low enough that
only genuinely collapsed trials fall below it, and the printed distribution
lets you see whether the classification is clean or whether trials sit near the
boundary. If the histogram is not clearly bimodal, the split is arbitrary and
should not be used.

Usage
-----
    python state_check.py --data-root data_background_rate_big \\
        --out ./state.npz

    # then in the notebook, restrict the entropy statistics to active trials:
    #   m = np.load('state.npz')['mask']          # (n_drives, n_trials)
    #   vals = [v for k, v in enumerate(entropy_all[p][i]) if m[i, k]]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

POPS = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]


def drive_value(path):
    return float(re.search(r"[\d.]+", os.path.basename(path.rstrip("/"))).group())


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--pop-indices", type=int, nargs="+",
                   default=list(range(8)))
    p.add_argument("--ref-pop", type=int, default=0,
                   help="population used to classify a trial as active")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="mean rate in spikes/s below which a trial counts as "
                        "collapsed")
    p.add_argument("--out", default="./state.npz")
    args = p.parse_args(argv)

    drive_dirs = sorted(
        (d for d in glob.glob(os.path.join(args.data_root, "*"))
         if os.path.isdir(d)), key=drive_value)
    if not drive_dirs:
        raise SystemExit(f"no drive directories under {args.data_root}")

    drives, rates = [], []
    for d in drive_dirs:
        trials = sorted(glob.glob(os.path.join(d, "trial*")))
        row = []
        for t in trials:
            base = os.path.join(t, "measurements", "pop_activities")
            row.append([np.loadtxt(os.path.join(base, f"pop_activity_{j}.dat")).mean()
                        for j in args.pop_indices])
        rates.append(row)
        drives.append(drive_value(d))
        print(f"[state] {drive_value(d):6.2f}: {len(trials)} trials", flush=True)

    n_tr = min(len(r) for r in rates)
    rates = np.array([r[:n_tr] for r in rates])      # (n_drives, n_trials, n_pops)
    drives = np.array(drives)
    ref = rates[:, :, args.pop_indices.index(args.ref_pop)]
    mask = ref >= args.threshold

    print()
    print(f"Classification on {POPS[args.ref_pop]} at {args.threshold} spikes/s")
    print(f"{'drive':>7} {'active':>8} {'ref min':>8} {'ref max':>8} "
          f"{'silent pops in collapsed trials':>34}")
    for i, dv in enumerate(drives):
        frac = mask[i].mean()
        dead = ""
        if not mask[i].all():
            sub = rates[i][~mask[i]]
            quiet = [POPS[j] for k, j in enumerate(args.pop_indices)
                     if sub[:, k].max() < args.threshold]
            dead = ",".join(quiet) if quiet else "none (only ref)"
        print(f"{dv:>7.2f} {frac:>8.0%} {ref[i].min():>8.3f} "
              f"{ref[i].max():>8.3f} {dead:>34}")

    # is the split clean, or are trials sitting on the boundary?
    flat = ref.ravel()
    lo = flat[flat < args.threshold]
    hi = flat[flat >= args.threshold]
    print()
    print(f"Below threshold: n={lo.size}, max={lo.max() if lo.size else float('nan'):.3f}")
    print(f"Above threshold: n={hi.size}, min={hi.min() if hi.size else float('nan'):.3f}")
    if lo.size and hi.size:
        gap = hi.min() - lo.max()
        print(f"Gap across the threshold: {gap:.3f} spikes/s")
        if gap <= 0:
            print("WARNING: the two groups overlap at the threshold. The split "
                  "is arbitrary at the boundary, so treat trials near it as "
                  "unclassified rather than assigning them.")

    first_all = next((drives[i] for i in range(len(drives))
                      if mask[i:].all()), None)
    if first_all is not None:
        print(f"\nAll trials active from {first_all:.2f} spikes/s upward. "
              f"Below that, statistics pooled across trials mix regimes.")

    np.savez(args.out, drives=drives, rates=rates, mask=mask,
             pops=np.array([POPS[j] for j in args.pop_indices]),
             threshold=args.threshold, ref_pop=args.ref_pop)
    print(f"[state] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)