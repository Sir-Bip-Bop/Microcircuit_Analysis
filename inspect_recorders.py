#!/usr/bin/env python3
"""
inspect_recorders.py

Motivation
----------
The synchrony null needs to rebuild population activity from the raw spike
recorders, which requires knowing which recorder file belongs to which
population and how many neurons each one covers. Both were assumptions, and the
validation check in synchrony_null.py rejected them: the rebuilt trace was
uncorrelated with the stored pop_activity and the mean rate was off by a factor
of roughly thirty.

Rather than guess at the cause, this determines the mapping from the data. It
correlates every candidate rebuilt trace against every stored pop_activity trace
and reports the assignment that actually matches, along with the neuron count
each recorder implies. Uncorrelated rows mean a time-window problem rather than
an ordering problem, and the two are distinguished here.

What it measures
----------------
For each recorder id: spike count, time range, number of distinct senders, and
the sender id range. For each stored pop_activity: mean rate and duration.

Then a correlation matrix between the two sets, computed on a short common
window, plus the implied neuron count per recorder derived from the ratio of
raw spike counts to the stored per-neuron rate. If a clean permutation appears,
that is the mapping to use. If every entry is near zero, the recorders and the
pop_activity files do not describe the same time interval, and the printed time
ranges will show why.

Usage
-----
    python inspect_recorders.py \\
        --trial-dir data_background_rate_big/09.07/trial_0 \\
        --n-full n_full.npy --dt-ms 0.2
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np


def read_nest_ascii(path):
    senders, times = [], []
    with open(path, "r") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                s = int(float(parts[0]))
                t = float(parts[1])
            except ValueError:
                continue
            senders.append(s)
            times.append(t)
    return (np.asarray(senders, dtype=np.int64),
            np.asarray(times, dtype=np.float64))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trial-dir", required=True)
    p.add_argument("--n-full", default=None)
    p.add_argument("--dt-ms", type=float, default=0.2)
    p.add_argument("--window-ms", type=float, default=5000.0,
                   help="length of the common window used for correlating")
    args = p.parse_args(argv)

    # ---- recorders -------------------------------------------------------
    files = sorted(glob.glob(os.path.join(args.trial_dir,
                                          "spike_recorder-*.dat")))
    if not files:
        raise SystemExit(f"no spike_recorder-*.dat under {args.trial_dir}")
    groups = {}
    for f in files:
        m = re.search(r"spike_recorder-(\d+)-(\d+)\.dat$", os.path.basename(f))
        if m:
            groups.setdefault(int(m.group(1)), []).append(f)

    rec = {}
    print(f"{'rec id':>10} {'files':>6} {'spikes':>10} {'t min':>10} "
          f"{'t max':>10} {'senders':>8} {'sender range':>20}")
    for rid in sorted(groups):
        sn, tm = [], []
        for f in groups[rid]:
            s, t = read_nest_ascii(f)
            sn.append(s)
            tm.append(t)
        s = np.concatenate(sn) if sn else np.zeros(0, dtype=np.int64)
        t = np.concatenate(tm) if tm else np.zeros(0)
        rec[rid] = (s, t)
        if t.size:
            u = np.unique(s)
            print(f"{rid:>10} {len(groups[rid]):>6} {t.size:>10} "
                  f"{t.min():>10.1f} {t.max():>10.1f} {u.size:>8} "
                  f"{str((int(u.min()), int(u.max()))):>20}")
        else:
            print(f"{rid:>10} {len(groups[rid]):>6} {0:>10} "
                  f"{'-':>10} {'-':>10} {0:>8} {'-':>20}")

    # ---- stored pop_activity --------------------------------------------
    pa_dir = os.path.join(args.trial_dir, "measurements", "pop_activities")
    pa_files = sorted(glob.glob(os.path.join(pa_dir, "pop_activity_*.dat")),
                      key=lambda f: int(re.search(r"_(\d+)\.dat", f).group(1)))
    if not pa_files:
        raise SystemExit(f"no pop_activity files under {pa_dir}")

    print()
    print(f"{'pop file':>14} {'samples':>10} {'duration ms':>12} "
          f"{'mean rate Hz':>13}")
    pa = {}
    for f in pa_files:
        j = int(re.search(r"_(\d+)\.dat", f).group(1))
        a = np.loadtxt(f)
        pa[j] = a
        print(f"{os.path.basename(f):>14} {a.size:>10} "
              f"{a.size * args.dt_ms:>12.1f} {a.mean():>13.4f}")

    # ---- implied neuron counts ------------------------------------------
    dur_ms = min(a.size for a in pa.values()) * args.dt_ms
    print()
    print("Implied neuron count per recorder, if it were population j:")
    print("  (total spikes) / (duration_s * mean per-neuron rate of pop j)")
    n_full = np.load(args.n_full) if args.n_full else None
    hdr = "".join(f"{f'pop{j}':>9}" for j in sorted(pa))
    print(f"{'rec id':>10}{hdr}")
    for rid in sorted(rec):
        t = rec[rid][1]
        row = ""
        for j in sorted(pa):
            r = pa[j].mean()
            implied = t.size / (dur_ms * 1e-3 * r) if r > 0 else np.nan
            row += f"{implied:>9.0f}"
        print(f"{rid:>10}{row}")
    if n_full is not None:
        print(f"{'n_full':>10}" + "".join(f"{int(n):>9}" for n in n_full))

    # ---- correlation on a common window ---------------------------------
    t0 = max(min(t.min() for _, t in rec.values() if t.size), 0.0)
    t1 = t0 + args.window_ms
    n_bins = int(round((t1 - t0) / args.dt_ms))

    built = {}
    for rid, (s, t) in rec.items():
        sel = (t >= t0) & (t < t1)
        idx = ((t[sel] - t0) / args.dt_ms).astype(np.int64)
        built[rid] = np.bincount(idx, minlength=n_bins)[:n_bins].astype(float)

    print()
    print(f"Correlation on [{t0:.0f}, {t1:.0f}] ms, assuming pop_activity "
          f"sample 0 is at t = {t0:.0f} ms.")
    print("A clean permutation with entries near 1 gives the mapping.")
    print(f"{'rec id':>10}" + "".join(f"{f'pop{j}':>9}" for j in sorted(pa))
          + "   verdict")
    best = {}
    for rid in sorted(rec):
        row, vals = "", []
        for j in sorted(pa):
            a = pa[j][:n_bins]
            n = min(len(a), n_bins)
            if n < 10 or built[rid][:n].std() == 0 or a[:n].std() == 0:
                c = np.nan
            else:
                c = float(np.corrcoef(built[rid][:n], a[:n])[0, 1])
            vals.append(c)
            row += f"{c:>9.3f}"
        v = np.asarray(vals, dtype=float)
        j = int(np.nanargmax(v))
        rest = np.delete(v, j)
        # The absolute correlation depends on binning noise and firing rate, so
        # the criterion is contrast, not magnitude: the true match must stand
        # well clear of every other candidate.
        margin = v[j] / max(np.nanmax(np.abs(rest)), 1e-9)
        print(f"{rid:>10}{row}   best pop{j}, {margin:>5.0f}x next")
        if v[j] > 0.05 and margin > 5:
            best[rid] = j

    print()
    if len(best) == len(rec) and len(set(best.values())) == len(best):
        print("Mapping recovered, recorder id -> pop_activity index:")
        for rid in sorted(best):
            print(f"  {rid} -> pop_activity_{best[rid]}")
    else:
        print("No clean mapping. If the whole matrix is near zero, the "
              "recorders and pop_activity do not cover the same interval: "
              "compare the recorder t min/max above against the pop_activity "
              "durations, and set --t-start and --t-offset accordingly.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)