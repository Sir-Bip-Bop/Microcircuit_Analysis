#!/usr/bin/env python3
"""
packet_mi.py

Motivation
----------
Turns the packet simulations into the transmission measurement: how much
information about the injected packet each population carries, and how that
decays as the signal moves away from the injection site.

Mutual information is the right quantity here because the injected amplitude is
a known discrete variable under experimental control, so there is no need for
the surrogate machinery that the transfer entropy analysis required. The
estimate is still biased, and the bias is the main thing this script has to
handle: plug-in mutual information between a K-level stimulus and a B-bin
response is biased upward by roughly (K-1)(B-1)/(2N ln2) bits, which with a few
hundred trials is not negligible. The correction here is a shuffle null, which
estimates that bias empirically under the same binning, sample size and
marginals as the real estimate, and is subtracted from it.

What it measures
----------------
For each drive, population and time bin relative to packet onset:

    I(amplitude ; spike count in bin)

reported as the raw plug-in estimate, the mean and standard deviation of the
shuffle null, and the difference. Zero after correction means no information,
whatever the raw value looks like.

From that, per population:

    peak MI     the maximum corrected information
    latency     the bin at which that peak occurs
    attenuation the peak relative to the injected population

The latency ordering is the part worth checking before interpreting anything.
If information appears in L2/3 and L5 later than in L4E, the signal is
propagating. If it appears everywhere simultaneously, the populations are
responding to a common input rather than relaying, and the word transmission
should not be used.

A caveat to carry into any caption: MI between a K-level stimulus and any
response is bounded above by log2(K) bits, so with four amplitudes nothing can
exceed 2 bits. Attenuation should be read as a ratio between populations, not
against that ceiling.

Usage
-----
    python packet_mi.py --data-root ./packet_out \\
        --bin-ms 5 --window-ms 150 --n-shuffle 200 \\
        --out packet_mi.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

LAYER_COLOURS = {"L23": "#0072B2", "L4": "#D55E00", "L5": "#009E73",
                 "L6": "#CC79A7"}


def pop_style(pop):
    colour = LAYER_COLOURS[pop[:-1]]
    if pop.endswith("E"):
        return dict(color=colour, linestyle="-")
    return dict(color=colour, linestyle="--")


def _read_slow(path):
    """Line-by-line fallback. Slow but tolerant of any header NEST writes."""
    out = []
    with open(path) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                out.append(float(parts[1]))
            except ValueError:
                continue          # the uncommented "sender time_ms" header
    return np.asarray(out, dtype=np.float64)


def read_nest_ascii(paths):
    """
    Read spike times from NEST ascii recorder files.

    NEST writes a column header line that is NOT comment-prefixed, so a parser
    told to expect floats fails on the first row. Values are therefore coerced
    rather than declared, and anything non-numeric is dropped. Parsing is done
    per file with pandas and concatenated as arrays: at these sizes the
    recorders hold millions of spikes per population, and a Python list of that
    many floats costs far more memory than the data.
    """
    chunks = []
    for path in paths:
        arr = None
        try:
            import pandas as pd
            df = pd.read_csv(path, sep=r"\s+", header=None, comment="#",
                             names=["sender", "t"], usecols=[0, 1],
                             on_bad_lines="skip")
            arr = pd.to_numeric(df["t"], errors="coerce").to_numpy(np.float64)
        except Exception:
            arr = _read_slow(path)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            chunks.append(arr)
    if not chunks:
        return np.zeros(0)
    return np.sort(np.concatenate(chunks))


def plugin_mi(levels, response, n_bins):
    """
    Plug-in mutual information in bits between a discrete stimulus level and a
    scalar response, with the response quantile-binned.

    Quantile binning rather than equal-width because the response distribution
    is heavily skewed in a bursting regime: equal-width bins would put almost
    every trial in one bin and report no information regardless of the truth.
    """
    if response.std() == 0:
        return 0.0
    edges = np.quantile(response, np.linspace(0, 1, n_bins + 1)[1:-1])
    rb = np.searchsorted(edges, response, side="right")
    n = response.size
    joint = np.zeros((int(levels.max()) + 1, n_bins))
    np.add.at(joint, (levels, rb), 1.0)
    joint /= n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log2(joint[nz] / (px @ py)[nz])))


def mi_with_null(levels, response, n_bins, n_shuffle, rng):
    raw = plugin_mi(levels, response, n_bins)
    null = np.empty(n_shuffle)
    for i in range(n_shuffle):
        null[i] = plugin_mi(rng.permutation(levels), response, n_bins)
    return raw, float(null.mean()), float(null.std(ddof=1))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--drives", nargs="+", default=None)
    ap.add_argument("--bin-ms", type=float, default=5.0)
    ap.add_argument("--window-ms", type=float, default=150.0,
                    help="analysis window after packet onset")
    ap.add_argument("--pre-ms", type=float, default=50.0,
                    help="baseline window before onset, shown as a check that "
                         "the corrected estimate sits at zero where it must")
    ap.add_argument("--evoked-lo", type=float, default=0.0)
    ap.add_argument("--evoked-hi", type=float, default=15.0,
                    help="window for the evoked-response check. Keep it short: "
                         "the injected population responds within a couple of "
                         "milliseconds before recurrent inhibition cuts it off, "
                         "so a wide window averages that response away and "
                         "makes the injection site look unresponsive.")
    ap.add_argument("--n-response-bins", type=int, default=8)
    ap.add_argument("--n-shuffle", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--psth", action="store_true",
                    help="plot the mean response per amplitude level instead of "
                         "estimating information. This is the tool for choosing "
                         "the amplitude: it works at low trial counts, where a "
                         "mutual information estimate is pure bias, and it "
                         "shows the effect directly rather than through a "
                         "statistic.")
    ap.add_argument("--out", default="packet_mi.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drives = args.drives or sorted(
        os.path.basename(d.rstrip("/"))
        for d in glob.glob(os.path.join(args.data_root, "*"))
        if os.path.isdir(d))
    if not drives:
        raise SystemExit(f"no drive directories under {args.data_root}")

    edges = np.arange(-args.pre_ms, args.window_ms + args.bin_ms, args.bin_ms)
    centres = edges[:-1] + args.bin_ms / 2

    fig, axes = plt.subplots(1, len(drives), figsize=(5.0 * len(drives), 4.0),
                             squeeze=False, sharey=True)

    for k, drive in enumerate(drives):
        d_dir = os.path.join(args.data_root, str(drive))
        sched = json.load(open(os.path.join(d_dir, "schedule.json")))
        onsets = np.asarray(sched["onsets_ms"])
        levels = np.asarray(sched["levels"], dtype=int)
        pops = sched["pops"]
        rng = np.random.default_rng(args.seed)

        ax = axes[0, k]
        summary, evoked = [], {}
        psth = {}
        for pop in pops:
            paths = sorted(glob.glob(os.path.join(d_dir, f"spikes_{pop}*.dat")))
            if not paths:
                print(f"[mi] {drive} {pop}: no spike files, skipping")
                continue
            print(f"[mi] {drive} {pop}: reading {len(paths)} files",
                  flush=True)
            times = read_nest_ascii(paths)
            print(f"[mi] {drive} {pop}: {times.size} spikes, "
                  f"t {times.min():.0f} to {times.max():.0f} ms", flush=True)

            # counts per trial per bin, from one sorted spike train
            counts = np.empty((onsets.size, centres.size))
            for i, t0 in enumerate(onsets):
                lo = np.searchsorted(times, t0 + edges[0])
                hi = np.searchsorted(times, t0 + edges[-1])
                rel = times[lo:hi] - t0
                counts[i] = np.histogram(rel, bins=edges)[0]

            # Evoked-response check, before any information estimate. If the
            # mean count in the response window does not separate by level,
            # there is nothing for the information measure to find and the
            # amplitudes need raising rather than the trial count.
            w = (centres >= args.evoked_lo) & (centres <= args.evoked_hi)
            by_lev = [counts[levels == L][:, w].sum(axis=1).mean()
                      for L in range(int(levels.max()) + 1)]
            evoked[pop] = by_lev

            if args.psth:
                psth[pop] = [(counts[levels == L].mean(axis=0),
                              counts[levels == L].std(axis=0, ddof=1)
                              / np.sqrt(max((levels == L).sum(), 1)))
                             for L in range(int(levels.max()) + 1)]
                continue

            mi = np.empty(centres.size)
            for b in range(centres.size):
                raw, nm, _ = mi_with_null(levels, counts[:, b],
                                          args.n_response_bins,
                                          args.n_shuffle, rng)
                mi[b] = raw - nm

            ax.plot(centres, mi, lw=1.5, label=pop, **pop_style(pop))
            i_pk = int(np.argmax(mi))
            summary.append((pop, mi[i_pk], centres[i_pk],
                            float(mi[centres < 0].mean())))

        if args.psth:
            continue
        ax.axvline(0, color="0.7", ls=":", lw=1.0)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_xlabel("time from packet onset (ms)")
        if k == 0:
            ax.set_ylabel("information about packet amplitude (bits)")
        ax.set_title(f"{drive} spikes/s", fontsize=11)
        ax.legend(fontsize=7, ncol=2, frameon=False)

        print(f"\ndrive {drive}: mean spike count "
              f"{args.evoked_lo:.0f}-{args.evoked_hi:.0f} ms after onset, "
              f"by level")
        print(f"{'pop':>6} " + " ".join(f"{f'lev{i}':>10}"
                                        for i in range(len(sched["amplitudes"])))
              + f"{'max/min':>9}")
        for pop, v in evoked.items():
            ratio = max(v) / v[0] if v[0] > 0 else float("nan")
            print(f"{pop:>6} " + " ".join(f"{x:>10.1f}" for x in v)
                  + f"{ratio:>9.2f}")
        print("Read this both ways. If the injected population barely "
              "separates, the pulse is too weak. If distant populations rise "
              "several-fold, the pulse is triggering a network-wide event and "
              "there is no attenuation left to measure: reduce the "
              "amplitudes until the response stays graded.")

        print(f"\ndrive {drive}   ceiling log2(K) = "
              f"{np.log2(len(sched['amplitudes'])):.2f} bits")
        print(f"{'pop':>6} {'peak MI':>9} {'latency':>9} {'baseline':>9} "
              f"{'vs L4E':>8}")
        ref = dict((p, v) for p, v, _, _ in summary).get("L4E", np.nan)
        for pop, v, lat, base in summary:
            print(f"{pop:>6} {v:>9.4f} {lat:>7.1f}ms {base:>9.4f} "
                  f"{v / ref if ref else np.nan:>8.2f}")
        print("baseline is the mean corrected MI before onset and should be "
              "near zero; if it is not, the correction is not working")

    if args.psth and psth:
        plt.close(fig)
        names = list(psth)
        fig, axp = plt.subplots(2, 4, figsize=(17, 7), sharex=True)
        amps = sched["amplitudes"]
        print("\nseparation of the largest level from level 0, in units of the "
              "combined standard error, peak over the response window:")
        for a, pop in zip(axp.ravel(), names):
            series = psth[pop]
            for L, (m, se) in enumerate(series):
                c = plt.cm.viridis(L / max(len(series) - 1, 1))
                a.plot(centres, m, color=c, lw=1.4, label=f"{amps[L]:g}")
                a.fill_between(centres, m - se, m + se, color=c, alpha=0.20,
                               lw=0)
            m0, se0 = series[0]
            mN, seN = series[-1]
            w = centres >= 0
            z = (mN - m0) / np.sqrt(se0 ** 2 + seN ** 2 + 1e-12)
            i = int(np.argmax(np.abs(z[w])))
            print(f"  {pop:>6}  z = {z[w][i]:+6.2f} at "
                  f"{centres[w][i]:6.1f} ms")
            a.axvline(0, color="0.7", ls=":", lw=1.0)
            a.set_title(pop, fontsize=10)
            a.set_xlabel("time from onset (ms)")
            a.set_ylabel("spikes per bin")
        axp.ravel()[0].legend(fontsize=7, frameon=False, title="amplitude")
        print("A pulse worth running has a clear graded separation in L4E and "
              "progressively weaker separation with distance from it. |z| "
              "below about 2 anywhere means the trial count is too low to "
              "judge, not that the pulse is too weak.")

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)