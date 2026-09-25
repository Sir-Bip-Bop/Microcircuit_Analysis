#!/usr/bin/env python3
"""
entropy_evidence.py

Motivation
----------
The spectral entropy curve is a summary statistic, and a reader seeing only the
curve has to take on trust that a low value corresponds to something meaningful
about the dynamics. This builds the supporting panel that shows what it is
measuring.

Population activity in this regime is burst-like, with true silences between
events rather than a tonic oscillation, so neither a raster nor a raw rate trace
illustrates the point well: bursts at high drive look larger and cleaner than
bursts at the entropy minimum, which invites exactly the wrong conclusion. What
differs is the regularity of the inter-burst interval, and the autocorrelation
of the population rate shows that directly. Persistent side lobes at multiples
of the interval mean the burst timing repeats reliably; decay after the first
lobe means it does not.

Every trial contributes rather than one, because the across-trial variability in
this sweep is large enough that a single seed can misrepresent a drive. The band
is the across-trial spread, so a reader can see whether the difference between
drives exceeds the difference between seeds.

What it measures
----------------
Nothing new. For each population and drive, the normalised autocorrelation of
the population rate, computed per trial over the full trace and shown as the
across-trial mean with a band. Spectral entropy is printed as mean and standard
deviation across the same trials, so the number quoted alongside the figure
carries its own variability rather than coming from one trial. The
autocorrelation at one, two and three periods is printed too, which is the
quantitative form of the claim the figure makes visually.

Usage
-----
    python entropy_evidence.py \\
        --data-root data_background_rate_big \\
        --drives 09.58 16.19 --pops 2 0 \\
        --acf-max-ms 600 --period-ms 150 \\
        --out entropy_evidence.png

    # pops are pop_activity indices: 0 = L23E, 2 = L4E, 4 = L5E, 6 = L6E
    # --band sd (default), iqr, or range. iqr is safer if one seed is an outlier
    # --show-trials overlays every individual trial as a thin line
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

POPS = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
LAYER_COLOURS = {"L23": "#0072B2", "L4": "#D55E00", "L5": "#009E73",
                 "L6": "#CC79A7"}


def autocorr(x, max_lag):
    """
    Normalised autocorrelation of the population rate, computed by FFT.

    Periodic bursting gives side lobes at multiples of the inter-burst interval
    whose decay reflects how reliably the timing repeats. Irregular bursting
    decays to zero after the first trough.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    nf = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, nf)
    ac = np.fft.irfft(F * np.conj(F), nf)[:max_lag + 1]
    return ac / ac[0]


def spectral_entropy(f, P, f_max=150.0):
    """Shannon entropy in bits of the normalised power spectrum, 0 to f_max."""
    b = (f > 0) & (f <= f_max)
    p = P[b]
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--drives", nargs="+", required=True)
    ap.add_argument("--pops", type=int, nargs="+", default=[2, 0])
    ap.add_argument("--max-trials", type=int, default=None)
    ap.add_argument("--dt-ms", type=float, default=0.2)
    ap.add_argument("--acf-max-ms", type=float, default=600.0)
    ap.add_argument("--period-ms", type=float, default=None)
    ap.add_argument("--band", choices=["sd", "iqr", "range"], default="sd")
    ap.add_argument("--show-trials", action="store_true")
    ap.add_argument("--f-max", type=float, default=150.0)
    ap.add_argument("--nperseg", type=int, default=16384)
    ap.add_argument("--ncols", type=int, default=None)
    ap.add_argument("--out", default="entropy_evidence.png")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    fs = 1000.0 / args.dt_ms
    max_lag = int(round(args.acf_max_ms / args.dt_ms))
    lags = np.arange(max_lag + 1) * args.dt_ms

    def trial_files(drive, pop):
        ts = sorted(glob.glob(os.path.join(args.data_root, drive, "trial*")))
        if args.max_trials:
            ts = ts[:args.max_trials]
        return [os.path.join(t, "measurements", "pop_activities",
                             f"pop_activity_{pop}.dat") for t in ts]

    n_pan = len(args.pops)
    ncols = args.ncols or n_pan
    nrows = int(np.ceil(n_pan / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.8 * 2* ncols, 3.3 * 2 *  nrows), squeeze=False)
    flat = axes.ravel()

    for k, pop in enumerate(args.pops):
        ax = flat[k]
        name = POPS[pop]
        base = LAYER_COLOURS[name[:-1]]

        for c, drive in enumerate(args.drives):
            files = trial_files(drive, pop)
            if not files:
                raise SystemExit(f"no trials for drive {drive}")
            acs, ents = [], []
            for f_ in files:
                a = np.loadtxt(f_)
                acs.append(autocorr(a, max_lag))
                fr, P = welch(a - a.mean(), fs=fs,
                              nperseg=min(args.nperseg, len(a)))
                ents.append(spectral_entropy(fr, P, args.f_max))
            acs = np.array(acs)
            m = acs.mean(axis=0)
            n_tr = len(acs)
            sd = np.std(ents, ddof=1) if n_tr > 1 else 0.0

            if args.band == "sd":
                s = acs.std(axis=0, ddof=1) if n_tr > 1 else np.zeros_like(m)
                lo, hi = m - s, m + s
            elif args.band == "iqr":
                lo, hi = np.percentile(acs, [25, 75], axis=0)
            else:
                lo, hi = acs.min(axis=0), acs.max(axis=0)

            colour = base if c == 0 else "0.45"
            if args.show_trials:
                for row in acs:
                    ax.plot(lags, row, color=colour, lw=0.5, alpha=0.25)
            ax.fill_between(lags, lo, hi, color=colour, alpha=0.20, lw=0)
            ax.plot(lags, m, color=colour, lw=1.6,
                    label=f"drive {drive}   H = {np.mean(ents):.2f} "
                          f"± {sd:.2f} bits")

            print(f"{name} drive {drive}: n = {n_tr} trials, "
                  f"H = {np.mean(ents):.3f} ± {sd:.3f} bits")
            if args.period_ms:
                for j in (1, 2, 3):
                    lag = j * args.period_ms
                    if lag > args.acf_max_ms:
                        break
                    i = int(round(lag / args.dt_ms))
                    across = acs[:, i]
                    print(f"    ACF at {j} period(s) ({lag:.0f} ms): "
                          f"{across.mean():+.3f} ± "
                          f"{across.std(ddof=1) if n_tr > 1 else 0:.3f}")

        if args.period_ms:
            j = 1
            while j * args.period_ms <= args.acf_max_ms:
                ax.axvline(j * args.period_ms, color="0.8", ls=":", lw=0.9)
                j += 1
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_xlim(0, args.acf_max_ms)
        ax.set_xlabel("lag (ms)", fontsize=14)
        ax.set_ylabel("autocorrelation", fontsize=14)
        ax.set_title(name, fontsize=22)
        ax.legend(fontsize=14, frameon=False)

    for k in range(n_pan, len(flat)):
        flat[k].axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)