#!/usr/bin/env python3
"""
analyze_subharmonic.py

Measures the fundamental peak frequency (f0, expected ~80 Hz) and the
lower-band peak frequency (f_low, expected ~40 Hz) in the Welch PSD of
each population's activity, for every background/input drive and trial,
and reports the ratio f_low / f0 (subharmonic hypothesis predicts ~0.5).

Expected directory layout
--------------------------
<data_dir>/<RATE>/trial_<x>/measurements/pop_activities/pop_activity_<y>.dat

    RATE  : subdirectory name, one per drive condition (e.g. "6", "9", "12", "18")
    x     : trial index, 0..9
    y     : population index, 0..7
        0 = L2/3 E   1 = L2/3 I   2 = L4 E   3 = L4 I
        4 = L5 E     5 = L5 I     6 = L6 E   7 = L6 I

Each .dat file is a population rate time series on a uniform time grid.
It may be stored as:
  - a single column of rate values (dt/fs must be supplied via --fs or --dt), or
  - two columns "time  rate" (dt is inferred from the time column).

Usage
-----
    python analyze_subharmonic.py --data-dir data_background_rate_big --fs 1000

    # or, if you know dt instead of fs:
    python analyze_subharmonic.py --data-dir data_background_rate_big --dt 0.001

Outputs (written next to --out-prefix, default "subharmonic"):
    <out-prefix>_trials.csv    one row per (rate, trial, population)
    <out-prefix>_summary.csv   one row per (rate, population), aggregated over trials

Tune the search bands with --f0-band / --flow-band if your peaks sit
somewhere other than the defaults (60-100 Hz and 20-55 Hz).
"""

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks

POP_NAMES = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]


def natural_key(s):
    """Sort strings that contain numbers in human order (e.g. '9' before '12')."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def discover_rates(data_dir):
    rates = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    rates.sort(key=natural_key)
    return rates


def load_series(path, fs=None, dt=None):
    """Load a pop_activity_*.dat file, return (signal, fs)."""
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        if fs is None and dt is None:
            raise ValueError(
                f"{path}: single-column file, but neither --fs nor --dt was given"
            )
        signal = arr
        fs_local = fs if fs is not None else 1.0 / dt
    else:
        # assume first column is time, last column is the rate
        t = arr[:, 0]
        signal = arr[:, -1]
        inferred_dt = np.median(np.diff(t))
        fs_local = 1.0 / inferred_dt
        if fs is not None and not np.isclose(fs_local, fs, rtol=1e-2):
            print(
                f"  [warn] {path}: inferred fs={fs_local:.3f} Hz from time column "
                f"differs from --fs={fs} Hz; using inferred value",
                file=sys.stderr,
            )
    return signal, fs_local


def dominant_peak_in_band(freqs, psd, band, prominence_rel=0.05):
    """
    Return (peak_freq, peak_power) of the most prominent local maximum of
    psd within [band[0], band[1]], or (nan, nan) if none is found.

    prominence_rel: required peak prominence, as a fraction of the psd's
    dynamic range within the band, to reject noise fluctuations.
    """
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return np.nan, np.nan
    f_band = freqs[mask]
    p_band = psd[mask]

    band_range = p_band.max() - p_band.min()
    if band_range <= 0:
        return np.nan, np.nan
    min_prominence = prominence_rel * band_range

    peaks, props = find_peaks(p_band, prominence=min_prominence)
    if len(peaks) == 0:
        # fall back to the raw maximum if no clean local peak was found
        idx = np.argmax(p_band)
        return f_band[idx], p_band[idx]

    # pick the most prominent peak
    best = peaks[np.argmax(props["prominences"])]
    return f_band[best], p_band[best]


def analyze_file(path, fs_arg, dt_arg, f0_band, flow_band, nperseg, prominence_rel):
    signal, fs = load_series(path, fs=fs_arg, dt=dt_arg)

    if nperseg is None:
        nseg = min(len(signal), 4096)
    else:
        nseg = min(nperseg, len(signal))

    freqs, psd = welch(signal, fs=fs, nperseg=nseg)

    f0, p0 = dominant_peak_in_band(freqs, psd, f0_band, prominence_rel)
    flow, plow = dominant_peak_in_band(freqs, psd, flow_band, prominence_rel)

    freq_resolution = fs / nseg
    ratio = flow / f0 if (not np.isnan(f0) and not np.isnan(flow) and f0 != 0) else np.nan

    return {
        "fs": fs,
        "n_samples": len(signal),
        "freq_resolution_hz": freq_resolution,
        "f0_hz": f0,
        "f0_power": p0,
        "flow_hz": flow,
        "flow_power": plow,
        "flow_over_f0": ratio,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data_background_rate_big",
                     help="Top-level data directory (default: %(default)s)")
    ap.add_argument("--fs", type=float, default=None,
                     help="Sampling frequency in Hz (use if files are single-column)")
    ap.add_argument("--dt", type=float, default=None,
                     help="Sampling interval in seconds (alternative to --fs)")
    ap.add_argument("--trials", type=int, default=10,
                     help="Number of trials, trial_0 .. trial_(N-1) (default: %(default)s)")
    ap.add_argument("--pops", type=int, default=8,
                     help="Number of populations, pop_activity_0 .. pop_activity_(N-1) (default: %(default)s)")
    ap.add_argument("--f0-band", type=float, nargs=2, default=[60.0, 100.0],
                     metavar=("LOW", "HIGH"), help="Search band for the fundamental peak (default: 60 100)")
    ap.add_argument("--flow-band", type=float, nargs=2, default=[20.0, 55.0],
                     metavar=("LOW", "HIGH"), help="Search band for the lower/subharmonic peak (default: 20 55)")
    ap.add_argument("--nperseg", type=int, default=None,
                     help="Welch segment length in samples (default: min(len(signal), 4096))")
    ap.add_argument("--prominence-rel", type=float, default=0.05,
                     help="Relative peak prominence threshold within each band (default: %(default)s)")
    ap.add_argument("--ratio-tol", type=float, default=None,
                     help="Tolerance around 0.5 for flagging a clean subharmonic ratio "
                          "(default: computed per-row from the spectral resolution)")
    ap.add_argument("--out-prefix", default="subharmonic",
                     help="Prefix for output CSV files (default: %(default)s)")
    args = ap.parse_args()

    if args.fs is None and args.dt is None:
        print(
            "[info] neither --fs nor --dt was given; will try to infer dt from a "
            "time column in each file. Pass --fs/--dt explicitly if your files are "
            "single-column rate values.",
            file=sys.stderr,
        )

    rates = discover_rates(args.data_dir)
    if not rates:
        print(f"No rate subdirectories found under {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(rates)} drive conditions: {rates}")

    rows = []
    for rate in rates:
        for x in range(args.trials):
            trial_dir = os.path.join(args.data_dir, rate, f"trial_{x}", "measurements", "pop_activities")
            if not os.path.isdir(trial_dir):
                continue
            for y in range(args.pops):
                fpath = os.path.join(trial_dir, f"pop_activity_{y}.dat")
                if not os.path.isfile(fpath):
                    continue
                try:
                    res = analyze_file(
                        fpath, args.fs, args.dt,
                        tuple(args.f0_band), tuple(args.flow_band),
                        args.nperseg, args.prominence_rel,
                    )
                except Exception as e:
                    print(f"  [error] {fpath}: {e}", file=sys.stderr)
                    continue
                res.update({
                    "rate": rate,
                    "trial": x,
                    "pop_idx": y,
                    "pop_name": POP_NAMES[y] if y < len(POP_NAMES) else f"pop{y}",
                })
                rows.append(res)

    if not rows:
        print("No data files were found/parsed. Check --data-dir, --trials, --pops.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    cols_first = ["rate", "trial", "pop_idx", "pop_name"]
    df = df[cols_first + [c for c in df.columns if c not in cols_first]]
    trials_path = f"{args.out_prefix}_trials.csv"
    df.to_csv(trials_path, index=False)
    print(f"Wrote per-trial results to {trials_path}")

    # aggregate across trials for each (rate, population)
    def agg(g):
        n = g["flow_over_f0"].notna().sum()
        out = {
            "n_trials_with_both_peaks": n,
            "f0_hz_mean": g["f0_hz"].mean(),
            "f0_hz_std": g["f0_hz"].std(),
            "flow_hz_mean": g["flow_hz"].mean(),
            "flow_hz_std": g["flow_hz"].std(),
            "flow_over_f0_mean": g["flow_over_f0"].mean(),
            "flow_over_f0_std": g["flow_over_f0"].std(),
            "flow_over_f0_sem": g["flow_over_f0"].std() / np.sqrt(n) if n > 0 else np.nan,
            "freq_resolution_hz_mean": g["freq_resolution_hz"].mean(),
        }
        return pd.Series(out)

    summary = df.groupby(["rate", "pop_idx", "pop_name"], sort=False).apply(agg).reset_index()

    # flag rows consistent with a clean 0.5 subharmonic ratio, within spectral resolution
    def is_subharmonic(row):
        if np.isnan(row["flow_over_f0_mean"]) or np.isnan(row["f0_hz_mean"]) or row["f0_hz_mean"] == 0:
            return False
        tol = args.ratio_tol
        if tol is None:
            # convert an absolute frequency resolution into a tolerance on the ratio
            tol = row["freq_resolution_hz_mean"] / row["f0_hz_mean"]
        return abs(row["flow_over_f0_mean"] - 0.5) <= tol

    summary["consistent_with_0.5_subharmonic"] = summary.apply(is_subharmonic, axis=1)

    # keep rates in natural (numeric-aware) order in the summary table
    summary["rate"] = pd.Categorical(summary["rate"], categories=sorted(summary["rate"].unique(), key=natural_key), ordered=True)
    summary = summary.sort_values(["rate", "pop_idx"]).reset_index(drop=True)

    summary_path = f"{args.out_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote per-(rate, population) summary to {summary_path}")

    print("\nSummary preview:")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(summary[["rate", "pop_name", "f0_hz_mean", "flow_hz_mean",
                        "flow_over_f0_mean", "flow_over_f0_sem",
                        "consistent_with_0.5_subharmonic"]])


if __name__ == "__main__":
    main()