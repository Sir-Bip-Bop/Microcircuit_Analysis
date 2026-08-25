"""
reconstruct_pop_activity.py

Motivation
----------
Figure 2 of the microcircuit paper (2:1 phase locking and bicoherence
across the background-rate sweep) consumes per-population activity
traces. The pop_activity files shipped with the simulations carry an
ambiguous time base (0.2 ms window indexing next to a freq_sample
constant implying 17 us bins), so instead of trusting them this script
rebuilds the population activity directly from the raw NEST spike
recorder output, with an explicit and self-documented time base.
Binning spikes into 1 ms windows and smoothing with a 2 ms Gaussian
kernel yields a rate estimate whose sampling rate (1 kHz), Nyquist
limit (500 Hz) and smoothing transfer function are known exactly,
which is what the downstream band-pass, phase extraction and
bicoherence steps need.

What it produces
----------------
For every background rate, trial and population: the population firing
rate in spikes/s per neuron (or raw counts per bin with --units count)
on a regular grid of --bin-ms milliseconds from --t-start to --t-stop,
smoothed with a zero-phase Gaussian kernel of sigma --sigma-ms.
The time grid is identical for every trial: either given explicitly
with --t-stop (recommended) or inferred ONCE, globally, by tail-reading
the last spike time of every recorder file in scope before processing
(NEST ASCII output is time ordered, so this is O(1) per file). A grid
inferred per trial from that trial's last spike would silently truncate
trials in which the network falls silent before the recording ends,
which happens routinely at the low end of a background-rate sweep;
trailing silence is data and is written out as zero-rate bins.
Spikes are pooled from ALL spike_recorder-*.dat files in each trial
directory (NEST writes one file per virtual process) and split by the
node-id ranges in population_nodeids.dat, so nothing is assumed about
which recorder id belongs to which population. Senders that fall in no
population range are counted and reported, never silently dropped.
Output goes to <trial>/measurements/<dest-name>/pop_activity_<p>.dat,
one float per line, with all reconstruction parameters recorded in
'#' header lines (np.loadtxt-compatible), leaving the original
pop_activities directory untouched.

Usage
-----
  python reconstruct_pop_activity.py --root data_background_rate_big \
      --t-start 500 --t-stop 5500
  python reconstruct_pop_activity.py --root data_background_rate_big
  python reconstruct_pop_activity.py --root data_background_rate_big \
      --only-bg bg_rate_4.0 --dest-name pop_activities_1ms
  python reconstruct_pop_activity.py --root data_background_rate_big \
      --bin-ms 1 --sigma-ms 2 --units count
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ----------------------------------------------------------------------
# reading NEST ASCII spike files
# ----------------------------------------------------------------------

def find_data_start(path, max_scan=10):
    """Return the index of the first data row, or None if the file
    contains no data rows (header-only files are legal NEST output when
    a virtual process recorded no spikes)."""
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i >= max_scan:
                break
            parts = line.split()
            if len(parts) >= 2:
                try:
                    int(parts[0])
                    float(parts[1])
                    return i
                except ValueError:
                    continue
    return None


def tail_last_time(path, block=4096):
    """Return the time of the last data row by reading only the file
    tail. NEST ASCII files are written in time order, so the last
    parseable row carries the file maximum. None for header-only files."""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - block))
        chunk = fh.read().decode(errors="replace")
    for line in reversed(chunk.splitlines()):
        parts = line.split()
        if len(parts) >= 2:
            try:
                int(parts[0])
                return float(parts[1])
            except ValueError:
                continue
    return None


def read_spike_file(path):
    """Return (senders int64, times_ms float64) for one recorder file."""
    start = find_data_start(path)
    if start is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    arr = np.loadtxt(path, skiprows=start, usecols=(0, 1), ndmin=2)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    return arr[:, 0].astype(np.int64), arr[:, 1].astype(np.float64)


# ----------------------------------------------------------------------
# per-trial reconstruction
# ----------------------------------------------------------------------

def process_trial(trial_dir, args):
    """Reconstruct all population activities for one trial directory.
    Returns a summary dict; raises on unrecoverable problems."""
    nodeid_path = os.path.join(trial_dir, "population_nodeids.dat")
    ranges = np.loadtxt(nodeid_path, dtype=np.int64, ndmin=2)
    if ranges.shape[1] != 2:
        raise ValueError(f"unexpected population_nodeids shape {ranges.shape}")
    starts, ends = ranges[:, 0], ranges[:, 1]
    if not np.all(np.diff(starts) > 0):
        raise ValueError("population_nodeids start ids are not increasing")
    n_pops = len(ranges)
    n_neurons = ends - starts + 1

    spike_files = sorted(glob.glob(os.path.join(trial_dir, args.recorder_glob)))
    if not spike_files:
        raise FileNotFoundError(
            f"no files matching {args.recorder_glob!r} in {trial_dir}")

    senders_list, times_list = [], []
    for f in spike_files:
        s, t = read_spike_file(f)
        if s.size:
            senders_list.append(s)
            times_list.append(t)
    if not senders_list:
        raise ValueError("all recorder files are empty of spikes")
    senders = np.concatenate(senders_list)
    times = np.concatenate(times_list)

    t_start = args.t_start
    if args.t_stop is None:
        raise ValueError("t_stop must be resolved before process_trial; "
                         "main() infers it globally when not given")
    t_stop = args.t_stop
    n_bins = int(round((t_stop - t_start) / args.bin_ms))
    if n_bins <= 0:
        raise ValueError(f"empty time window: t_start={t_start}, t_stop={t_stop}")

    # map each spike to a population via the node-id ranges
    pop_idx = np.searchsorted(starts, senders, side="right") - 1
    in_pop = pop_idx >= 0
    in_pop &= senders <= ends[np.clip(pop_idx, 0, None)]
    n_orphans = int(np.count_nonzero(~in_pop))

    # map each spike to a time bin
    bin_idx = np.floor((times - t_start) / args.bin_ms).astype(np.int64)
    in_win = (times >= t_start) & (times < t_stop)

    # spikes strictly beyond t_stop indicate a too-small --t-stop or an
    # unsorted recorder file; a single spike exactly at t_stop is dropped
    # silently by the half-open grid convention
    n_trimmed = int(np.count_nonzero(in_pop & (times > t_stop)))

    keep = in_pop & in_win
    flat = pop_idx[keep] * n_bins + bin_idx[keep]
    counts = np.bincount(flat, minlength=n_pops * n_bins).astype(np.float64)
    counts = counts.reshape(n_pops, n_bins)

    if args.units == "rate":
        activity = counts / (n_neurons[:, None] * args.bin_ms * 1e-3)
        unit_label = "spikes/s per neuron"
    else:
        activity = counts
        unit_label = "spike count per bin"

    activity = gaussian_filter1d(
        activity, sigma=args.sigma_ms / args.bin_ms, axis=1, mode="reflect")

    out_dir = os.path.join(trial_dir, "measurements", args.dest_name)
    os.makedirs(out_dir, exist_ok=True)
    for p in range(n_pops):
        header = (
            "reconstructed population activity\n"
            "source: pooled spike_recorder-*.dat, split by "
            "population_nodeids.dat ranges\n"
            f"pop {p}: nodes {starts[p]}-{ends[p]} (N={n_neurons[p]})\n"
            f"t_start_ms={t_start} t_stop_ms={t_stop} "
            f"bin_ms={args.bin_ms} sigma_ms={args.sigma_ms}\n"
            f"units: {unit_label}, one value per bin"
        )
        np.savetxt(os.path.join(out_dir, f"pop_activity_{p}.dat"),
                   activity[p], fmt="%.8g", header=header)

    return {
        "n_spikes": senders.size,
        "n_orphans": n_orphans,
        "n_trimmed": n_trimmed,
        "t_stop": float(t_stop),
        "n_bins": n_bins,
        "mean_rate": float(np.mean(counts.sum(axis=1)
                                   / (n_neurons * (t_stop - t_start) * 1e-3))),
    }


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Rebuild population activity from NEST spike recorders.")
    ap.add_argument("--root", default="data_background_rate_big",
                    help="dataset root containing one directory per "
                         "background rate")
    ap.add_argument("--bin-ms", type=float, default=1.0)
    ap.add_argument("--sigma-ms", type=float, default=2.0)
    ap.add_argument("--t-start", type=float, default=500.0,
                    help="start of the reconstructed grid in ms")
    ap.add_argument("--t-stop", type=float, default=None,
                    help="end of the grid in ms (recommended: the recording "
                         "stop time); if omitted, one global value is "
                         "inferred from the last spike across ALL recorder "
                         "files in scope and used for every trial")
    ap.add_argument("--units", choices=("rate", "count"), default="rate")
    ap.add_argument("--dest-name", default="pop_activities_1ms",
                    help="output subdirectory under <trial>/measurements/")
    ap.add_argument("--recorder-glob", default="spike_recorder-*.dat")
    ap.add_argument("--only-bg", default=None,
                    help="process a single background-rate directory "
                         "(useful for running several instances in parallel)")
    args = ap.parse_args(argv)

    bg_dirs = sorted(d for d in os.listdir(args.root)
                     if os.path.isdir(os.path.join(args.root, d)))
    if args.only_bg is not None:
        bg_dirs = [d for d in bg_dirs if d == args.only_bg]
        if not bg_dirs:
            print(f"no directory named {args.only_bg!r} under {args.root}")
            return 1

    def list_trials(bg_path):
        return sorted(
            d for d in os.listdir(bg_path)
            if d.startswith("trial")
            and os.path.isdir(os.path.join(bg_path, d)))

    if args.t_stop is None:
        t_max, n_scanned = -np.inf, 0
        for bg in bg_dirs:
            bg_path = os.path.join(args.root, bg)
            for trial in list_trials(bg_path):
                for f in glob.glob(os.path.join(bg_path, trial,
                                                args.recorder_glob)):
                    v = tail_last_time(f)
                    n_scanned += 1
                    if v is not None:
                        t_max = max(t_max, v)
        if not np.isfinite(t_max):
            print("could not infer t_stop: no spikes found in scope")
            return 1
        args.t_stop = args.t_start + args.bin_ms * np.ceil(
            (t_max - args.t_start) / args.bin_ms)
        print(f"t_stop not given: using {args.t_stop:g} ms, inferred from "
              f"the last spike ({t_max:g} ms) across {n_scanned} recorder "
              f"files; pass --t-stop to override", flush=True)
        if args.only_bg is not None:
            print("note: with --only-bg the inference sees only this "
                  "directory; pass --t-stop explicitly when running "
                  "several instances in parallel", flush=True)

    failures = []
    for k, bg in enumerate(bg_dirs):
        bg_path = os.path.join(args.root, bg)
        trial_dirs = list_trials(bg_path)
        n_spikes = 0
        rates = []
        orphans = 0
        trimmed = 0
        for trial in trial_dirs:
            trial_path = os.path.join(bg_path, trial)
            try:
                info = process_trial(trial_path, args)
            except Exception as err:
                failures.append((bg, trial, f"{type(err).__name__}: {err}"))
                continue
            n_spikes += info["n_spikes"]
            rates.append(info["mean_rate"])
            orphans += info["n_orphans"]
            trimmed += info["n_trimmed"]
        msg = (f"[{k + 1}/{len(bg_dirs)}] {bg}: {len(trial_dirs)} trials, "
               f"{n_spikes / 1e6:.2f}M spikes, "
               f"mean rate {np.mean(rates) if rates else float('nan'):.2f} "
               f"spikes/s per neuron")
        if orphans:
            msg += f", WARNING: {orphans} spikes outside all node-id ranges"
        if trimmed:
            msg += (f", WARNING: {trimmed} spikes after t_stop trimmed "
                    f"(t_stop too small, or unsorted recorder files)")
        print(msg, flush=True)

    if failures:
        print(f"\n{len(failures)} trial(s) failed:")
        for bg, trial, msg in failures:
            print(f"  {bg}/{trial}: {msg}")
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())