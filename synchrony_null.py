#!/usr/bin/env python3
"""
synchrony_null.py

Motivation
----------
The spectral entropy dip across the background-rate sweep is currently tested
against a rate-matched Poisson null. That null has a white spectrum by
construction, so its entropy sits at the ceiling at every drive and the
comparison cannot fail. It establishes that the population activity is not shot
noise, which was never in question.

This script builds the null that can fail. Each neuron's spike train is given an
independent random circular time shift within the analysis window. That
preserves the neuron's firing rate exactly and its interspike-interval
distribution exactly (bar one wrapped interval), and destroys only the phase
alignment between neurons. The population activity is then rebuilt from the
shifted trains and handed back in the same units as pop_activity, so the
existing compute_FFT and compute_spectral_entropy can be run on it unchanged.

The comparison this enables is the one the abstract implicitly claims:

    dip survives the shuffle  -> the entropy minimum reflects single-neuron
                                 regularity, and calling it a population
                                 phenomenon is not supported
    dip is abolished          -> the minimum requires cross-neuron synchrony
                                 and is genuinely a network effect

A partial outcome is the likely one and is still informative: the fraction of
the dip that survives quantifies how much comes from individual regularity
versus collective alignment. Note that independently phased rhythmic neurons
still sum to a weakly rhythmic population, with oscillation amplitude scaling as
sqrt(N) rather than N, so a residual dip is expected even under a complete loss
of synchrony.

What it measures
----------------
For each requested drive and trial, writes an npz containing:

    real        (n_pops, n_samples)  population activity rebuilt from the raw
                                     spike recorders, in per-neuron spikes/s
    surrogate   (n_surr, n_pops, n_samples)  the same after per-neuron circular
                                     shifts
    n_full, pops, dt_ms, t_start     metadata needed to interpret them

The rebuilt `real` trace is also compared against the stored pop_activity file
when one is present. That comparison validates the mapping from spike-recorder
ids to populations, which is otherwise an assumption, and the script refuses to
continue if the two disagree.

Usage
-----
    # one drive first, to check the recorder mapping and see the timing
    python synchrony_null.py --data-root data_background_rate_big \\
        --drives 09.07 --trials 1 --n-surrogates 5 \\
        --n-full n_full.npy --out ./null_out

    # then the drives spanning the dip
    python synchrony_null.py --data-root data_background_rate_big \\
        --drives 05.0 06.53 07.54 08.56 09.07 10.08 11.10 12.12 14.15 17.20 20.0 \\
        --trials 2 --n-surrogates 20 \\
        --n-full n_full.npy --out ./null_out

    # then, in the notebook, feed both through the existing entropy code:
    #   z = np.load('null_out/null_09.07_trial_0.npz')
    #   a, b, _ = compute_FFT(z['real'], freq_sample=freq_sample)
    #   e_real  = compute_spectral_entropy(a, b)
    #   e_surr  = [compute_spectral_entropy(*compute_FFT(s, freq_sample=freq_sample)[:2])
    #              for s in z['surrogate']]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

POPS = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]


def read_nest_ascii(path):
    """Robust reader for NEST 3.x ASCII spike recorder output."""
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


def load_population_spikes(trial_dir, pop_indices):
    """
    Return {pop_index: (senders, times)} from the spike recorder files.

    Recorder ids sorted ascending are assumed to follow the population creation
    order. That assumption is checked downstream against pop_activity.
    """
    files = sorted(glob.glob(os.path.join(trial_dir, "spike_recorder-*.dat")))
    if not files:
        raise FileNotFoundError(
            f"no spike_recorder-*.dat in {trial_dir}. This null needs the raw "
            f"spike trains; pop_activity alone cannot support it.")
    groups = {}
    for f in files:
        m = re.search(r"spike_recorder-(\d+)-(\d+)\.dat$", os.path.basename(f))
        if m:
            groups.setdefault(int(m.group(1)), []).append(f)
    rec_ids = sorted(groups)

    out = {}
    for p in pop_indices:
        if p >= len(rec_ids):
            raise ValueError(f"population index {p} but only {len(rec_ids)} "
                             f"recorders found")
        sn, tm = [], []
        for f in groups[rec_ids[p]]:
            s, t = read_nest_ascii(f)
            sn.append(s)
            tm.append(t)
        out[p] = (np.concatenate(sn) if sn else np.zeros(0, dtype=np.int64),
                  np.concatenate(tm) if tm else np.zeros(0))
    return out


def histogram_rate(times, t_start, t_stop, dt_ms, n_neurons):
    """Population histogram in per-neuron spikes/s, matching pop_activity."""
    n_bins = int(round((t_stop - t_start) / dt_ms))
    sel = (times >= t_start) & (times < t_stop)
    idx = ((times[sel] - t_start) / dt_ms).astype(np.int64)
    counts = np.bincount(idx, minlength=n_bins)[:n_bins]
    return counts / (n_neurons * dt_ms * 1e-3)


def circular_shift_per_neuron(senders, times, t_start, t_stop, rng):
    """
    Independent random circular shift per neuron inside [t_start, t_stop).

    Rate is preserved exactly. The ISI distribution is preserved exactly except
    for the single interval that wraps the window boundary. Cross-neuron phase
    alignment is destroyed, which is the only thing this null removes.
    """
    T = t_stop - t_start
    sel = (times >= t_start) & (times < t_stop)
    s, t = senders[sel], times[sel] - t_start
    if s.size == 0:
        return t + t_start

    # one shift per distinct neuron, looked up per spike
    uniq, inv = np.unique(s, return_inverse=True)
    shifts = rng.uniform(0.0, T, size=uniq.size)
    return np.mod(t + shifts[inv], T) + t_start


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--drives", nargs="+", required=True)
    p.add_argument("--trials", type=int, default=1,
                   help="number of trials per drive to process")
    p.add_argument("--pop-indices", type=int, nargs="+", default=[0, 2, 4, 6],
                   help="default is the four excitatory populations")
    p.add_argument("--n-full", required=True,
                   help="npy of net_dict['full_num_neurons']")
    p.add_argument("--dt-ms", type=float, default=0.2)
    p.add_argument("--t-start", type=float, default=500.0)
    p.add_argument("--t-stop", type=float, default=None,
                   help="defaults to the end of the stored pop_activity trace")
    p.add_argument("--t-offset", type=float, default=500.0,
                   help="simulation time of the first pop_activity sample")
    p.add_argument("--min-drive", type=float, default=None,
                   help="skip drives below this. Below roughly 7.8 spikes/s in "
                        "this sweep the across-trial rate varies by an order of "
                        "magnitude at a fixed drive, so a surrogate comparison "
                        "there is not interpretable.")
    p.add_argument("--n-surrogates", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="./null_out")
    p.add_argument("--check-r", type=float, default=0.95,
                   help="minimum correlation between the rebuilt trace and the "
                        "stored pop_activity before the run is allowed to "
                        "continue")
    p.add_argument("--skip-check", action="store_true")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    n_full = np.load(args.n_full)
    pops = [POPS[i] for i in args.pop_indices]

    for drive in args.drives:
        if args.min_drive is not None:
            m = re.search(r"[\d.]+", drive)
            if m and float(m.group()) < args.min_drive:
                print(f"[null] {drive}: below --min-drive, skipping")
                continue
        drive_dir = os.path.join(args.data_root, drive)
        trial_dirs = sorted(glob.glob(os.path.join(drive_dir, "trial*")))
        if not trial_dirs:
            print(f"[null] {drive}: no trials found, skipping")
            continue

        for ti, trial_dir in enumerate(trial_dirs[:args.trials]):
            tag = f"{drive}_trial_{ti}"
            out_path = os.path.join(args.out, f"null_{tag}.npz")
            if os.path.exists(out_path):
                print(f"[null] {tag}: exists, skipping")
                continue

            # reference trace, used both to set the window and to validate the
            # recorder-to-population mapping
            ref = None
            ref_path = os.path.join(trial_dir, "measurements", "pop_activities",
                                    f"pop_activity_{args.pop_indices[0]}.dat")
            if os.path.exists(ref_path):
                ref = np.loadtxt(ref_path)
            t_stop = args.t_stop
            if t_stop is None:
                if ref is None:
                    raise SystemExit("--t-stop is required when no "
                                     "pop_activity file is present")
                t_stop = args.t_offset + args.dt_ms * len(ref)

            spikes = load_population_spikes(trial_dir, args.pop_indices)

            real = np.array([
                histogram_rate(spikes[p][1], args.t_start, t_stop,
                               args.dt_ms, n_full[p])
                for p in args.pop_indices])

            # validate the mapping before spending time on surrogates
            if ref is not None and not args.skip_check:
                i0 = int(round((args.t_start - args.t_offset) / args.dt_ms))
                seg = ref[i0:i0 + real.shape[1]]
                n = min(len(seg), real.shape[1])
                # Correlation, not max absolute difference. A wrong
                # population gives a correlation near zero, while binning
                # rounding at the sub-sample level gives a large max
                # difference on a small count with no loss of correlation.
                r = float(np.corrcoef(seg[:n], real[0, :n])[0, 1])
                rate_err = abs(seg[:n].mean() - real[0, :n].mean()) / \
                    max(seg[:n].mean(), 1e-12)
                print(f"[null] {tag}: rebuilt vs stored pop_activity, "
                      f"r = {r:.4f}, mean rate differs by {rate_err:.2%}")
                if r < args.check_r or rate_err > 0.02:
                    raise SystemExit(
                        f"{tag}: rebuilt trace does not match the stored "
                        f"pop_activity (r = {r:.3f}, rate error "
                        f"{rate_err:.1%}). The recorder-to-population mapping "
                        f"is probably wrong, or n_full is misindexed. Inspect "
                        f"the recorder ids before trusting anything "
                        f"downstream.")

            rng = np.random.default_rng(args.seed + hash(tag) % 10000)
            surr = np.empty((args.n_surrogates, len(args.pop_indices),
                             real.shape[1]))
            for s in range(args.n_surrogates):
                for k, p in enumerate(args.pop_indices):
                    sen, tm = spikes[p]
                    shifted = circular_shift_per_neuron(
                        sen, tm, args.t_start, t_stop, rng)
                    surr[s, k] = histogram_rate(shifted, args.t_start, t_stop,
                                                args.dt_ms, n_full[p])
                print(f"[null] {tag}: surrogate {s + 1}/{args.n_surrogates}",
                      flush=True)

            np.savez_compressed(
                out_path, real=real, surrogate=surr,
                pops=np.array(pops), n_full=n_full[args.pop_indices],
                dt_ms=args.dt_ms, t_start=args.t_start, t_stop=t_stop)
            print(f"[null] {tag}: wrote {out_path}")

    print("[null] done")


if __name__ == "__main__":
    sys.exit(main() or 0)