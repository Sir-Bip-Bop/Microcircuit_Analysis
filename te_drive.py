#!/usr/bin/env python3
"""
te_drive.py

Motivation
----------
The background-rate sweep of the Potjans-Diesmann microcircuit currently yields
undirected, single-population regularity measures (spectral entropy, burst
amplitude versus inter-burst interval correlation). Those quantify how ordered a
population is, not whether one population predicts another. This script adds the
directed half of the picture: it estimates the transfer entropy between the eight
populations as a function of the external drive, so that the question "where in
the drive sweep does the effective connectivity best reflect the anatomy" can be
answered on the same axis as the entropy curves.

The design is driven by the three failure modes that make naive TE(drive) curves
worthless:

1. Rate confound. Firing rate rises with drive, and KSG bias depends on rate and
   on sample size. A monotonic estimator bias is indistinguishable from a real
   drive dependence. This script therefore optionally applies binomial thinning
   to every population so that all drive levels are compared at a common mean
   count per bin, and always reports the raw and thinned estimates side by side.
2. Periodicity. In a strongly rhythmic regime, KSG-type estimators return large
   and nearly symmetric values on deterministically coupled signals. The
   surrogate here is a cyclic time shift of the source, which preserves the
   rhythm and the autocorrelation and destroys only the cross-timing. Directional
   asymmetry, not magnitude, is the validity check.
3. Selection bias. The source-target delay u is chosen by maximising TE over a
   scan. The surrogate distribution is therefore built by running each surrogate
   through the identical max-over-u selection, so the null carries the same
   optimistic bias as the estimate.

What it measures
----------------
For every ordered pair of populations (X -> Y), every background rate, and every
source-target delay u in a scan:

    TE(X->Y, u) = I( Y_{t+1} ; X_{t+1-u}^{(l)} | Y_t^{(k)} , Z_t )

estimated with the Frenzel-Pompe conditional-mutual-information form of the KSG
estimator (max-norm, k nearest neighbours, digamma corrections). Z is an optional
conditioning set that controls for common drive:

    --condition none    Z is empty, plain pairwise TE
    --condition global  Z is the past of the summed population rate excluding
                        source and target
    --condition topN    Z is the past of the N other populations with the
                        strongest pairwise TE into the same target, read from a
                        completed pairwise pass

Reported per pair and drive: the raw TE at the selected delay, the mean and
standard deviation of the surrogate distribution, the effective TE
(raw minus surrogate mean), a surrogate-based p value, and the selected delay.
The active information storage AIS(Y) = I(Y_{t+1}; Y_t^{(k)}) is computed per
population and drive, in the same units, as the principled counterpart to
spectral entropy.

Stages, each resumable, each writing its own checkpoint:

    build     NEST spike recorder files -> per-trial 1 ms population counts
    embed     AIS scan over history length k, per population and drive
    te        the TE scan itself, checkpointed per drive
    plot      TE(drive) curves, TE matrices, TE versus structural connectivity
    selftest  verifies the estimator against a linear Gaussian system whose TE
              is known analytically. Run this first on any new machine.

Cost note. With the defaults (16 drives, 56 ordered pairs, delay scan of 6, 19
surrogates each run through the full scan, 20000 sample points) this is about
1.1e5 CMI evaluations. On 24 threads expect a few hours. Reduce --n-surrogates
or --max-points to trade precision for time. Every stage checkpoints per drive,
so an interrupted run resumes.

Usage
-----
    # 0. Sanity check the estimator, takes about a minute
    python te_drive.py selftest

    # 1. Population sizes, once, from the microcircuit parameters
    #      import numpy as np
    #      from network_params import net_dict
    #      np.save('n_full.npy', np.array(net_dict['full_num_neurons']))
    #      np.save('conn_probs.npy', np.array(net_dict['conn_probs']))

    # 2. Read the same pop_activity traces the spectral entropy notebook uses,
    #    at their native 0.2 ms, decimated by 5 to 1 ms for the TE
    python te_drive.py build \
        --data-root data_background_rate_big --out-dir ./te_out \
        --source popact --dt-ms 0.2 --decimate 5 \
        --t-offset 500 --t-start 1000 --t-stop 5500 \
        --n-full ./n_full.npy

    # 3. History length per population and drive, plus AIS
    python te_drive.py embed --out-dir ./te_out --k-max 6 --n-jobs 20

    # 4. Pairwise TE. Measure f0 first and set --u-max-ms above one full period
    #    (1000/f0), or a peak at the cycle period cannot be distinguished from
    #    a genuine short-latency transfer. Step at 1 ms or finer or the scan
    #    walks over the real synaptic delays entirely.
    python te_drive.py te --out-dir ./te_out --period-ms 34 \
        --u-max-ms 40 --u-step-ms 1 --n-surrogates 19 \
        --max-points 20000 --thin --n-jobs 20

    # 5. Optional second pass conditioning on the two strongest other inputs
    python te_drive.py te --out-dir ./te_out \
        --condition topN --top-n 2 --tag cond2 --n-jobs 20

    # 6. Figures, including the delay-profile diagnostic
    python te_drive.py plot --out-dir ./te_out --conn-probs ./conn_probs.npy
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

# Population order as created by the NEST microcircuit example. Override with
# --pop-order if your recorder ids map differently.
POPS = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]

LAYER_COLOURS = {"L23": "#0072B2", "L4": "#D55E00", "L5": "#009E73", "L6": "#CC79A7"}


def pop_style(pop):
    """Line style for a population, matching the paper figure convention."""
    layer = pop[:-1]
    colour = LAYER_COLOURS[layer]
    if pop.endswith("E"):
        return dict(color=colour, linestyle="-", marker="o")
    return dict(color=colour, linestyle="--", marker="s")


# ----------------------------------------------------------------------------
# Estimator core
# ----------------------------------------------------------------------------

def _prepare(x, rng, dither=1e-6):
    """
    Standardise columns and add tie-breaking noise. KSG needs unique points.

    The dither scale is a real parameter, not a formality. Population counts are
    integers, so after standardising, adjacent values sit 1/sd apart, typically
    0.1 to 0.5. Thousands of exact duplicates land in one dither cloud, and if
    that cloud is narrower than the offset used for strict neighbour counting
    the k-th neighbour distance collapses and the estimate diverges. 1e-6 is
    five orders of magnitude below the data spacing and six above the offset.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"expected (n, d), got {x.shape}")
    bad = ~np.isfinite(x)
    if bad.any():
        raise ValueError(f"{bad.sum()} non-finite values of {x.size} in the "
                         f"estimator input. Run 'te_drive.py check'.")
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    x = (x - x.mean(axis=0)) / sd
    return x + dither * rng.standard_normal(x.shape)


def _count_within(x, eps, workers=1):
    """
    Number of other points strictly inside radius eps, max-norm.

    Two guards matter here. A non-finite radius makes scipy allocate on a
    garbage size and abort the process rather than raise, so it is checked
    explicitly. And the radius is floored at zero: a point is always inside its
    own radius, so the count can never legitimately be zero, and letting a zero
    through puts digamma(0) = -inf into the estimate.
    """
    eps = np.asarray(eps, dtype=np.float64)
    if not np.all(np.isfinite(eps)):
        nbad = int((~np.isfinite(eps)).size - np.isfinite(eps).sum())
        fin = eps[np.isfinite(eps)]
        raise ValueError(
            f"non-finite neighbour radius on {nbad} of {eps.size} points. "
            f"n_points={x.shape[0]}, n_dims={x.shape[1]}, "
            f"finite eps range=[{fin.min() if fin.size else float('nan'):.3g}, "
            f"{fin.max() if fin.size else float('nan'):.3g}], "
            f"distinct rows={len(np.unique(x, axis=0))}. This means the tree "
            f"could not find enough distinct neighbours.")
    r = np.maximum(eps - 1e-12, 0.0)
    tree = cKDTree(x)
    # workers is deliberately pinned to 1. scipy's threaded query_ball_point
    # with an array of per-point radii and return_length=True corrupts the heap
    # on some builds, which aborts the process rather than raising. Parallelism
    # belongs at the pair level, not inside the neighbour search.
    n = tree.query_ball_point(x, r=r, p=np.inf,
                              return_length=True, workers=1)
    return np.maximum(np.asarray(n, dtype=np.int64) - 1, 0)


def cmi_ksg(a, b, c=None, k=4, workers=1):
    """
    Conditional mutual information I(A;B|C) by the Frenzel-Pompe form of KSG.

    With C empty this reduces to the standard KSG algorithm-1 mutual information.
    Inputs are (n, d) arrays, already standardised and dithered. Returns nats.
    """
    n = a.shape[0]
    have_c = c is not None and c.shape[1] > 0
    joint = np.hstack([a, b, c]) if have_c else np.hstack([a, b])

    tree = cKDTree(joint)
    dists, _ = tree.query(joint, k=k + 1, p=np.inf, workers=1)
    eps = dists[:, k]

    # Fraction of points whose k nearest neighbours are exact duplicates. Above
    # a fifth of the sample the estimate is measuring ties rather than
    # dependence, and reporting a number would be worse than reporting nothing.
    degenerate = float(np.mean(eps <= 1e-9))
    if degenerate > 0.2:
        return float("nan")

    if have_c:
        n_ac = _count_within(np.hstack([a, c]), eps, workers)
        n_bc = _count_within(np.hstack([b, c]), eps, workers)
        n_c = _count_within(c, eps, workers)
        val = digamma(k) - np.mean(
            digamma(n_ac + 1) + digamma(n_bc + 1) - digamma(n_c + 1)
        )
    else:
        n_a = _count_within(a, eps, workers)
        n_b = _count_within(b, eps, workers)
        val = digamma(k) + digamma(n) - np.mean(digamma(n_a + 1) + digamma(n_b + 1))

    return float(val)


# ----------------------------------------------------------------------------
# Embedding construction
# ----------------------------------------------------------------------------

def _past_block(sig, t_idx, n_lags, offset):
    """
    Columns [sig[t - offset], sig[t - offset - 1], ...] for n_lags lags.
    offset = 0 gives the target past starting at t, offset = u - 1 gives the
    source past starting at t + 1 - u.
    """
    return np.stack([sig[t_idx - offset - j] for j in range(n_lags)], axis=1)


def te_vectors(src_trials, tgt_trials, k_hist, l_hist, u):
    """
    Build (y_future, y_past, x_past) pooled over trials without crossing trial
    boundaries. Each element of *_trials is a 1D array for one trial.
    """
    yf, yp, xp = [], [], []
    t0 = max(k_hist - 1, u + l_hist - 2)
    for src, tgt in zip(src_trials, tgt_trials):
        T = len(tgt)
        if T - 2 < t0:
            continue
        t = np.arange(t0, T - 1)
        yf.append(tgt[t + 1][:, None])
        yp.append(_past_block(tgt, t, k_hist, 0))
        xp.append(_past_block(src, t, l_hist, u - 1))
    return np.vstack(yf), np.vstack(yp), np.vstack(xp)


def ais_vectors(tgt_trials, k_hist):
    yf, yp = [], []
    t0 = k_hist - 1
    for tgt in tgt_trials:
        T = len(tgt)
        if T - 2 < t0:
            continue
        t = np.arange(t0, T - 1)
        yf.append(tgt[t + 1][:, None])
        yp.append(_past_block(tgt, t, k_hist, 0))
    return np.vstack(yf), np.vstack(yp)


def _subsample(rng, n, max_points):
    if max_points is None or n <= max_points:
        return np.arange(n)
    return rng.choice(n, size=max_points, replace=False)


# ----------------------------------------------------------------------------
# Stage: build
# ----------------------------------------------------------------------------

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
                continue  # column header line
            senders.append(s)
            times.append(t)
    return np.asarray(senders, dtype=np.int64), np.asarray(times, dtype=np.float64)


def bin_trial(trial_dir, t_start, t_stop, bin_ms, n_pops):
    """
    Return (n_pops, n_bins) spike counts for one trial. Files are grouped by the
    recorder id embedded in the filename, and recorder ids sorted ascending are
    assumed to follow the population creation order.
    """
    files = sorted(glob.glob(os.path.join(trial_dir, "spike_recorder-*.dat")))
    if not files:
        raise FileNotFoundError(f"no spike_recorder files in {trial_dir}")

    groups = {}
    for f in files:
        m = re.search(r"spike_recorder-(\d+)-(\d+)\.dat$", os.path.basename(f))
        if m is None:
            continue
        groups.setdefault(int(m.group(1)), []).append(f)

    rec_ids = sorted(groups)
    if len(rec_ids) != n_pops:
        raise ValueError(
            f"{trial_dir}: found {len(rec_ids)} recorder ids, expected {n_pops}"
        )

    n_bins = int(round((t_stop - t_start) / bin_ms))
    counts = np.zeros((n_pops, n_bins), dtype=np.int32)

    for p, rid in enumerate(rec_ids):
        for f in groups[rid]:
            _, times = read_nest_ascii(f)
            if times.size == 0:
                continue
            sel = (times >= t_start) & (times < t_stop)
            if not np.any(sel):
                continue
            idx = ((times[sel] - t_start) / bin_ms).astype(np.int64)
            counts[p] += np.bincount(idx, minlength=n_bins)[:n_bins].astype(np.int32)
    return counts


def _build_popact(args):
    drive_dirs = sorted(
        d for d in glob.glob(os.path.join(args.data_root, "*")) if os.path.isdir(d)
    )
    if not drive_dirs:
        raise SystemExit(f"no drive directories under {args.data_root}")

    n_full = np.load(args.n_full) if args.n_full else None
    pop_idx = args.pop_indices
    pops = [args.pop_order[i] if i < len(args.pop_order) else f"pop{i}"
            for i in range(len(pop_idx))]
    # sample indices relative to the start of the trace, which begins at
    # t_offset ms (the transient the simulation already discarded)
    s_start = int(round((args.t_start - args.t_offset) / args.dt_ms))
    s_stop = int(round((args.t_stop - args.t_offset) / args.dt_ms))
    bin_ms = args.dt_ms * args.decimate

    meta = {"bin_ms": bin_ms, "dt_ms": args.dt_ms, "decimate": args.decimate,
            "t_start": args.t_start, "t_stop": args.t_stop, "pops": pops,
            "source": "popact", "units": "counts" if n_full is not None else "rate",
            "drives": []}

    worst_resid = 0.0
    for d in drive_dirs:
        drive = os.path.basename(d)
        meta["drives"].append(drive)
        out = os.path.join(args.out_dir, f"counts_{drive}.npz")
        if os.path.exists(out) and not args.force:
            print(f"[build] {drive}: checkpoint present, skipping")
            continue
        trials = sorted(glob.glob(os.path.join(d, "trial*")))
        if args.n_trials:
            trials = trials[:args.n_trials]
        arrs = []
        for t in trials:
            rates = read_popact_trial(t, pop_idx, s_start, s_stop)
            if n_full is not None:
                sel = np.asarray(n_full)[list(pop_idx)]
                counts, resid = popact_to_counts(rates, sel, args.dt_ms)
                worst_resid = max(worst_resid, resid)
            else:
                counts = rates
            arrs.append(decimate_counts(counts, args.decimate))
        counts = np.stack(arrs, axis=0)
        np.savez_compressed(out, counts=counts, pops=np.array(pops))
        print(f"[build] {drive}: {counts.shape} at {bin_ms:.2f} ms bins, "
              f"mean/bin " + ", ".join(f"{p}={c:.2f}" for p, c in
                                       zip(pops, counts.mean(axis=(0, 2)))))

    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    if n_full is not None:
        print(f"[build] integrality check: max deviation from integer counts "
              f"= {worst_resid:.4g}")
        if worst_resid > 1e-6:
            print("[build] WARNING: pop_activity is NOT a raw histogram. It has "
                  "been smoothed or filtered upstream. Short-lag TE will be "
                  "contaminated by the kernel. Rebuild from the spike recorders "
                  "with --source spikes, or scan delays beyond the kernel width "
                  "only.")
    else:
        print("[build] no --n-full given: traces kept as rates, so binomial "
              "rate matching (--thin) is unavailable and the integrality check "
              "was skipped.")
    print(f"[build] wrote {len(meta['drives'])} drive checkpoints")


def read_popact_trial(trial_dir, pop_indices, s_start, s_stop):
    """
    Load measurements/pop_activities/pop_activity_<p>.dat for one trial.

    These are the same traces the spectral entropy notebook consumes, sampled at
    dt = 0.2 ms and expressed as per-neuron rate in spikes/s. Using them rather
    than re-binning the raw spike recorders guarantees that TE and spectral
    entropy are computed on the identical signal.
    """
    base = os.path.join(trial_dir, "measurements", "pop_activities")
    rows = []
    for p in pop_indices:
        f = os.path.join(base, f"pop_activity_{p}.dat")
        rows.append(_fast_loadtxt(f)[s_start:s_stop])
    return np.array(rows, dtype=np.float64)


def _fast_loadtxt(path):
    """np.loadtxt parses these 300k-line files at a few seconds each, which is
    hours across the whole sweep. pandas is roughly twenty times faster."""
    try:
        import pandas as pd
        df = pd.read_csv(path, header=None, sep=r"\s+", comment="#",
                         dtype=np.float64, skip_blank_lines=True)
        if df.shape[1] != 1:
            raise ValueError(f"{path}: expected 1 column, got {df.shape[1]}")
        a = df.values.ravel()
    except ValueError:
        raise
    except Exception:
        a = np.loadtxt(path)
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        n_bad = int((~np.isfinite(a)).sum())
        raise ValueError(f"{path}: {n_bad} non-finite values in the trace")
    return a


def popact_to_counts(rates, n_full, dt_ms, tol=1e-6):
    """
    Recover integer spike counts per bin from the per-neuron rate trace.

    counts = rate * N * dt. If the trace was written as a raw histogram this is
    exactly integral, and the deviation from integrality is therefore a direct
    test of whether the trace has been smoothed or filtered upstream. Any
    smoothing manufactures cross-correlation at short lags and invalidates the
    delay scan, so this check is run at build time and reported loudly.
    """
    scale = np.asarray(n_full, dtype=np.float64)[:, None] * (dt_ms * 1e-3)
    raw = rates * scale
    resid = np.abs(raw - np.round(raw))
    return np.round(raw), float(resid.max())


def decimate_counts(counts, factor):
    """Aggregate counts by summing adjacent bins. Summing is the correct
    operation for counts: it is exactly the histogram at the coarser bin width,
    unlike averaging or subsampling."""
    if factor <= 1:
        return counts
    n = (counts.shape[-1] // factor) * factor
    c = counts[..., :n]
    return c.reshape(*c.shape[:-1], n // factor, factor).sum(axis=-1)


def stage_build(args):
    os.makedirs(args.out_dir, exist_ok=True)
    if args.source == "popact":
        return _build_popact(args)
    drive_dirs = sorted(
        d for d in glob.glob(os.path.join(args.data_root, "*")) if os.path.isdir(d)
    )
    if not drive_dirs:
        raise SystemExit(f"no drive directories under {args.data_root}")

    meta = {"bin_ms": args.bin_ms, "t_start": args.t_start, "t_stop": args.t_stop,
            "pops": args.pop_order, "drives": []}

    for d in drive_dirs:
        drive = os.path.basename(d)
        out = os.path.join(args.out_dir, f"counts_{drive}.npz")
        meta["drives"].append(drive)
        if os.path.exists(out) and not args.force:
            print(f"[build] {drive}: checkpoint present, skipping")
            continue
        trials = sorted(glob.glob(os.path.join(d, "trial*")))
        arrs = []
        for t in trials:
            arrs.append(bin_trial(t, args.t_start, args.t_stop, args.bin_ms,
                                  len(args.pop_order)))
            print(f"[build] {drive}/{os.path.basename(t)} done", flush=True)
        counts = np.stack(arrs, axis=0)  # (n_trials, n_pops, n_bins)
        np.savez_compressed(out, counts=counts, pops=np.array(args.pop_order))
        rates = counts.mean(axis=(0, 2)) / (args.bin_ms * 1e-3)
        print(f"[build] {drive}: {counts.shape}, mean counts/bin "
              + ", ".join(f"{p}={c:.2f}" for p, c in
                          zip(args.pop_order, counts.mean(axis=(0, 2)))))

    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[build] wrote {len(meta['drives'])} drive checkpoints")


# ----------------------------------------------------------------------------
# Loading and thinning
# ----------------------------------------------------------------------------

def load_meta(out_dir):
    with open(os.path.join(out_dir, "meta.json")) as fh:
        return json.load(fh)


def load_counts(out_dir, drive, validate=True):
    z = np.load(os.path.join(out_dir, f"counts_{drive}.npz"))
    c = z["counts"].astype(np.float64)
    if validate and not np.all(np.isfinite(c)):
        n = int((~np.isfinite(c)).sum())
        raise ValueError(
            f"drive {drive}: {n} non-finite values in the built counts. "
            f"Run 'te_drive.py check --out-dir <dir>' for the breakdown, then "
            f"either rebuild that drive with --force or exclude it with "
            f"--drives.")
    return c


def thinning_table(out_dir, drives):
    """Per population, the smallest mean count per bin across all drives."""
    means = []
    for d in drives:
        means.append(load_counts(out_dir, d, validate=False).mean(axis=(0, 2)))
    means = np.array(means)                     # (n_drives, n_pops)
    return means, means.min(axis=0)             # target = min over drives


def apply_thinning(counts, mean_now, target, rng):
    """Binomial thinning of binned counts. Exactly equivalent to independent
    per-spike thinning, so the timing structure is preserved while the rate is
    matched across drives."""
    out = np.empty_like(counts)
    for p in range(counts.shape[1]):
        prob = min(1.0, target[p] / max(mean_now[p], 1e-12))
        if prob >= 1.0:
            out[:, p] = counts[:, p]
        else:
            out[:, p] = rng.binomial(counts[:, p].astype(np.int64), prob)
    return out.astype(np.float64)


def smooth(counts, sigma_bins):
    if sigma_bins <= 0:
        return counts
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(counts, sigma=sigma_bins, axis=-1, mode="nearest")


# ----------------------------------------------------------------------------
# Stage: embed (history length selection plus AIS)
# ----------------------------------------------------------------------------

def stage_embed(args):
    meta = load_meta(args.out_dir)
    drives = meta["drives"]
    pops = meta["pops"]
    means, target = thinning_table(args.out_dir, drives)

    rows = []
    for di, drive in enumerate(drives):
        rng = np.random.default_rng(args.seed)
        counts = load_counts(args.out_dir, drive)
        if args.thin:
            counts = apply_thinning(counts, means[di], target, rng)
        counts = smooth(counts, args.smooth_sigma)

        for p, pop in enumerate(pops):
            trials = [counts[tr, p, :] for tr in range(counts.shape[0])]
            best_k, best_ais = 1, -np.inf
            curve = []
            for k_hist in range(1, args.k_max + 1):
                yf, yp = ais_vectors(trials, k_hist)
                idx = _subsample(np.random.default_rng(args.seed), len(yf),
                                 args.max_points)
                a = _prepare(yf[idx], rng)
                b = _prepare(yp[idx], rng)
                val = cmi_ksg(a, b, None, k=args.knn, workers=1)
                curve.append(val)
                if val > best_ais:
                    best_k, best_ais = k_hist, val
            rows.append(dict(drive=drive, pop=pop, k_hist=best_k,
                             ais=best_ais, ais_curve=curve))
            print(f"[embed] {drive} {pop}: k={best_k} AIS={best_ais:.4f} nats",
                  flush=True)

    np.save(os.path.join(args.out_dir, "embed.npy"), np.array(rows, dtype=object))
    print("[embed] wrote embed.npy")


def load_embed(out_dir):
    rows = np.load(os.path.join(out_dir, "embed.npy"), allow_pickle=True)
    khist = {(r["drive"], r["pop"]): r["k_hist"] for r in rows}
    ais = {(r["drive"], r["pop"]): r["ais"] for r in rows}
    return khist, ais


# ----------------------------------------------------------------------------
# Stage: te
# ----------------------------------------------------------------------------

def _cyclic_shift(trials, rng, min_shift):
    out = []
    for x in trials:
        T = len(x)
        s = rng.integers(min_shift, T - min_shift)
        out.append(np.roll(x, s))
    return out


def _te_scan(src_trials, tgt_trials, cond_trials, k_hist, l_hist, u_list,
             knn, max_points, seed, workers):
    """Delay scan. Returns (best_te, best_u, profile over u_list)."""
    rng = np.random.default_rng(seed)
    best, best_u = -np.inf, u_list[0]
    profile = []
    for u in u_list:
        yf, yp, xp = te_vectors(src_trials, tgt_trials, k_hist, l_hist, u)
        cond = [yp]
        for c in cond_trials:
            _, _, cp = te_vectors(c, tgt_trials, k_hist, l_hist, u)
            cond.append(cp)
        C = np.hstack(cond)
        idx = _subsample(np.random.default_rng(seed), len(yf), max_points)
        a = _prepare(yf[idx], rng)
        b = _prepare(xp[idx], rng)
        c_arr = _prepare(C[idx], rng)
        val = cmi_ksg(a, b, c_arr, k=knn, workers=workers)
        profile.append(val)
        if val > best:
            best, best_u = val, u
    if not np.isfinite(best):
        return float("nan"), u_list[0], np.array(profile)
    return best, best_u, np.array(profile)


def _pair_job(payload):
    """One ordered pair at one drive: estimate plus its surrogate distribution.

    Failures are captured and returned rather than raised. A single degenerate
    pair should cost that pair, not the hours already spent on the rest of the
    drive.
    """
    try:
        return _pair_job_inner(payload)
    except Exception as e:
        si, ti = payload[1], payload[2]
        return dict(source=si, target=ti, u=0, te_raw=float("nan"),
                    surr_mean=float("nan"), surr_std=float("nan"),
                    te_eff=float("nan"), p=float("nan"),
                    u_list=np.array(payload[6]),
                    profile=np.full(len(payload[6]), np.nan),
                    profile_surr=np.zeros(len(payload[6])),
                    error=f"{type(e).__name__}: {e}")


def _pair_job_inner(payload):
    (counts, si, ti, cond_idx, k_hist, l_hist, u_list, knn, max_points,
     n_surr, seed, min_shift, workers) = payload

    n_trials = counts.shape[0]
    src = [counts[tr, si, :] for tr in range(n_trials)]
    tgt = [counts[tr, ti, :] for tr in range(n_trials)]
    cond = [[counts[tr, ci, :] for tr in range(n_trials)] for ci in cond_idx]

    te_raw, u_star, profile = _te_scan(src, tgt, cond, k_hist, l_hist, u_list,
                                       knn, max_points, seed, workers)

    if n_surr == 0:
        return dict(source=si, target=ti, u=u_star, te_raw=te_raw,
                    surr_mean=0.0, surr_std=float("nan"),
                    te_eff=te_raw, p=float("nan"),
                    u_list=np.array(u_list), profile=profile,
                    profile_surr=np.zeros(len(u_list)))

    rng = np.random.default_rng(seed + 9973)
    surr = np.empty(n_surr)
    surr_prof = np.empty((n_surr, len(u_list)))
    for s in range(n_surr):
        src_s = _cyclic_shift(src, rng, min_shift)
        # the surrogate goes through the identical max-over-u selection, so the
        # null inherits the same selection bias as the estimate
        surr[s], _, surr_prof[s] = _te_scan(src_s, tgt, cond, k_hist, l_hist,
                                            u_list, knn, max_points, seed,
                                            workers)

    p = (1.0 + np.sum(surr >= te_raw)) / (1.0 + n_surr)
    return dict(source=si, target=ti, u=u_star, te_raw=te_raw,
                surr_mean=float(surr.mean()), surr_std=float(surr.std()),
                te_eff=te_raw - float(surr.mean()), p=float(p),
                u_list=np.array(u_list), profile=profile,
                profile_surr=surr_prof.mean(axis=0))



def _run_jobs(jobs, n_jobs, on_result, pair_index=None, backend="serial"):
    """
    Run pair jobs, tolerating worker deaths that are not Python exceptions.

    A native abort inside the neighbour search kills the worker process
    outright. ProcessPoolExecutor then reports BrokenProcessPool and discards
    every in-flight job, so one bad pair can destroy an hour of completed work.
    The remedy is to fall back to one isolated single-worker pool per remaining
    job: a crash then costs exactly that pair, which is recorded as a failure
    and skipped.
    """
    import multiprocessing as mp
    from concurrent.futures.process import BrokenProcessPool

    ctx = mp.get_context("spawn")

    def _fix0(r, i):
        if pair_index is not None:
            r["source"], r["target"] = pair_index[i]
        return r

    if backend == "thread":
        # cKDTree releases the GIL during query and query_ball_point, so plain
        # threads parallelise across pairs without pickling, forking, or the
        # pool machinery that deadlocked. Each pair still searches serially.
        from concurrent.futures import ThreadPoolExecutor
        out = []
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(_pair_job, jobs[i]): i for i in range(len(jobs))}
            for f in as_completed(futs):
                r = _fix0(f.result(), futs[f])
                out.append(r)
                on_result(r)
        return out

    if backend == "serial":
        # No process pool at all. Pairs run in this process one after another
        # and scipy's own threading parallelises each neighbour search via the
        # workers argument. Slower per pair than a working pool, but it cannot
        # deadlock, cannot lose in-flight jobs, and a failure gives a real
        # traceback instead of silence.
        out = []
        for i, job in enumerate(jobs):
            payload = list(job)
            payload[-1] = n_jobs          # scipy workers, not processes
            r = _fix0(_pair_job(tuple(payload)), i)
            out.append(r)
            on_result(r)
        return out

    def _fix(r, i):
        if pair_index is not None:
            r["source"], r["target"] = pair_index[i]
        return r

    results = []
    pending = set(range(len(jobs)))

    if n_jobs > 1:
        try:
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
                futs = {ex.submit(_pair_job, jobs[i]): i for i in sorted(pending)}
                for f in as_completed(futs):
                    r = _fix(f.result(), futs[f])
                    pending.discard(futs[f])
                    results.append(r)
                    on_result(r)
        except BrokenProcessPool:
            print(f"[te] a worker died at C level. Isolating the remaining "
                  f"{len(pending)} pairs, one process each.", flush=True)

    for i in sorted(pending):
        si, ti, u_list = jobs[i][1], jobs[i][2], jobs[i][6]
        try:
            with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
                r = _fix(ex.submit(_pair_job, jobs[i]).result(), i)
        except BrokenProcessPool:
            r = dict(source=si, target=ti, u=0, te_raw=float("nan"),
                     surr_mean=float("nan"), surr_std=float("nan"),
                     te_eff=float("nan"), p=float("nan"),
                     u_list=np.array(u_list),
                     profile=np.full(len(u_list), np.nan),
                     profile_surr=np.zeros(len(u_list)),
                     error="worker aborted natively in the neighbour search")
            _fix(r, i)
        results.append(r)
        on_result(r)
    return results


def stage_te(args):
    meta = load_meta(args.out_dir)
    drives, pops = meta["drives"], meta["pops"]
    n_pops = len(pops)
    khist, _ = load_embed(args.out_dir)
    means, target = thinning_table(args.out_dir, drives)

    bin_ms = meta.get("bin_ms", 1.0)
    if args.u_grid_ms:
        # An explicit grid lets one run cover synaptic latencies at fine
        # resolution and the cycle period at coarse resolution. A uniform grid
        # fine enough for the former is unaffordable out to the latter.
        u_list = sorted({max(1, int(round(u / bin_ms)))
                         for u in args.u_grid_ms})
    else:
        u_max_bins = max(1, int(round(args.u_max_ms / bin_ms)))
        u_step = max(1, int(round(args.u_step_ms / bin_ms)))
        u_list = list(range(1, u_max_bins + 1, u_step))
    print(f"[te] bin = {bin_ms:.2f} ms, delay scan {u_list[0]*bin_ms:.2f} to "
          f"{u_list[-1]*bin_ms:.2f} ms in {len(u_list)} steps")
    u_reach = max(u_list) * bin_ms
    if args.period_ms and u_reach < args.period_ms:
        print(f"[te] NOTE: the scan reaches {u_reach:.0f} ms but one "
              f"period of the population rhythm is {args.period_ms:.0f} ms. "
              f"A peak at the cycle period cannot be seen, and the "
              f"max-over-delay selection is truncated.", flush=True)

    prev = None
    if args.condition == "topN":
        prev_path = os.path.join(args.out_dir, f"te_{args.prev_tag}.npy")
        prev = np.load(prev_path, allow_pickle=True)

    sel_drives = list(enumerate(drives))
    if args.min_drive is not None:
        sel_drives = [(i, d) for i, d in sel_drives
                      if float(re.search(r"[\d.]+", str(d)).group())
                      >= args.min_drive]
    if args.drive_stride > 1:
        sel_drives = sel_drives[::args.drive_stride]
    if args.drives:
        keep = set(args.drives)
        sel_drives = [(i, d) for i, d in sel_drives if d in keep]
    print(f"[te] {len(sel_drives)} of {len(drives)} drives selected", flush=True)

    for di, drive in sel_drives:
        ckpt = os.path.join(args.out_dir, f"te_{args.tag}_{drive}.npy")
        if os.path.exists(ckpt) and not args.force:
            print(f"[te] {drive}: checkpoint present, skipping")
            continue

        rng = np.random.default_rng(args.seed)
        counts = load_counts(args.out_dir, drive)
        if args.thin:
            counts = apply_thinning(counts, means[di], target, rng)
        counts = smooth(counts, args.smooth_sigma)

        jobs, pair_index = [], []
        for ti in range(n_pops):
            k_hist = khist[(drive, pops[ti])]
            for si in range(n_pops):
                if si == ti:
                    continue
                if args.condition == "none":
                    cond_idx = []
                elif args.condition == "global":
                    cond_idx = [j for j in range(n_pops) if j not in (si, ti)]
                    cond_idx = ["SUM", cond_idx]
                elif args.condition == "topN":
                    cand = [r for r in prev
                            if r["drive"] == drive and r["target"] == ti
                            and r["source"] != si and r["p"] <= 0.05]
                    cand.sort(key=lambda r: -r["te_eff"])
                    cond_idx = [r["source"] for r in cand[:args.top_n]]
                else:
                    raise ValueError(args.condition)

                if args.condition == "global":
                    # append the summed rate of the remaining populations as an
                    # extra channel rather than each population separately
                    others = cond_idx[1]
                    extra = counts[:, others, :].sum(axis=1, keepdims=True)
                    local = np.concatenate([counts, extra], axis=1)
                    use_counts, use_cond = local, [local.shape[1] - 1]
                else:
                    use_counts, use_cond = counts, cond_idx

                # slice out only the rows this pair needs and remap the
                # indices, so the payload is a few MB rather than the whole
                # drive array repeated once per pair
                take = [si, ti] + list(use_cond)
                sub = np.ascontiguousarray(use_counts[:, take, :])
                jobs.append((sub, 0, 1, list(range(2, len(take))), k_hist,
                             args.l_hist, u_list, args.knn, args.max_points,
                             args.n_surrogates, args.seed + 1000 * si + ti,
                             args.min_shift, 1))
                pair_index.append((si, ti))

        def _report(r):
            r["drive"] = drive

            if r.get("error"):
                print(f"[te] {drive} {pops[r['source']]}->{pops[r['target']]}"
                      f" FAILED: {r['error'][:80]}", flush=True)
            else:
                print(f"[te] {drive} {pops[r['source']]}->{pops[r['target']]}"
                      f" u={r['u']} TEeff={r['te_eff']:.4f} p={r['p']:.3f}",
                      flush=True)

        results = _run_jobs(jobs, args.n_jobs, _report, pair_index,
                            backend=args.backend)

        errs = [r for r in results if r.get("error")]
        if errs:
            print(f"[te] {drive}: {len(errs)} of {len(results)} pairs failed")
            seen = set()
            for r in errs:
                msg = r["error"]
                key = msg.split(".")[0]
                if key not in seen:
                    seen.add(key)
                    print(f"[te]   {pops[r['source']]}->{pops[r['target']]}"
                          f" k_hist={khist[(drive, pops[r['target']])]}: {msg}")
        np.save(ckpt, np.array(results, dtype=object))
        print(f"[te] {drive}: checkpoint written "
              f"({len(results) - len(errs)} usable)")

    merged = []
    for drive in drives:
        ckpt = os.path.join(args.out_dir, f"te_{args.tag}_{drive}.npy")
        if os.path.exists(ckpt):
            merged.extend(list(np.load(ckpt, allow_pickle=True)))
    np.save(os.path.join(args.out_dir, f"te_{args.tag}.npy"),
            np.array(merged, dtype=object))
    write_csv(args.out_dir, args.tag, merged, pops, bin_ms)
    print(f"[te] merged {len(merged)} rows")


def write_csv(out_dir, tag, rows, pops, bin_ms):
    path = os.path.join(out_dir, f"te_{tag}.csv")
    with open(path, "w") as fh:
        fh.write("drive,source,target,u_ms,te_raw,surr_mean,surr_std,te_eff,p\n")
        for r in rows:
            fh.write(f"{r['drive']},{pops[r['source']]},{pops[r['target']]},"
                     f"{r['u'] * bin_ms:.3f},{r['te_raw']:.6f},{r['surr_mean']:.6f},"
                     f"{r['surr_std']:.6f},{r['te_eff']:.6f},{r['p']:.4f}\n")
    print(f"[te] wrote {path}")


# ----------------------------------------------------------------------------
# Stage: plot
# ----------------------------------------------------------------------------

def stage_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    meta = load_meta(args.out_dir)
    drives, pops = meta["drives"], meta["pops"]
    n_pops = len(pops)
    rows = list(np.load(os.path.join(args.out_dir, f"te_{args.tag}.npy"),
                        allow_pickle=True))
    _, ais = load_embed(args.out_dir)

    def drive_value(d):
        m = re.search(r"[\d.]+", str(d))
        return float(m.group()) if m else np.nan

    # Only drives that actually carry results. A partial sweep (--drive-stride,
    # --drives, or an interrupted run) otherwise contributes drives whose TE
    # matrix is entirely NaN, and nansum turns those into a spurious zero.
    have = {r["drive"] for r in rows if np.isfinite(r.get("te_eff", np.nan))}
    if args.min_drive is not None:
        have = {d for d in have
                if drive_value(d) >= args.min_drive}
    missing = [d for d in drives if d not in have]
    drives = [d for d in drives if d in have]
    if not drives:
        raise SystemExit(f"no usable results in te_{args.tag}.npy")
    if missing:
        print(f"[plot] {len(drives)} drives with results, "
              f"{len(missing)} without (not plotted)")

    xs = np.array([drive_value(d) for d in drives])
    order = np.argsort(xs)
    drives = [drives[i] for i in order]
    xs = xs[order]

    mats, mats_all = {}, {}
    for d in drives:
        M = np.full((n_pops, n_pops), np.nan)
        A = np.full((n_pops, n_pops), np.nan)
        for r in rows:
            if r["drive"] != d or not np.isfinite(r["te_eff"]):
                continue
            A[r["source"], r["target"]] = r["te_eff"]
            if not np.isfinite(r["p"]) or r["p"] <= args.alpha:
                M[r["source"], r["target"]] = r["te_eff"]
        mats[d] = M
        mats_all[d] = A

    # Panel A: total significant transfer and AIS versus drive
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    tot = [np.nansum(mats[d]) if np.isfinite(mats[d]).any() else np.nan
           for d in drives]
    ax[0].plot(xs, tot, "-o", color="k")
    ax[0].set_xlabel("background rate (spikes/s)")
    ax[0].set_ylabel("summed significant TE (nats)")
    ax[0].text(-0.15, 1.02, "A", transform=ax[0].transAxes,
               fontsize=14, fontweight="bold")
    for p, pop in enumerate(pops):
        ax[1].plot(xs, [ais[(d, pop)] for d in drives], label=pop,
                   markersize=4, **pop_style(pop))
    ax[1].set_xlabel("background rate (spikes/s)")
    ax[1].set_ylabel("active information storage (nats)")
    ax[1].legend(ncol=2, fontsize=8)
    ax[1].text(-0.15, 1.02, "B", transform=ax[1].transAxes,
               fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_te_vs_drive.png"), dpi=200)

    # Panel B: matrices at a few drives
    show = [drives[i] for i in
            np.unique(np.linspace(0, len(drives) - 1, 4).astype(int))]
    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 4))
    vmax = np.nanmax([np.nanmax(mats[d]) for d in show])
    for a, d in zip(np.atleast_1d(axes), show):
        im = a.imshow(mats[d], vmin=0, vmax=vmax, cmap="magma")
        a.set_xticks(range(n_pops)); a.set_xticklabels(pops, rotation=90, fontsize=8)
        a.set_yticks(range(n_pops)); a.set_yticklabels(pops, fontsize=8)
        a.set_title(f"drive {d}")
        a.set_xlabel("target"); a.set_ylabel("source")
    fig.colorbar(im, ax=axes, label="effective TE (nats)")
    fig.savefig(os.path.join(args.out_dir, "fig_te_matrices.png"), dpi=200)

    # Delay profiles: the diagnostic that separates a genuine short-latency
    # transfer from a peak sitting at the period of the shared rhythm. A real
    # directed interaction gives a single peak at a few ms; a shared oscillation
    # gives a profile that echoes at multiples of the cycle and is close to
    # symmetric between the two directions.
    bin_ms = meta.get("bin_ms", 1.0)
    prof_rows = [r for r in rows if "profile" in r
                 and np.isfinite(r["te_eff"])]
    if prof_rows:
        with_prof = [d for d in drives
                     if any(r["drive"] == d for r in prof_rows)]
        show = [with_prof[i] for i in
                np.unique(np.linspace(0, len(with_prof) - 1, 3).astype(int))]
        fig, axes = plt.subplots(1, len(show), figsize=(4.5 * len(show), 3.8),
                                 sharey=True)
        for a, d in zip(np.atleast_1d(axes), show):
            sel = [r for r in prof_rows if r["drive"] == d]
            sel.sort(key=lambda r: -r["te_eff"])
            for r in sel[:6]:
                lbl = f"{pops[r['source']]}->{pops[r['target']]}"
                a.plot(r["u_list"] * bin_ms, r["profile"] - r["profile_surr"],
                       lw=1.4, label=lbl)
            if args.period_ms:
                a.axvline(args.period_ms, color="0.6", ls=":", lw=1.0)
                a.axvline(args.period_ms / 2, color="0.85", ls=":", lw=1.0)
            a.axhline(0, color="0.8", lw=0.8)
            a.set_xlabel("source-target delay u (ms)")
            a.set_title(f"drive {d}")
            a.legend(fontsize=7)
        np.atleast_1d(axes)[0].set_ylabel("TE minus surrogate mean (nats)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_te_delay_profile.png"), dpi=200)

    # Panel C: effective versus structural connectivity
    if args.conn_probs:
        conn = np.load(args.conn_probs)
        if conn.shape != (n_pops, n_pops):
            raise SystemExit(f"conn_probs shape {conn.shape} != {(n_pops, n_pops)}")
        # NEST conn_probs is indexed [target, source]; TE matrix is [source, target]
        struct = np.log10(np.where(conn.T > 0, conn.T, np.nan))
        # Correlate over ALL pairs, not only those passing alpha. Restricting to
        # significant pairs changes the number and identity of points entering
        # the correlation from drive to drive, so the resulting curve varies
        # with how many pairs cleared threshold rather than with how well the
        # effective connectivity tracks anatomy.
        rho, pv, npts = [], [], []
        for d in drives:
            M = mats[d] if args.structure_significant_only else mats_all[d]
            m = np.isfinite(M) & np.isfinite(struct)
            np.fill_diagonal(m, False)
            r, p = spearmanr(M[m], struct[m])
            rho.append(r); pv.append(p); npts.append(int(m.sum()))
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(xs, rho, "-o", color="k")
        for x, r, p in zip(xs, rho, pv):
            if p <= 0.05:
                ax.plot(x, r, "*", color="firebrick", markersize=10)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("background rate (spikes/s)")
        ax.set_ylabel(r"Spearman $\rho$(TE, log connection probability)")
        ax.set_title("all pairs" if not args.structure_significant_only
                     else f"pairs with p <= {args.alpha}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_te_vs_structure.png"), dpi=200)
        print("[plot] structure correlation "
              + ("(significant pairs only)" if args.structure_significant_only
                 else "(all pairs)") + ":")
        for d, r, p, n in zip(drives, rho, pv, npts):
            print(f"[plot]   {d:>8}  rho = {r:+.3f}  p = {p:.3f}  n = {n}")
        print(f"[plot]   mean rho = {np.mean(rho):+.3f}, "
              f"sd = {np.std(rho, ddof=1):.3f}")
    print(f"[plot] figures written to {args.out_dir}")


# ----------------------------------------------------------------------------
# Stage: selftest
# ----------------------------------------------------------------------------

def stage_selftest(args):
    """
    Linear Gaussian benchmark. For
        X[t+1] = a X[t] + e,  Y[t+1] = b Y[t] + c X[t] + n
    the transfer entropy X -> Y equals half the log ratio of the residual
    variances of Y[t+1] regressed on Y[t] alone versus on (Y[t], X[t]), and the
    reverse direction is exactly zero. The KSG estimate should match the first
    and be indistinguishable from zero for the second.
    """
    rng = np.random.default_rng(0)
    n = 60000
    a, b, c = 0.5, 0.5, 0.6
    x = np.zeros(n); y = np.zeros(n)
    ex = rng.standard_normal(n); ey = rng.standard_normal(n)
    for t in range(n - 1):
        x[t + 1] = a * x[t] + ex[t]
        y[t + 1] = b * y[t] + c * x[t] + ey[t]
    x, y = x[1000:], y[1000:]

    def ols_resid_var(target, regressors):
        A = np.column_stack([np.ones(len(target))] + regressors)
        beta, *_ = np.linalg.lstsq(A, target, rcond=None)
        return np.var(target - A @ beta)

    yf, yp, xp = y[1:], y[:-1], x[:-1]
    te_true = 0.5 * np.log(ols_resid_var(yf, [yp]) /
                           ols_resid_var(yf, [yp, xp]))
    xf, xpp, ypp = x[1:], x[:-1], y[:-1]
    te_rev_true = 0.5 * np.log(ols_resid_var(xf, [xpp]) /
                               ols_resid_var(xf, [xpp, ypp]))

    r = np.random.default_rng(1)
    m = min(args.max_points, len(yf))
    idx = r.choice(len(yf), m, replace=False)
    te_ksg = cmi_ksg(_prepare(yf[idx], r), _prepare(xp[idx], r),
                     _prepare(yp[idx], r), k=args.knn, workers=args.n_jobs)
    te_rev = cmi_ksg(_prepare(xf[idx], r), _prepare(ypp[idx], r),
                     _prepare(xpp[idx], r), k=args.knn, workers=args.n_jobs)

    print(f"  analytic  TE(X->Y) = {te_true:.4f} nats")
    print(f"  KSG       TE(X->Y) = {te_ksg:.4f} nats  "
          f"(relative error {abs(te_ksg - te_true) / te_true:.1%})")
    print(f"  analytic  TE(Y->X) = {te_rev_true:.4f} nats")
    print(f"  KSG       TE(Y->X) = {te_rev:.4f} nats")
    ok = abs(te_ksg - te_true) / te_true < 0.10 and abs(te_rev) < 0.02
    print("  PASS" if ok else "  FAIL: check scipy version and knn setting")
    return 0 if ok else 1


# ----------------------------------------------------------------------------


def stage_check(args):
    """
    Inspect the built checkpoints before spending compute on them. Reports, per
    drive and population: shape, non-finite count, range, and the number of
    distinct values. The last one matters because a KSG estimate on a variable
    that takes only a handful of values is dominated by ties rather than by the
    dependence you are trying to measure.
    """
    meta = load_meta(args.out_dir)
    drives, pops = meta["drives"], meta["pops"]
    print(f"{'drive':>8} {'pop':>6} {'shape':>18} {'nonfinite':>10} "
          f"{'min':>8} {'max':>8} {'n_unique':>9} {'mean/bin':>9}")
    bad_drives, thin_pops = [], set()
    for d in drives:
        try:
            c = load_counts(args.out_dir, d, validate=False)
        except Exception as e:
            print(f"{d:>8} FAILED TO LOAD: {e}")
            bad_drives.append(d)
            continue
        for p, pop in enumerate(pops):
            x = c[:, p, :]
            nbad = int((~np.isfinite(x)).sum())
            nun = int(np.unique(x[np.isfinite(x)]).size) if np.isfinite(x).any() else 0
            mn = float(np.nanmin(x)) if np.isfinite(x).any() else float("nan")
            mx = float(np.nanmax(x)) if np.isfinite(x).any() else float("nan")
            flag = ""
            if nbad:
                flag = "  <-- NON-FINITE"
                bad_drives.append(d)
            elif x.mean() < args.min_counts or nun < 20:
                flag = f"  <-- too sparse for KSG ({x.mean():.2f} counts/bin)"
                thin_pops.add(pop)
            if args.verbose or flag:
                print(f"{d:>8} {pop:>6} {str(x.shape):>18} {nbad:>10} "
                      f"{mn:>8.2f} {mx:>8.2f} {nun:>9} {x.mean():>9.2f}{flag}")
    print()
    if bad_drives:
        print("Drives with non-finite or unreadable data: "
              + ", ".join(sorted(set(bad_drives))))
        print("Rebuild those with --force after checking the source .dat files, "
              "or exclude them from the te stage with --drives.")
    if thin_pops:
        print("Sparse populations: " + ", ".join(sorted(thin_pops)))
        print("Coarser bins do not fix this. Widening the bin smears the "
              "source-target lag as fast as it adds counts, so TE into a "
              "sparse target stays at the noise floor at every bin width. "
              "A population firing this rarely has almost no entropy to "
              "receive, which is a fact about the circuit rather than a "
              "shortcoming of the estimate. Exclude these rows and say so.")
    if not bad_drives and not thin_pops:
        print("All drives finite and adequately resolved.")
    return 1 if bad_drives else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    def common(sp):
        sp.add_argument("--out-dir", default="./te_out")
        sp.add_argument("--seed", type=int, default=1234)
        sp.add_argument("--knn", type=int, default=4,
                        help="KSG neighbour count")
        sp.add_argument("--max-points", type=int, default=20000)
        sp.add_argument("--n-jobs", type=int, default=1)
        sp.add_argument("--thin", action="store_true",
                        help="binomial rate matching across drives")
        sp.add_argument("--smooth-sigma", type=float, default=0.0,
                        help="Gaussian smoothing in bins. Leave at 0 for TE. "
                             "Any nonzero value contaminates delays below "
                             "roughly 3 sigma with the kernel itself.")
        sp.add_argument("--force", action="store_true")

    sb = sub.add_parser("build"); common(sb)
    sb.add_argument("--data-root", required=True)
    sb.add_argument("--source", choices=["popact", "spikes"], default="popact",
                    help="popact reads measurements/pop_activities/*.dat, the "
                         "same traces the spectral entropy notebook uses. "
                         "spikes re-bins the raw NEST spike recorders.")
    sb.add_argument("--t-start", type=float, default=1000.0)
    sb.add_argument("--t-stop", type=float, default=5500.0)
    sb.add_argument("--t-offset", type=float, default=500.0,
                    help="simulation time in ms of the first sample in the "
                         "pop_activity files")
    sb.add_argument("--dt-ms", type=float, default=0.2,
                    help="native sampling interval of pop_activity")
    sb.add_argument("--decimate", type=int, default=5,
                    help="sum this many native bins together. 5 takes the "
                         "0.2 ms traces to 1 ms, which is the right resolution "
                         "for TE: at 0.2 ms a population rhythm of a few tens "
                         "of Hz is sampled well over a hundred times per cycle "
                         "and consecutive samples are near-deterministic, which "
                         "is exactly the regime where KSG returns large "
                         "symmetric values on coupled periodic signals.")
    sb.add_argument("--pop-indices", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 5, 6, 7],
                    help="pop_activity file indices to load. Use 0 2 4 6 for "
                         "the four excitatory populations only.")
    sb.add_argument("--n-full", default=None,
                    help="npy of net_dict['full_num_neurons'], enabling exact "
                         "count recovery, the smoothing check, and --thin")
    sb.add_argument("--bin-ms", type=float, default=1.0,
                    help="only used by --source spikes")
    sb.add_argument("--n-trials", type=int, default=None,
                    help="use only the first N trials per drive")
    sb.add_argument("--pop-order", nargs="+", default=POPS)

    se = sub.add_parser("embed"); common(se)
    se.add_argument("--k-max", type=int, default=6)

    st = sub.add_parser("te"); common(st)
    st.add_argument("--u-max-ms", type=float, default=16.0,
                    help="scan delays out to here. Measure the population "
                         "rhythm first and keep this above one full period, so "
                         "that a peak at the cycle period is visible rather "
                         "than hidden beyond the end of the scan.")
    st.add_argument("--u-step-ms", type=float, default=1.0)
    st.add_argument("--u-grid-ms", type=float, nargs="+", default=None,
                    help="explicit delay values in ms, overriding u-max/u-step. "
                         "Use a fine grid at short latency and a coarse one out "
                         "to the cycle period.")
    st.add_argument("--period-ms", type=float, default=None,
                    help="measured period of the population rhythm, 1000/f0. "
                         "Used to warn about a truncated scan and to mark the "
                         "cycle period on the delay-profile figure.")
    st.add_argument("--l-hist", type=int, default=1)
    st.add_argument("--n-surrogates", type=int, default=19)
    st.add_argument("--min-shift", type=int, default=50,
                    help="minimum cyclic shift in bins")
    st.add_argument("--condition", choices=["none", "global", "topN"],
                    default="none")
    st.add_argument("--top-n", type=int, default=2)
    st.add_argument("--backend", choices=["thread", "serial", "process"],
                    default="thread",
                    help="thread runs pairs on threads in this process and is "
                         "the default. serial runs one pair at a time, for "
                         "debugging. process uses a worker pool, which can "
                         "deadlock and is kept only as a fallback.")
    st.add_argument("--min-drive", type=float, default=None,
                    help="skip drives below this. Below the drive at which all "
                         "trials remain active, pooling trials mixes states "
                         "with order-of-magnitude different rates.")
    st.add_argument("--drive-stride", type=int, default=1,
                    help="take every Nth drive, for a coarse first pass")
    st.add_argument("--drives", nargs="+", default=None,
                    help="explicit drive directory names to run")
    st.add_argument("--tag", default="pairwise")
    st.add_argument("--prev-tag", default="pairwise")

    sp_ = sub.add_parser("plot"); common(sp_)
    sp_.add_argument("--tag", default="pairwise")
    sp_.add_argument("--conn-probs", default=None)
    sp_.add_argument("--alpha", type=float, default=0.05)
    sp_.add_argument("--structure-significant-only", action="store_true",
                     help="restrict the structure correlation to pairs passing "
                          "alpha. Off by default: the pair set then changes "
                          "between drives and the curve tracks threshold "
                          "crossings rather than anatomy.")
    sp_.add_argument("--min-drive", type=float, default=None,
                     help="drop drives below this from every panel")
    sp_.add_argument("--period-ms", type=float, default=None,
                     help="marks one period and half a period on the "
                          "delay-profile figure")

    sc = sub.add_parser("check"); common(sc)
    sc.add_argument("--min-counts", type=float, default=3.0,
                    help="flag populations averaging fewer counts per bin than "
                         "this. Below about 3 the variable is dominated by its "
                         "mass at zero and KSG is measuring ties.")
    sc.add_argument("--verbose", action="store_true",
                    help="print every population, not only the problems")

    ss = sub.add_parser("selftest"); common(ss)

    args = p.parse_args(argv)
    return {"build": stage_build, "embed": stage_embed, "te": stage_te,
            "plot": stage_plot, "check": stage_check,
            "selftest": stage_selftest}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main() or 0)