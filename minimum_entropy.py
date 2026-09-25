#!/usr/bin/env python3
"""
entropy_minimum.py

Motivation
----------
The spectral entropy curve has a clear minimum in every excitatory population,
but the poster currently asserts its location by eye. Reading the minimum off
the grid is also unstable: with trial-to-trial scatter of the size visible in
the L4E error bars, the argmin can jump several grid points between resamples,
so a bare "the minimum is at 9.1" overstates what the data supports.

This estimates the location with an interval, so it can be compared directly
against the drives where the other measures place their transitions.

What it measures
----------------
Two estimators, deliberately both:

  argmin      the drive with the lowest mean entropy. Honest but quantised to
              the sweep grid and unstable under resampling.
  vertex      the vertex of a quadratic fitted to a window of points around the
              crude minimum. Sub-grid resolution, but it assumes the curve is
              locally parabolic, which is an assumption the data can violate.

Both are bootstrapped by resampling trials within each drive. If the two
disagree by more than the interval width, the curve is not locally parabolic
and only the argmin should be quoted.

Usage
-----
    # entropy_per_trial[p][i] is the list of per-trial entropies for
    # population p at drive i; drives is the array of drive values
    from entropy_minimum import estimate_minimum, annotate
    res = estimate_minimum(entropy_per_trial, drives, pops)
    annotate(ax, res, colors_pops)
"""

import numpy as np


def _vertex(x, y, i0, half_window=4):
    lo = max(0, i0 - half_window)
    hi = min(len(x), i0 + half_window + 1)
    if hi - lo < 3:
        return np.nan
    c = np.polyfit(x[lo:hi], y[lo:hi], 2)
    if c[0] <= 0:                      # not a minimum
        return np.nan
    return -c[1] / (2 * c[0])


def _interior_argmin(mean, lo_i, hi_i):
    """
    Deepest interior local minimum inside [lo_i, hi_i].

    The global minimum over a finite sweep window is not usable when it lands on
    a boundary: there is no data beyond the edge, so nothing establishes that the
    edge point is a minimum rather than the visible end of a further decline. In
    this sweep the entropy is also low at the lowest drive, giving a boundary
    argmin for some populations that describes the edge of the window rather
    than the dip. Requiring both neighbours to be higher restricts the estimate
    to features the data actually resolves.
    """
    cand = [i for i in range(max(lo_i, 1), min(hi_i, len(mean) - 2) + 1)
            if mean[i] <= mean[i - 1] and mean[i] <= mean[i + 1]]
    if cand:
        return int(min(cand, key=lambda i: mean[i]))
    idx = np.arange(lo_i, hi_i + 1)
    return int(idx[np.argmin(mean[idx])])


def estimate_minimum(entropy_per_trial, drives, pops,
                     n_boot=2000, half_window=4, seed=0,
                     search_range=None, min_drive=None):
    """
    entropy_per_trial: dict or list indexed [p][i] -> sequence of trial values.
    drives: array of drive values, same order as the i index.
    search_range: (lo, hi) in drive units. The minimum is sought only here.
        Defaults to the full range.
    min_drive: drives below this are dropped entirely, from the curve and from
        the bootstrap, not merely excluded from the search. Use this where the
        across-trial rate variability makes the spectral estimate unreliable
        rather than merely uninteresting: in that region a trial with almost no
        spikes has a spectrum set by a handful of events, so its entropy is not
        comparable to the rest of the sweep and should not contribute to the
        curve at all.
    Returns a dict keyed by population name.
    """
    rng = np.random.default_rng(seed)
    drives = np.asarray(drives, dtype=float)

    if min_drive is not None:
        keep = np.where(drives >= min_drive)[0]
        n_drop = len(drives) - keep.size
        if keep.size < 3:
            raise ValueError(f"min_drive={min_drive} leaves {keep.size} drives")
        if n_drop:
            print(f"Dropping {n_drop} drives below {min_drive} spikes/s "
                  f"({drives[0]:.2f} to {drives[keep[0] - 1]:.2f}).")
        drives = drives[keep]
        entropy_per_trial = {p: [entropy_per_trial[p][i] for i in keep]
                             for p in range(len(pops))}

    if search_range is None:
        lo_i, hi_i = 0, len(drives) - 1
    else:
        inside = np.where((drives >= search_range[0]) &
                          (drives <= search_range[1]))[0]
        if inside.size < 3:
            raise ValueError("search_range covers fewer than three drives")
        lo_i, hi_i = int(inside[0]), int(inside[-1])

    out = {}
    for p, name in enumerate(pops):
        mean = np.array([np.mean(entropy_per_trial[p][i])
                         for i in range(len(drives))])
        i0 = _interior_argmin(mean, lo_i, hi_i)
        if i0 in (0, len(drives) - 1):
            print(f"WARNING: {name} minimum is at a sweep boundary "
                  f"({drives[i0]:.2f}). No data lies beyond it, so this is the "
                  f"edge of the window rather than a resolved minimum. Pass "
                  f"search_range to exclude it.")
        est = dict(argmin=float(drives[i0]),
                   vertex=float(_vertex(drives, mean, i0, half_window)),
                   min_value=float(mean[i0]))

        b_arg, b_vtx = [], []
        for _ in range(n_boot):
            m = np.empty(len(drives))
            for i in range(len(drives)):
                v = np.asarray(entropy_per_trial[p][i], dtype=float)
                m[i] = v.mean() if v.size < 2 else \
                    rng.choice(v, size=v.size, replace=True).mean()
            j = _interior_argmin(m, lo_i, hi_i)
            b_arg.append(drives[j])
            b_vtx.append(_vertex(drives, m, j, half_window))
        b_arg = np.asarray(b_arg)
        b_vtx = np.asarray(b_vtx)
        b_vtx = b_vtx[np.isfinite(b_vtx)]

        est["argmin_ci"] = tuple(np.percentile(b_arg, [2.5, 97.5]))
        est["vertex_ci"] = (tuple(np.percentile(b_vtx, [2.5, 97.5]))
                            if b_vtx.size > n_boot // 10 else (np.nan, np.nan))
        est["n_trials"] = int(np.median([len(entropy_per_trial[p][i])
                                         for i in range(len(drives))]))
        out[name] = est
    return out


def report(res):
    print(f"{'pop':>6} {'argmin':>8} {'95% CI':>16} {'vertex':>8} "
          f"{'95% CI':>16} {'trials':>7}")
    for name, e in res.items():
        a = f"[{e['argmin_ci'][0]:.2f}, {e['argmin_ci'][1]:.2f}]"
        v = f"[{e['vertex_ci'][0]:.2f}, {e['vertex_ci'][1]:.2f}]" \
            if np.isfinite(e['vertex_ci'][0]) else "n/a"
        print(f"{name:>6} {e['argmin']:>8.2f} {a:>16} "
              f"{e['vertex']:>8.2f} {v:>16} {e['n_trials']:>7}")


def annotate(ax, res, colors, use="vertex", band=True):
    """Mark each population's minimum on an existing axis."""
    for (name, e), col in zip(res.items(), colors):
        x = e[use] if np.isfinite(e[use]) else e["argmin"]
        ci = e[f"{use}_ci"] if np.isfinite(e[f"{use}_ci"][0]) \
            else e["argmin_ci"]
        if band:
            ax.axvspan(ci[0], ci[1], color=col, alpha=0.10, lw=0)
        ax.axvline(x, color=col, linestyle=":", linewidth=1.4, alpha=0.9)