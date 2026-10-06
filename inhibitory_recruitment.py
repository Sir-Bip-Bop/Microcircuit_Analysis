#!/usr/bin/env python3
"""
inhibitory_recruitment.py

Motivation
----------
The manuscript's central mechanism is stated, not measured: as background
drive increases, the inhibition recruited by a large population event is
supposed to become deep and long-lasting enough that the network cannot mount
a full-amplitude event on the next cycle, so large events recur every other
cycle and deposit power at f0/2. Sections III-D and IV infer this from the
interval statistics and from the amplitude-to-next-interval correlation. A
reviewer will ask for the inhibition itself. The synaptic currents were
recorded, so the question can be answered directly: does the inhibitory
current onto an excitatory population, time-locked to that population's
events, scale with event amplitude, and does the inhibitory charge delivered
after an event predict how long the network waits before the next one?

What it measures
----------------
For every background rate, realization and excitatory population:
  1. Population events, detected as peaks of the smoothed population rate
     (the same 1 ms rate traces used in Fig. 5, if present) or, as a
     fallback, of the population-summed excitatory current.
  2. The event-triggered average (ETA) of the population-summed inhibitory
     current |I_in|(t) in a window around each event, separately for events
     in the lowest and highest amplitude terciles.
  3. Per event k: amplitude A_k, inhibitory charge Q_k = integral of
     (|I_in| - baseline) over [t_k, t_k + T_int] with the baseline taken from
     [t_k - 10, t_k - 5] ms, the post-event peak inhibitory current, and the
     interval D_k to the next event.
  4. Spearman correlations rho(A, Q), rho(Q, D) and rho(A, D), per
     realization and pooled, plus the ratio of inhibitory charge recruited by
     high- versus low-tercile events.
Currents are per recorded neuron (the multimeter subset), so absolute values
are comparable across rates within a population but are not population
totals unless rescaled by N_pop / N_recorded.

Outputs (in --out):
  inhib_events.csv         one row per event: rate, trial, pop, t, A, Q, Ipeak, D
  inhib_summary.csv        per rate/trial/pop: rho(A,Q), rho(Q,D), rho(A,D), n,
                           Q_high/Q_low, ETA peak for low/high tercile
  inhib_eta.npz            ETAs per rate/trial/pop/tercile on a common lag axis
  fig_inhib_<pop>.png/pdf  2 x n_rates panel: ETA (low vs high tercile) and
                           D vs Q scatter, for the population given by --fig-pop

Usage
-----
  # four representative drives, all excitatory populations, figure for L2/3E
  python inhibitory_recruitment.py --root ~/data/data_background_rate_big_3 \
      --rates 6 9 12 18 --out inhib_out

  # full sweep, one population, events from the excitatory current instead of
  # the rate file
  python inhibitory_recruitment.py --root ... --populations L4E --events current

  # synthetic self-test of the code paths (no data needed)
  python inhibitory_recruitment.py --selftest

Requires lfp_proxy_spectra.py in the same directory (loader is imported).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfp_proxy_spectra import POPS, E_POPS, load_trial, parse_rate  # noqa: E402


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------
def load_rate_trace(trial_dir, pop, subdir="measurements/pop_activities_1ms",
                    t0_ms=500.0, dt_ms=1.0):
    """Population rate trace written by the simulation, or None if absent."""
    f = trial_dir / subdir / f"pop_activity_{POPS.index(pop)}.dat"
    if not f.exists():
        return None, None
    x = np.loadtxt(f)
    t = t0_ms + dt_ms * np.arange(len(x))
    return t, x


def detect_events(t, x, sigma_ms=2.0, min_dist_ms=6.0, height="median"):
    """Peaks of the Gaussian-smoothed signal above a threshold.

    Returns event times (ms) and amplitudes (units of x).
    min_dist_ms = 6 ms is half a gamma period: it forbids double detections
    within one cycle without forbidding events on consecutive cycles.
    """
    dt = float(np.median(np.diff(t)))
    xs = gaussian_filter1d(x, sigma_ms / dt, mode="nearest")
    thr = np.median(xs) if height == "median" else float(height)
    idx, _ = find_peaks(xs, height=thr, distance=max(1, int(round(min_dist_ms / dt))))
    return t[idx], xs[idx]


# ---------------------------------------------------------------------------
# Event-triggered inhibitory current
# ---------------------------------------------------------------------------
def event_triggered(t_cur, I_abs, t_events, pre_ms, post_ms):
    """Stack |I_in| around each event on a common lag axis (ms).

    Returns lags (ms) and an array (n_events, n_lags); events too close to the
    trace edges are dropped and their indices returned in `kept`.
    """
    dt = float(np.median(np.diff(t_cur)))
    n_pre, n_post = int(round(pre_ms / dt)), int(round(post_ms / dt))
    lags = dt * np.arange(-n_pre, n_post + 1)
    idx0 = np.searchsorted(t_cur, t_events)
    kept, rows = [], []
    for k, i in enumerate(idx0):
        if i - n_pre < 0 or i + n_post >= len(I_abs):
            continue
        rows.append(I_abs[i - n_pre: i + n_post + 1])
        kept.append(k)
    return lags, np.asarray(rows), np.asarray(kept, dtype=int)


def per_event_metrics(lags, eta_rows, t_events_kept, T_int_ms, base_win=(-10.0, -5.0)):
    """Inhibitory charge of the post-event pulse, its peak, and the interval to the next event.

    The inhibitory current after an event is a pulse of a few ms that then
    dips below baseline (the interneurons are silent after their volley), so
    Q integrates only from the event to the first return to baseline after
    the peak, capped at T_int_ms; integrating a fixed long window would add
    the dip and, at short intervals, the next event's pulse.
    """
    dt = float(np.median(np.diff(lags)))
    mb = (lags >= base_win[0]) & (lags < base_win[1])
    i0 = int(np.argmin(np.abs(lags)))
    imax = i0 + int(round(T_int_ms / dt))
    base = eta_rows[:, mb].mean(axis=1)
    Q = np.empty(len(eta_rows)); Ipk = np.empty(len(eta_rows)); t_ret = np.empty(len(eta_rows))
    for k, row in enumerate(eta_rows):
        y = row[i0:imax + 1] - base[k]
        ip = int(np.argmax(y))
        below = np.flatnonzero(y[ip:] <= 0)
        iend = ip + (below[0] if below.size else len(y) - ip)
        Q[k] = y[:iend].clip(min=0).sum() * dt          # pA*ms per neuron, pulse only
        Ipk[k] = y[ip]
        t_ret[k] = iend * dt
    D = np.full(len(t_events_kept), np.nan)
    D[:-1] = np.diff(t_events_kept)
    return Q, Ipk, D, base, t_ret


def rank_partial(a, b, c):
    """Spearman-type partial correlation of a and b controlling for c (rank residuals)."""
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    if m.sum() < 10:
        return np.nan
    from scipy.stats import rankdata
    ra, rb, rc = rankdata(a[m]), rankdata(b[m]), rankdata(c[m])
    X = np.column_stack([np.ones(m.sum()), rc])
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


def safe_spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(a[m], b[m])
    return float(r), float(p), int(m.sum())


# ---------------------------------------------------------------------------
# Per-trial analysis
# ---------------------------------------------------------------------------
def analyse_trial(trial_dir, rate, trial, pops, args, override=None):
    data = load_trial(trial_dir, override, set(pops))
    ev_rows, sum_rows, etas = [], [], {}
    for pop in pops:
        if pop not in data or "I_in" not in data[pop]:
            print(f"    {pop}: no inhibitory current, skipped")
            continue
        d = data[pop]
        n_rec = d.get("n_rec", 1)
        t_cur, I_in = d["time"], np.abs(d["I_in"]) / n_rec       # per recorded neuron
        keep = t_cur >= (t_cur[0] + args.transient_ms)
        t_cur, I_in = t_cur[keep], I_in[keep]

        # event signal
        t_ev_sig, x_ev_sig = (None, None)
        if args.events == "rate":
            t_ev_sig, x_ev_sig = load_rate_trace(trial_dir, pop, args.rate_subdir,
                                                 args.rate_t0_ms, args.rate_dt_ms)
        if t_ev_sig is None:
            t_ev_sig, x_ev_sig = t_cur, np.abs(d["I_ex"][keep]) / n_rec
            src = "I_ex"
        else:
            m = (t_ev_sig >= t_cur[0]) & (t_ev_sig <= t_cur[-1])
            t_ev_sig, x_ev_sig = t_ev_sig[m], x_ev_sig[m]
            src = "rate"
        t_ev, A = detect_events(t_ev_sig, x_ev_sig, args.sigma_ms, args.min_dist_ms)
        if len(t_ev) < 10:
            print(f"    {pop}: only {len(t_ev)} events, skipped")
            continue

        lags, rows, kept = event_triggered(t_cur, I_in, t_ev, args.pre_ms, args.post_ms)
        t_ev, A = t_ev[kept], A[kept]
        Q, Ipk, D, base, t_ret = per_event_metrics(lags, rows, t_ev, args.t_int_ms)

        # amplitude terciles
        lo, hi = np.quantile(A, [1 / 3, 2 / 3])
        low, high = A <= lo, A >= hi
        eta_low, eta_high = rows[low].mean(axis=0), rows[high].mean(axis=0)
        etas[(rate, trial, pop, "low")] = eta_low
        etas[(rate, trial, pop, "high")] = eta_high
        etas["lags"] = lags

        rAQ, pAQ, n = safe_spearman(A, Q)
        rQD, pQD, _ = safe_spearman(Q, D)
        rAD, pAD, _ = safe_spearman(A, D)
        rID, pID, _ = safe_spearman(Ipk, D)
        rAD_Q = rank_partial(A, D, Q)
        rQD_A = rank_partial(Q, D, A)
        Qratio = np.nanmean(Q[high]) / np.nanmean(Q[low]) if np.nanmean(Q[low]) > 0 else np.nan
        sum_rows.append(dict(rate=rate, trial=trial, population=pop, events_from=src,
                             n_events=len(A), rho_A_Q=rAQ, p_A_Q=pAQ, rho_Q_D=rQD, p_Q_D=pQD,
                             rho_A_D=rAD, p_A_D=pAD, rho_Ipk_D=rID, p_Ipk_D=pID,
                             rho_A_D_given_Q=rAD_Q, rho_Q_D_given_A=rQD_A,
                             pulse_ms=float(np.nanmedian(t_ret)), Q_high_over_low=Qratio,
                             Ipeak_low=float(np.nanmean(Ipk[low])), Ipeak_high=float(np.nanmean(Ipk[high])),
                             D_low_ms=float(np.nanmean(D[low])), D_high_ms=float(np.nanmean(D[high]))))
        for k in range(len(A)):
            ev_rows.append(dict(rate=rate, trial=trial, population=pop, t_ms=t_ev[k], A=A[k],
                                Q=Q[k], Ipeak=Ipk[k], pulse_ms=t_ret[k], D_ms=D[k], baseline=base[k]))
        print(f"    {pop}: {len(A)} events ({src}); pulse {np.nanmedian(t_ret):.1f} ms; "
              f"rho(A,Q)={rAQ:+.2f} rho(Q,D)={rQD:+.2f} rho(A,D)={rAD:+.2f} "
              f"rho(A,D|Q)={rAD_Q:+.2f} rho(Q,D|A)={rQD_A:+.2f}; Q_high/Q_low={Qratio:.2f}")
    return ev_rows, sum_rows, etas


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(events, etas, rates, pop, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    lags = etas["lags"]
    n = len(rates)
    fig, axes = plt.subplots(2, n, figsize=(min(7.16, 1.8 * n + 0.5), 3.8), dpi=300, squeeze=False)
    for j, r in enumerate(rates):
        ax = axes[0, j]
        for terc, col in (("low", "#7fb3d5"), ("high", "#c0392b")):
            rows = [v for k, v in etas.items() if k != "lags" and k[0] == r and k[2] == pop and k[3] == terc]
            if not rows:
                continue
            m, s = np.mean(rows, axis=0), np.std(rows, axis=0) / np.sqrt(len(rows))
            ax.plot(lags, m, color=col, lw=1.0, label=f"{terc} tercile")
            ax.fill_between(lags, m - s, m + s, color=col, alpha=0.25, lw=0)
        ax.axvline(0, color="0.4", lw=0.6, ls=":")
        ax.set_title(f"{r:g} spikes/s", fontsize=8)
        if j == 0:
            ax.set_ylabel(r"$|I_{\rm in}|$ [pA/neuron]")
            ax.legend(frameon=False, loc="upper right")
        ax.set_xlabel("time from event [ms]")

        ax = axes[1, j]
        e = events[(events.rate == r) & (events.population == pop)]
        ax.scatter(e.Q, e.D_ms, s=3, alpha=0.35, color="0.3", lw=0)
        rho, p, nn = safe_spearman(e.Q.to_numpy(), e.D_ms.to_numpy())
        ax.text(0.03, 0.95, f"$\\rho$ = {rho:+.2f}\nn = {nn}", transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xlabel("inhibitory charge $Q$ [pA$\\cdot$ms/neuron]")
        if j == 0:
            ax.set_ylabel("interval to next event [ms]")
        ax.set_ylim(0, np.nanpercentile(e.D_ms, 98) * 1.05 if len(e) else 50)
    for ax in axes.ravel():
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(pop, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"fig_inhib_{pop}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"fig_inhib_{pop}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Self-test on synthetic data
# ---------------------------------------------------------------------------
def selftest(tmp):
    """Write a synthetic trial in the expected layout and run the pipeline.

    The synthetic network: an 80 Hz event train whose amplitudes alternate
    randomly, an inhibitory current that is a decaying exponential scaled by
    event amplitude, and an inter-event interval that grows with the
    inhibitory charge. The test passes if rho(A,Q) and rho(Q,D) come out
    clearly positive.
    """
    rng = np.random.default_rng(0)
    root = Path(tmp) / "9.0" / "trial_0"
    root.mkdir(parents=True, exist_ok=True)
    dt, T = 0.1, 3000.0
    t = np.arange(0, T, dt)
    I_in = np.zeros_like(t); I_ex = np.zeros_like(t)
    tk, amps, events_t = 600.0, [], []
    while tk < T - 50:
        A = rng.uniform(0.5, 2.0)
        i = int(tk / dt)
        tau = 4.0
        kern = A * 100 * np.exp(-(t[i:] - tk) / tau)
        I_in[i:] -= kern
        I_ex[i:i + 20] += A * 50
        events_t.append(tk); amps.append(A)
        tk += 12.5 + 8.0 * (A - 0.5) + rng.normal(0, 0.8)     # larger event -> longer wait
    I_in += rng.normal(0, 3, len(t)); I_ex += 40 + rng.normal(0, 3, len(t))
    for prefix, I in (("ex", I_ex), ("in", I_in)):
        for gid in range(77170, 77186, 2):        # 8 recorders, one per population
            with open(root / f"{prefix}_current-{gid}-00.dat", "w") as fh:
                fh.write("sender\ttime_ms\tI_syn\n")
                for ti, Ii in zip(t[::5], I[::5]):
                    fh.write(f"1\t{ti:.1f}\t{Ii:.3f}\n")
    args = argparse.Namespace(transient_ms=500.0, events="current", rate_subdir="", rate_t0_ms=500.0,
                              rate_dt_ms=1.0, sigma_ms=2.0, min_dist_ms=6.0, pre_ms=10.0, post_ms=40.0,
                              t_int_ms=8.0)
    ev, sm, etas = analyse_trial(root, 9.0, 0, ["L23E"], args)
    s = sm[0]
    ok = s["rho_A_Q"] > 0.5 and s["rho_Q_D"] > 0.5
    print(f"\nselftest: rho(A,Q)={s['rho_A_Q']:.2f}, rho(Q,D)={s['rho_Q_D']:.2f}, "
          f"Q_high/Q_low={s['Q_high_over_low']:.2f} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/mnt/d/data_background_rate_big_3")
    ap.add_argument("--out", default="inhib_out")
    ap.add_argument("--rates", nargs="*", type=float, default=None)
    ap.add_argument("--populations", default="E", help="'E' (default), 'all', or list e.g. L23E,L4E")
    ap.add_argument("--events", choices=["rate", "current"], default="rate",
                    help="detect events on the simulation's 1 ms rate trace (default) or on |I_ex|")
    ap.add_argument("--rate-subdir", default="measurements/pop_activities_1ms")
    ap.add_argument("--rate-t0-ms", type=float, default=500.0, help="time of the first rate sample")
    ap.add_argument("--rate-dt-ms", type=float, default=1.0)
    ap.add_argument("--transient-ms", type=float, default=500.0)
    ap.add_argument("--sigma-ms", type=float, default=2.0)
    ap.add_argument("--min-dist-ms", type=float, default=6.0)
    ap.add_argument("--pre-ms", type=float, default=10.0)
    ap.add_argument("--post-ms", type=float, default=40.0)
    ap.add_argument("--t-int-ms", type=float, default=8.0,
                    help="cap on the pulse integration window for Q (pulse ends at return to baseline)")
    ap.add_argument("--fig-pop", default="L23E")
    ap.add_argument("--fig-rates", nargs="*", type=float, default=[6, 9, 12, 18])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            sys.exit(selftest(tmp))

    root, out = Path(args.root).expanduser(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pops = E_POPS if args.populations == "E" else (POPS if args.populations == "all"
                                                  else [p.strip() for p in args.populations.split(",")])
    rate_dirs = {parse_rate(d.name): d for d in sorted(root.iterdir())
                 if d.is_dir() and parse_rate(d.name) is not None}
    avail = np.array(sorted(rate_dirs))
    if args.rates is None:
        rates = list(avail)
    else:
        rates = []
        for r in args.rates:
            j = int(np.argmin(np.abs(avail - r)))
            if abs(avail[j] - r) > 0.3:
                sys.exit(f"no directory within 0.3 spikes/s of {r}; available: {avail.tolist()}")
            rates.append(float(avail[j]))
    print(f"Rates: {rates}")

    events, summary, etas = [], [], {}
    for r in rates:
        for td in sorted(d for d in rate_dirs[r].iterdir() if d.is_dir() and d.name.startswith("trial")):
            trial = int(re.search(r"\d+", td.name).group())
            print(f"rate {r:g}, {td.name}")
            ev, sm, et = analyse_trial(td, r, trial, pops, args)
            events += ev; summary += sm; etas.update(et)

    if not summary:
        sys.exit("Nothing computed.")
    events = pd.DataFrame(events); summary = pd.DataFrame(summary)
    events.to_csv(out / "inhib_events.csv", index=False)
    summary.to_csv(out / "inhib_summary.csv", index=False)
    np.savez_compressed(out / "inhib_eta.npz",
                        **{("lags" if k == "lags" else f"{k[0]:g}|{k[1]}|{k[2]}|{k[3]}"): v for k, v in etas.items()})

    fig_rates = [float(rates[int(np.argmin(np.abs(np.array(rates) - r)))]) for r in args.fig_rates]
    fig_rates = sorted(set(r for r in fig_rates if abs(r - min(args.fig_rates, key=lambda x: abs(x - r))) <= 0.3))
    if fig_rates and args.fig_pop in pops:
        make_figure(events, etas, fig_rates, args.fig_pop, out)

    print("\nPer rate and population (mean over realizations):")
    print(summary.groupby(["population", "rate"])[["rho_A_Q", "rho_Q_D", "rho_A_D", "rho_A_D_given_Q",
                                                   "rho_Q_D_given_A", "pulse_ms", "Q_high_over_low"]]
          .mean().round(3).to_string())
    print(f"\nWritten to {out.resolve()}")


if __name__ == "__main__":
    main()
