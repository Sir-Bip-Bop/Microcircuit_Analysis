#!/usr/bin/env python3
"""
lfp_proxy_spectra.py

Motivation
----------
The population spike rate of the Potjans-Diesmann microcircuit shows a single
~80 Hz E-I rhythm whose expression changes with background drive: irregular at
low drive, one-event-per-cycle at ~9 spikes/s, intermittent with a ~40 Hz
subharmonic at high drive, and with L6E withdrawing from the rhythm. Spike rates
are not what an electrode measures. The local field potential is dominated by
synaptic (transmembrane) currents onto pyramidal cells, so the question this
script answers is: do the same drive-dependent features (fundamental near 80 Hz,
growth of the f0/2 component, layer specificity) appear in a current-based LFP
proxy, or does the synaptic low-pass reshape them?

The proxy follows Mazzoni et al. (2015, PLoS Comput Biol 11:e1004584), who
compared candidate LFP proxies for LIF networks against a biophysical
forward model. Their best simple proxy is the reference weighted sum (RWS):

    LFP(t) = sum_pyr |I_AMPA(t - 6 ms)|  -  1.65 * sum_pyr |I_GABA(t)|

i.e. excitatory currents delayed by 6 ms and inhibitory currents weighted by
1.65, summed over pyramidal (excitatory) neurons. The older "sum of absolute
synaptic currents" proxy (Mazzoni et al. 2008) is also provided for comparison,
as are the excitatory-only and inhibitory-only sums.

Caveats to keep in mind when reading the output. (i) In NEST iaf_psc_exp the
recorded I_syn_ex includes the current injected by the background Poisson
generator, so the proxy contains the external drive, not only recurrent input.
Mean subtraction removes its DC part; its shot noise remains as a white floor.
(ii) With fixed-sign weights I_syn_ex >= 0 and I_syn_in <= 0 at every neuron, so
summing first and taking the absolute value afterwards is exact. (iii) The
recorded neurons are whatever subset the multimeter was attached to; the proxy
is per recorded neuron, not per population, unless you rescale.

What it measures
----------------
For every background rate, trial and population:
  * the population-summed I_syn_ex(t) and I_syn_in(t) (pA), from all thread
    files of the corresponding multimeter;
  * the chosen LFP proxy time series (default RWS, excitatory populations);
  * its Welch PSD with the same window length used for the rate spectra
    (Hann, 0.8192 s segments, 50 % overlap), after discarding a transient;
  * f0 = spectral peak in 60-100 Hz, band-integrated power in +/- 6 Hz around
    f0 and f0/2, and the subharmonic ratio P(f0/2)/P(f0), matching the
    definitions in Methods 3-4 of the manuscript.

Outputs (in --out):
  lfp_<proxy>_traces.npz    summed currents and proxy per rate/trial/population
  lfp_<proxy>_psd.npz       frequency axis and PSDs
  lfp_<proxy>_metrics.csv   per rate/trial/population: f0, P(f0), P(f0/2), ratio
  psd_grid_<proxy>.png      PSD per population, rates overlaid
  subharmonic_ratio_<proxy>.png   ratio vs background rate, mean +/- SEM

File layout expected (NEST multimeter ASCII):
  <root>/<rate>/trial_<k>/{ex,in}_current-<recorder_gid>-<thread>.dat
Header lines start with '#'; data columns are: sender  time_ms  value.
The 8 recorder GIDs found for each prefix are assigned, in ascending order, to
L23E L23I L4E L4I L5E L5I L6E L6I. Check the printed sender-GID ranges against
the population GID ranges of your network before trusting the labels, or pass
--recorder-map to override.

Usage
-----
  # RWS proxy on the four excitatory populations, all rates and trials found
  python lfp_proxy_spectra.py --root /mnt/d/data_background_rate_big_3 --out lfp_out

  # only some rates, sum-of-absolute-currents proxy, include inhibitory populations
  python lfp_proxy_spectra.py --root /mnt/d/data_background_rate_big_3 \
      --rates 6 9 12 18 --proxy abs_sum --populations all --out lfp_out

  # explicit recorder-to-population map (recorder GID -> population label)
  python lfp_proxy_spectra.py --root ... --recorder-map '{"77170":"L23E","77172":"L23I", ...}'

  # data copied into the WSL filesystem (much faster than /mnt/d for repeated reads)
  python lfp_proxy_spectra.py --root ~/data/data_background_rate_big_3 --out lfp_out
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

POPS = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
E_POPS = [p for p in POPS if p.endswith("E")]
FNAME_RE = re.compile(r"^(ex|in)_current-(\d+)-(\d+)\.dat$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def parse_rate(dirname):
    m = re.search(r"[-+]?\d*\.?\d+", dirname)
    return float(m.group()) if m else None


def read_multimeter_file(path):
    """Return DataFrame with columns sender, time, value.

    Handles NEST 2 (no header) and NEST 3 (comment lines followed by a
    column-name row without '#').
    """
    header = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            first = line.split()[0]
            try:
                float(first)
            except ValueError:
                header = 0          # first non-comment line is the column-name row
            break
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=header,
                     low_memory=False, engine="c")
    if df.shape[1] < 3:
        raise ValueError(f"{path}: expected >= 3 columns, got {df.shape[1]}")
    df = df.iloc[:, :3]
    df.columns = ["sender", "time", "value"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df["sender"] = df["sender"].astype(np.int64)
    return df

def sum_recorder(files):
    """Sum currents over all senders of one recorder (all thread files).

    Returns (time array in ms, summed current, sender_min, sender_max, n_senders).
    """
    total = None
    senders = set()
    for f in files:
        df = read_multimeter_file(f)
        senders.update(df["sender"].unique().tolist())
        s = df.groupby("time", sort=True)["value"].sum()
        total = s if total is None else total.add(s, fill_value=0.0)
    total = total.sort_index()
    return (total.index.to_numpy(dtype=float), total.to_numpy(dtype=float),
            min(senders), max(senders), len(senders))


def discover_recorders(trial_dir):
    """Map prefix ('ex'/'in') -> {recorder_gid: [files]}."""
    rec = {"ex": {}, "in": {}}
    for f in sorted(trial_dir.glob("*_current-*.dat")):
        m = FNAME_RE.match(f.name)
        if not m:
            continue
        prefix, gid, _thread = m.groups()
        rec[prefix].setdefault(int(gid), []).append(f)
    return rec


def build_recorder_map(rec, override):
    """recorder_gid -> population label, per prefix."""
    maps = {}
    for prefix in ("ex", "in"):
        gids = sorted(rec[prefix])
        if override:
            maps[prefix] = {g: override[str(g)] for g in gids if str(g) in override}
            missing = [g for g in gids if str(g) not in override]
            if missing:
                print(f"  warning: {prefix} recorders {missing} not in --recorder-map, skipped")
            continue
        if len(gids) != len(POPS):
            sys.exit(f"Found {len(gids)} '{prefix}_current' recorders, expected {len(POPS)}. "
                     f"GIDs: {gids}. Use --recorder-map.")
        maps[prefix] = dict(zip(gids, POPS))
    return maps


def load_trial(trial_dir, override, wanted_pops):
    """Return dict pop -> dict(time, I_ex, I_in) for one trial."""
    rec = discover_recorders(trial_dir)
    maps = build_recorder_map(rec, override)
    out = {}
    for prefix in ("ex", "in"):
        for gid, pop in maps[prefix].items():
            if pop not in wanted_pops:
                continue
            t, I, smin, smax, n = sum_recorder(rec[prefix][gid])
            print(f"    {prefix}_current recorder {gid} -> {pop}: "
                  f"{n} neurons, sender GIDs {smin}-{smax}, {len(t)} samples")
            d = out.setdefault(pop, {})
            if "time" in d and not np.array_equal(d["time"], t):
                # align on common grid
                common = np.intersect1d(d["time"], t)
                for k in ("I_ex", "I_in"):
                    if k in d:
                        d[k] = d[k][np.isin(d["time"], common)]
                I = I[np.isin(t, common)]
                d["time"] = common
            else:
                d["time"] = t
            d["I_ex" if prefix == "ex" else "I_in"] = I
    return out


# ---------------------------------------------------------------------------
# Proxy and spectra
# ---------------------------------------------------------------------------
def make_proxy(time_ms, I_ex, I_in, kind, delay_ms=6.0, alpha=1.65):
    dt = float(np.median(np.diff(time_ms)))
    if kind == "rws":
        nd = int(round(delay_ms / dt))
        ex_del = np.abs(I_ex[:len(I_ex) - nd]) if nd > 0 else np.abs(I_ex)
        inh = np.abs(I_in[nd:]) if nd > 0 else np.abs(I_in)
        lfp = ex_del - alpha * inh
        t = time_ms[nd:]
    elif kind == "abs_sum":
        lfp, t = np.abs(I_ex) + np.abs(I_in), time_ms
    elif kind == "ex":
        lfp, t = np.abs(I_ex), time_ms
    elif kind == "in":
        lfp, t = np.abs(I_in), time_ms
    else:
        raise ValueError(kind)
    return t, lfp, dt


def psd(x, fs, window_s):
    nperseg = int(round(window_s * fs))
    nperseg = min(nperseg, len(x))
    f, P = welch(x - x.mean(), fs=fs, window="hann", nperseg=nperseg,
                 noverlap=nperseg // 2, scaling="density", detrend=False)
    return f, P


def band_power(f, P, fc, half_width):
    m = (f >= fc - half_width) & (f <= fc + half_width)
    return np.trapz(P[m], f[m]) if m.sum() > 1 else float(P[m].sum())


def metrics(f, P, f0_band=(60.0, 100.0), half_width=6.0):
    m = (f >= f0_band[0]) & (f <= f0_band[1])
    f0 = f[m][np.argmax(P[m])]
    p0 = band_power(f, P, f0, half_width)
    p_half = band_power(f, P, f0 / 2.0, half_width)
    return dict(f0=f0, P_f0=p0, P_f0_half=p_half,
                ratio=p_half / p0 if p0 > 0 else np.nan)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all(rates, pops, psds, met, proxy, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    cols = {r: cmap(i / max(1, len(rates) - 1)) for i, r in enumerate(rates)}

    # PSD grid: one panel per population, rates overlaid (mean over trials)
    n = len(pops)
    ncol = 4 if n > 4 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    for ax, pop in zip(axes.ravel(), pops):
        for r in rates:
            entries = [v for (rr, _tr, pp), v in psds.items() if rr == r and pp == pop]
            if not entries:
                continue
            f = entries[0][0]
            Pm = np.mean([e[1] for e in entries], axis=0)
            ax.semilogy(f, Pm, color=cols[r], lw=1.2, label=f"{r:g}")
        ax.axvline(80, ls=":", c="k", lw=0.8)
        ax.axvline(40, ls="--", c="k", lw=0.8)
        ax.set_xlim(0, 150)
        ax.set_title(pop)
        ax.set_xlabel("frequency (Hz)")
    for ax in axes.ravel()[len(pops):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("PSD (pA$^2$/Hz)")
    axes[0, 3].legend(title="bg rate (spikes/s)", fontsize=8,loc='center left', bbox_to_anchor=(1, 0.5), ncols = 3)
    fig.suptitle(f"LFP proxy ({proxy}) power spectra")
    fig.tight_layout()
    fig.savefig(out / f"psd_grid_{proxy}.png", dpi=150)
    plt.close(fig)

    # subharmonic ratio vs rate
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for pop in pops:
        sub = met[met["population"] == pop].groupby("rate")["ratio"]
        mean, sem = sub.mean(), sub.sem()
        ax.errorbar(mean.index, mean.values, yerr=sem.values, marker="o",
                    ls="-" if pop.endswith("E") else "--", label=pop, capsize=2)
    ax.axhline(0.5, ls=":", c="k", lw=0.8)
    ax.set_xlabel("background rate (spikes/s)")
    ax.set_ylabel("P(f0/2) / P(f0)")
    ax.set_title(f"Subharmonic ratio, LFP proxy ({proxy})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"subharmonic_ratio_{proxy}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/mnt/d/data_background_rate_big_3")
    ap.add_argument("--out", default="lfp_out")
    ap.add_argument("--rates", nargs="*", type=float, default=None,
                    help="background rates to include (default: all found)")
    ap.add_argument("--proxy", choices=["rws", "abs_sum", "ex", "in"], default="rws")
    ap.add_argument("--populations", default="E",
                    help="'E' (default), 'all', or comma-separated list e.g. L4E,L6E")
    ap.add_argument("--transient-ms", type=float, default=500.0)
    ap.add_argument("--window-s", type=float, default=0.8192,
                    help="Welch segment length in seconds (4096 samples at 5 kHz)")
    ap.add_argument("--delay-ms", type=float, default=6.0, help="RWS excitatory delay")
    ap.add_argument("--alpha", type=float, default=1.65, help="RWS inhibitory weight")
    ap.add_argument("--recorder-map", type=str, default=None,
                    help="JSON string or path: recorder GID -> population label")
    args = ap.parse_args()

    root, out = Path(args.root).expanduser(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        sys.exit(f"{root} not found. In WSL, Windows D: is /mnt/d (sudo mount -t drvfs D: /mnt/d if missing).")

    if args.populations == "E":
        pops = E_POPS
    elif args.populations == "all":
        pops = POPS
    else:
        pops = [p.strip() for p in args.populations.split(",")]

    override = None
    if args.recorder_map:
        p = Path(args.recorder_map)
        override = json.loads(p.read_text() if p.is_file() else args.recorder_map)

    rate_dirs = {}
    for d in sorted(root.iterdir()):
        if d.is_dir():
            r = parse_rate(d.name)
            if r is not None and (args.rates is None or r in args.rates):
                rate_dirs[r] = d
    if not rate_dirs:
        sys.exit(f"No rate directories found under {root}")
    rates = sorted(rate_dirs)
    print(f"Rates: {rates}")

    traces, psds, rows = {}, {}, []
    for r in rates:
        trial_dirs = sorted(d for d in rate_dirs[r].iterdir() if d.is_dir() and d.name.startswith("trial"))
        for td in trial_dirs:
            trial = int(re.search(r"\d+", td.name).group()) if re.search(r"\d+", td.name) else 0
            print(f"rate {r:g}, {td.name}")
            data = load_trial(td, override, set(pops))
            for pop in pops:
                if pop not in data or "I_ex" not in data[pop] or "I_in" not in data[pop]:
                    print(f"    {pop}: missing ex or in currents, skipped")
                    continue
                d = data[pop]
                t, lfp, dt = make_proxy(d["time"], d["I_ex"], d["I_in"], args.proxy,
                                        args.delay_ms, args.alpha)
                keep = t >= (t[0] + args.transient_ms)
                t, lfp = t[keep], lfp[keep]
                fs = 1000.0 / dt
                f, P = psd(lfp, fs, args.window_s)
                m = metrics(f, P)
                traces[f"{r:g}|{trial}|{pop}"] = np.column_stack([t, lfp])
                traces[f"{r:g}|{trial}|{pop}|I_ex"] = d["I_ex"]
                traces[f"{r:g}|{trial}|{pop}|I_in"] = d["I_in"]
                psds[(r, trial, pop)] = (f, P)
                rows.append(dict(rate=r, trial=trial, population=pop, fs_Hz=fs,
                                 n_samples=len(lfp), **m))
                print(f"    {pop}: fs={fs:.0f} Hz, f0={m['f0']:.1f} Hz, "
                      f"P(f0/2)/P(f0)={m['ratio']:.3f}")

    if not rows:
        sys.exit("Nothing computed. Check the file layout and recorder mapping.")

    met = pd.DataFrame(rows)
    met.to_csv(out / f"lfp_{args.proxy}_metrics.csv", index=False)
    np.savez_compressed(out / f"lfp_{args.proxy}_traces.npz", **traces)
    np.savez_compressed(out / f"lfp_{args.proxy}_psd.npz",
                        **{f"{r:g}|{tr}|{pop}": np.column_stack(v) for (r, tr, pop), v in psds.items()})
    plot_all(rates, pops, psds, met, args.proxy, out)

    summary = met.groupby(["population", "rate"])["ratio"].agg(["mean", "sem", "count"])
    print("\nSubharmonic ratio P(f0/2)/P(f0), mean over trials:")
    print(summary.to_string())
    print(f"\nWritten to {out.resolve()}")


if __name__ == "__main__":
    main()