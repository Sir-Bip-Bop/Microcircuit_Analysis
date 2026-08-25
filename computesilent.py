"""
Events-per-cycle and silent-neuron analysis for the Potjans-Diesmann microcircuit.

Motivation
----------
Under Poisson background drive, the microcircuit produces fast population
oscillations, but a spectral peak alone does not say how the rhythm is
organised in time. The network could emit one discrete population event on
every oscillation cycle, or events could skip cycles and become sparser as the
drive changes. This script asks, per population and per background-rate
condition: does the number of discrete population events match the number of
cycles expected from the dominant frequency (ratio near 1), fall below it
(cycle skipping), or exceed it? And how sparse is participation, i.e. what
fraction of neurons never fires at all in the analysis window?

What it measures
----------------
1. Events-per-cycle ratio: N_events / (T_rec * f0). Population events are
   prominence-thresholded peaks of the binned (1 ms), Gaussian-smoothed
   (sigma 3 ms) population rate. Note: f0 is currently pinned to 80 Hz in
   `analyze_rate_condition`; re-enable the `dominant_frequency` call there to
   estimate it from the Welch PSD instead.
2. Silent-neuron fraction: the fraction of neurons in a population emitting
   zero spikes in [t_start, t_stop], relative to the full-scale population
   sizes in POP_SIZES (override that dict for downscaled runs).

Data layout
-----------
  data_background_rate_fig_1/<rate>/trial_0/spike_recorder-7717<x>-0<y>.dat
    x = population index (0-7 -> L23E, L23I, L4E, L4I, L5E, L5I, L6E, L6I)
    y = virtual process / thread index (0-9)
  Each file: 2 comment lines, then a 'sender  time_ms' header row, then data.

Usage
-----
  # Full analysis on the default data directory; prints both tables and saves
  # the two-panel figure (A: events per cycle, B: silent fraction):
  python epc_and_silent_fraction.py

  # From another script or a notebook:
  from epc_and_silent_fraction import plot_epc_and_silent_fraction
  plot_epc_and_silent_fraction("data_background_rate_fig_1",
                               rate_labels=("06", "09", "12", "18"),
                               out_path="fig1_epc_silent.svg")
"""
import glob
import numpy as np
from scipy.signal import welch, find_peaks
from scipy.ndimage import gaussian_filter1d

POP_NAMES = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]


def load_population_spikes(rate_dir, pop_idx, trial="trial_0"):
    """Concatenate spike times (ms) across all virtual-process files for one population."""
    pattern = f"{rate_dir}/{trial}/spike_recorder-7717{pop_idx}-0*.dat"
    times = []
    n_senders = set()
    for f in sorted(glob.glob(pattern)):
        data = np.loadtxt(f, skiprows=3)  # 2 comment lines + 1 header row
        if data.size == 0:
            continue
        data = data.reshape(-1, 2)
        n_senders.update(data[:, 0].astype(int))
        times.append(data[:, 1])
    if not times:
        return np.array([]), 0
    return np.concatenate(times), len(n_senders)


def population_rate(spike_times, t_start, t_stop, n_neurons, bin_ms=1.0, sigma_ms=3.0):
    """Binned, Gaussian-smoothed population firing rate (spikes/s per neuron)."""
    bins = np.arange(t_start, t_stop + bin_ms, bin_ms)
    counts, _ = np.histogram(spike_times, bins=bins)
    rate = counts / (n_neurons * bin_ms * 1e-3)
    rate_smooth = gaussian_filter1d(rate, sigma=sigma_ms / bin_ms)
    t = bins[:-1] + bin_ms / 2
    return t, rate_smooth


def dominant_frequency(rate_trace, bin_ms=1.0, fmin=1.0, fmax=100.0):
    """f0 via Welch PSD peak within a plausible cortical-oscillation band."""
    fs = 1000.0 / bin_ms
    f, pxx = welch(rate_trace - rate_trace.mean(), fs=fs, nperseg=min(2048, len(rate_trace)))
    mask = (f >= fmin) & (f <= fmax)
    f0 = f[mask][np.argmax(pxx[mask])]
    return f0


def events_per_cycle(rate_trace, t_start, t_stop, f0, peak_prominence_frac=0.3):
    """Count discrete population events (peaks above a prominence threshold) and
    compare to the number of cycles expected from f0 over the recording window."""
    prominence = peak_prominence_frac * (rate_trace.max() - rate_trace.min())
    peaks, _ = find_peaks(rate_trace, prominence=prominence)
    n_events = len(peaks)
    T_rec_s = (t_stop - t_start) / 1000.0
    expected_cycles = T_rec_s * f0
    ratio = n_events / expected_cycles if expected_cycles > 0 else np.nan
    return n_events, expected_cycles, ratio


def analyze_rate_condition(base_dir, rate_label, t_start=5000.0, t_stop=5500.0, trial="trial_0"):
    rate_dir = f"{base_dir}/{rate_label}"
    results = {}
    for pop_idx, pop_name in enumerate(POP_NAMES):
        spikes, n_neurons = load_population_spikes(rate_dir, pop_idx, trial)
        spikes = spikes[(spikes >= t_start) & (spikes <= t_stop)]
        if spikes.size == 0 or n_neurons == 0:
            results[pop_name] = None
            continue
        t, rate_trace = population_rate(spikes, t_start, t_stop, n_neurons)
        #f0 = dominant_frequency(rate_trace)
        f0 = 80
        n_events, expected_cycles, ratio = events_per_cycle(rate_trace, t_start, t_stop, f0)
        results[pop_name] = dict(f0=f0, n_events=n_events,
                                  expected_cycles=expected_cycles, ratio=ratio)
    return results


# --- Fraction of silent neurons per population/drive ---

# Default population sizes: full-scale Potjans & Diesmann (2014) microcircuit.
# Override this dict if you're running a downscaled model.
POP_SIZES = {
    "L23E": 20683, "L23I": 5834,
    "L4E": 21915, "L4I": 5479,
    "L5E": 4850, "L5I": 1065,
    "L6E": 14395, "L6I": 2948,
}


def silent_fraction(base_dir, rate_label, pop_name, pop_sizes=POP_SIZES,
                     t_start=5000.0, t_stop=5500.0, trial="trial_0"):
    """Fraction of neurons in `pop_name` that emit zero spikes in [t_start, t_stop]."""
    pop_idx = POP_NAMES.index(pop_name)
    rate_dir = f"{base_dir}/{rate_label}"

    pattern = f"{rate_dir}/{trial}/spike_recorder-7717{pop_idx}-0*.dat"
    active_senders = set()
    for f in sorted(glob.glob(pattern)):
        data = np.loadtxt(f, skiprows=3)
        if data.size == 0:
            continue
        data = data.reshape(-1, 2)
        mask = (data[:, 1] >= t_start) & (data[:, 1] <= t_stop)
        active_senders.update(data[mask, 0].astype(int))

    n_total = pop_sizes[pop_name]
    n_active = len(active_senders)
    n_silent = n_total - n_active
    frac_silent = n_silent / n_total
    return dict(n_total=n_total, n_active=n_active, n_silent=n_silent,
                frac_silent=frac_silent)


# --- Combined plot ---

# Okabe-Ito colours (colourblind-safe), one per cortical layer; shared by both panels.
LAYER_COLORS = {
    "L23": "#0072B2",  # blue
    "L4": "#D55E00",   # vermillion
    "L5": "#009E73",   # bluish green
    "L6": "#CC79A7",   # reddish purple (not #0079A7: that is near-identical
                       # to the L2/3 blue; #CC79A7 completes the Okabe-Ito
                       # set. Swap here if #0079A7 was intentional.)
}
INH_ALPHA = 0.6            # bar transparency for inhibitory populations (panel B)
LABEL_FONTSIZE = 14        # axis labels
TICK_FONTSIZE = 12         # tick labels
LEGEND_FONTSIZE = 11       # legend entries
PANEL_LABEL_FONTSIZE = 18  # bold A/B panel labels


def _layer_and_type(pop_name):
    """Split a population name into (layer, is_excitatory): 'L23E' -> ('L23', True)."""
    return pop_name[:-1], pop_name.endswith("E")


def plot_epc_and_silent_fraction(base_dir, rate_labels=("06", "09", "12", "18"),
                                  silent_pops=("L23E", "L23I", "L6E", "L6I"),
                                  out_path="epc_and_silent_fraction.svg"):
    """Two-panel figure. Panel A: events-per-cycle ratio per population vs drive
    rate (excitatory: solid line, circle markers; inhibitory: dashed line,
    square markers; colour encodes layer via LAYER_COLORS). Panel B: grouped
    bars of silent-neuron fraction for each population in `silent_pops` vs
    drive rate, same layer colours, inhibitory bars at alpha=INH_ALPHA."""
    import matplotlib.pyplot as plt

    drive_values = [int(r) for r in rate_labels]

    # Collect events-per-cycle ratio per population, per drive
    epc_by_pop = {pop: [] for pop in POP_NAMES}
    for rate_label in rate_labels:
        res = analyze_rate_condition(base_dir, rate_label)
        for pop in POP_NAMES:
            r = res.get(pop)
            epc_by_pop[pop].append(r["ratio"] if r is not None else np.nan)

    # Collect silent fraction for each requested population
    frac_silent_by_pop = {pop: [] for pop in silent_pops}
    for rate_label in rate_labels:
        for pop in silent_pops:
            s = silent_fraction(base_dir, rate_label, pop)
            frac_silent_by_pop[pop].append(s["frac_silent"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: events-per-cycle ratio
    ax = axes[0]
    for pop in POP_NAMES:
        layer, is_exc = _layer_and_type(pop)
        ax.plot(drive_values, epc_by_pop[pop],
                color=LAYER_COLORS[layer],
                linestyle="-" if is_exc else "--",
                marker="o" if is_exc else "s",
                label=pop)
    # Dotted, so the reference line is not read as an inhibitory (dashed) trace.
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("background rate [spikes/s]", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("events per cycle  $N_{events} / (T_{rec} \\cdot f_0)$",
                  fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, ncol=2)

    # Panel B: silent-neuron fraction
    ax = axes[1]
    n_groups = len(drive_values)
    n_pops = len(silent_pops)
    x = np.arange(n_groups)
    bar_width = 0.8 / n_pops
    for i, pop in enumerate(silent_pops):
        layer, is_exc = _layer_and_type(pop)
        offset = (i - (n_pops - 1) / 2) * bar_width
        ax.bar(x + offset, frac_silent_by_pop[pop], width=bar_width,
               color=LAYER_COLORS[layer],
               alpha=1.0 if is_exc else INH_ALPHA,
               label=pop)
    ax.set_xticks(x)
    ax.set_xticklabels(drive_values)
    ax.set_xlabel("background rate [spikes/s]", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("fraction of silent neurons", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=LEGEND_FONTSIZE)

    fig.tight_layout()
    # Reserve headroom for the panel labels, then place them.
    fig.subplots_adjust(top=0.90)
    for ax_i, panel_label in zip(axes, ("A", "B")):
        ax_i.text(-0.12, 1.02, panel_label, transform=ax_i.transAxes,
                  fontsize=PANEL_LABEL_FONTSIZE, fontweight="bold",
                  va="bottom", ha="left")

    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")
    return fig


if __name__ == "__main__":
    base_dir = "data_background_rate_fig_1"

    print("=== Events-per-cycle ===")
    for rate_label in ["06", "09", "12", "18"]:
        print(f"\n--- {rate_label} spikes/s ---")
        res = analyze_rate_condition(base_dir, rate_label)
        for pop, r in res.items():
            if r is None:
                print(f"  {pop}: no data")
            else:
                print(f"  {pop}: f0={r['f0']:.2f} Hz, "
                      f"N_events={r['n_events']}, "
                      f"expected_cycles={r['expected_cycles']:.2f}, "
                      f"ratio={r['ratio']:.2f}")

    print("\n=== Fraction of silent neurons ===")
    for rate_label in ["06", "09", "12", "18"]:
        for pop in ["L23E", "L23I", "L6E", "L6I"]:
            s = silent_fraction(base_dir, rate_label, pop)
            print(f"  {rate_label} spikes/s, {pop}: "
                  f"{s['n_silent']}/{s['n_total']} silent "
                  f"(frac_silent={s['frac_silent']:.3f})")

    plot_epc_and_silent_fraction(base_dir)