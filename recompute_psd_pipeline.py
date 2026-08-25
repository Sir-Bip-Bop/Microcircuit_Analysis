"""
recompute_psd_pipeline.py

Walks the data tree

    data_background_rate_big/[RATE]/trial_[X]/measurements/pop_activities/pop_activity_[Y].dat

recomputes the Welch PSD for every (rate, trial, population), averages PSDs
across trials for each (rate, population), and produces the quantities
needed for Fig. 2:

  - psd[population][rate]          -> Fig. 2A-H (per-population spectra
                                       across drive) and 2L-S (spectrograms,
                                       just psd stacked over sorted rates)
  - subharmonic_ratio[population][rate] = P(f0/2) / P(f0), f0 = 80 Hz
                                       -> Fig. 2K, T

It then reuses `test_subharmonic_coupling` from subharmonic_coupling.py
(2:1 phase-locking value + bicoherence at (f0/2, f0/2)) on the same raw
per-trial signals, per (population, rate).

===========================================================================
CONFIG -- check/adjust these before running
===========================================================================
"""

import os
import re
import numpy as np
from scipy.signal import welch

from subharmonic_coupling import test_subharmonic_coupling

# --- paths -----------------------------------------------------------------
BASE_DIR = "data_background_rate_big"

# --- rate subsampling ---------------------------------------------------------
# Process only every RATE_STRIDE-th background rate (sorted numerically).
# Set to 1 to process every rate. Useful to cut runtime/memory or work
# around a crash on a specific rate while you debug it.
RATE_STRIDE = 5

# --- population bookkeeping --------------------------------------------------
# Adjust order/labels if pop_activity_[Y].dat is indexed differently.
POP_LABELS = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
N_POPULATIONS = len(POP_LABELS)

# --- timing ------------------------------------------------------------------
# The raw pop_activity_[Y].dat files are natively sampled every
# RAW_BIN_SIZE_MS (i.e. this is the true time spacing between rows if the
# file is a single column of values). This matches the "spacing between
# time is 0.2 ms" used to index the raw array in your compute_FFT setup
# (analysis_interval_start = (t - 500) / 0.2).
RAW_BIN_SIZE_MS = 0.017  # <-- native sample spacing of the raw .dat files (ms)

# Spectral analysis (Welch/FFT) is run at a coarser, fixed sample interval,
# matching `freq_sample=0.001` (seconds) in your compute_FFT -- i.e. the
# raw signal is binned/downsampled to 1 ms resolution (1000 Hz) before any
# FFT/Welch call. RAW_BIN_SIZE_MS must evenly divide FFT_SAMPLE_INTERVAL_MS.
FFT_SAMPLE_INTERVAL_MS = 1.7  # <-- matches freq_sample = 0.001 s

# How to combine RAW_BIN_SIZE_MS-spaced samples into each
# FFT_SAMPLE_INTERVAL_MS-spaced bin. "mean" is appropriate if the raw
# column is already a rate/amplitude signal; use "sum" instead if the raw
# column is a spike COUNT per 0.2 ms bin (summing counts before dividing
# by the new, larger bin width is what actually preserves a rate).
DOWNSAMPLE_METHOD = "mean"  # "mean" or "sum"

# If a .dat file instead has two columns (time, value), fs is inferred
# directly from the time column and this raw/FFT downsampling is skipped
# -- adjust `load_pop_activity` if you need downsampling in that case too.

# --- fundamental frequency (given) -------------------------------------------
F0 = 80.0  # Hz

# --- Welch parameters ---------------------------------------------------------
WELCH_NPERSEG_SEC = 1.0   # segment length in seconds -> 1 Hz resolution
WELCH_NOVERLAP_FRAC = 0.5

# --- subharmonic band definition (for the P(f0/2)/P(f0) ratio) --------------
BAND_HALFWIDTH_HZ = 3.0  # integrate +/- this many Hz around f0/2 and f0


# ==============================================================================
# Data loading
# ==============================================================================

def downsample_signal(signal, raw_bin_size_ms=RAW_BIN_SIZE_MS,
                       target_bin_size_ms=FFT_SAMPLE_INTERVAL_MS,
                       method=DOWNSAMPLE_METHOD):
    """
    Bin raw_bin_size_ms-spaced samples into target_bin_size_ms-spaced bins.

    e.g. raw 0.2 ms -> target 1 ms bins the signal by a factor of 5.
    Trailing samples that don't fill a complete bin are dropped.
    """
    factor = target_bin_size_ms / raw_bin_size_ms
    if abs(factor - round(factor)) > 1e-6:
        raise ValueError(
            f"target_bin_size_ms ({target_bin_size_ms}) must be an integer "
            f"multiple of raw_bin_size_ms ({raw_bin_size_ms}); got factor={factor}"
        )
    factor = int(round(factor))
    if factor == 1:
        return signal, 1000.0 / target_bin_size_ms

    n_bins = len(signal) // factor
    trimmed = signal[: n_bins * factor].reshape(n_bins, factor)
    binned = trimmed.mean(axis=1) if method == "mean" else trimmed.sum(axis=1)
    fs_new = 1000.0 / target_bin_size_ms
    return binned, fs_new


def load_pop_activity(filepath, raw_bin_size_ms=RAW_BIN_SIZE_MS,
                       target_bin_size_ms=FFT_SAMPLE_INTERVAL_MS):
    """
    Load one pop_activity_[Y].dat file and return it at the sample rate
    used for spectral analysis (FFT_SAMPLE_INTERVAL_MS), downsampling
    from the file's native resolution (RAW_BIN_SIZE_MS) if needed.

    Auto-detects:
      - single column  -> raw signal, uniformly sampled at `raw_bin_size_ms`,
                           then binned down to `target_bin_size_ms`
      - two columns    -> (time, rate); fs inferred from the median time
                           step directly, no downsampling applied

    Returns
    -------
    rate : 1D array
    fs : float (Hz)
    """
    arr = np.loadtxt(filepath)
    if arr.ndim == 1:
        rate, fs = downsample_signal(arr, raw_bin_size_ms, target_bin_size_ms)
    elif arr.ndim == 2 and arr.shape[1] >= 2:
        time_col, rate = arr[:, 0], arr[:, 1]
        dt = np.median(np.diff(time_col))
        # heuristic: if dt looks like milliseconds (>0.01), convert; if
        # already in seconds, leave as is
        fs = 1000.0 / dt if dt > 0.01 else 1.0 / dt
    else:
        raise ValueError(f"Unexpected data shape {arr.shape} in {filepath}")
    return rate, fs


def discover_rates(base_dir=BASE_DIR):
    """All [RATE] subdirectories, sorted numerically where possible."""
    entries = [d for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d))]

    def sort_key(name):
        m = re.search(r"[-+]?\d*\.?\d+", name)
        return float(m.group()) if m else name

    return sorted(entries, key=sort_key)


def discover_trials(rate_dir):
    return sorted(
        d for d in os.listdir(rate_dir)
        if os.path.isdir(os.path.join(rate_dir, d)) and d.startswith("trial_")
    )


def trial_pop_file(base_dir, rate, trial, pop_idx):
    return os.path.join(base_dir, rate, trial, "measurements",
                         "pop_activities", f"pop_activity_{pop_idx}.dat")


# ==============================================================================
# Welch PSD
# ==============================================================================

def welch_psd(signal, fs, nperseg_sec=WELCH_NPERSEG_SEC,
              noverlap_frac=WELCH_NOVERLAP_FRAC):
    nperseg = int(nperseg_sec * fs)
    noverlap = int(nperseg * noverlap_frac)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap,
                        detrend="constant")
    return freqs, psd


def band_power(freqs, psd, f_center, halfwidth=BAND_HALFWIDTH_HZ):
    """Integrate PSD in [f_center - halfwidth, f_center + halfwidth]."""
    mask = (freqs >= f_center - halfwidth) & (freqs <= f_center + halfwidth)
    if not np.any(mask):
        return np.nan
    return np.trapz(psd[mask], freqs[mask])


# ==============================================================================
# Main pipeline
# ==============================================================================

def run_pipeline(base_dir=BASE_DIR, f0=F0, do_coupling_tests=True,
                  n_surrogates=300, rate_stride=1):
    """
    rate_stride : int
        Analyse every `rate_stride`-th background rate (after numeric
        sorting) instead of all of them, e.g. rate_stride=3 keeps rates
        at indices 0, 3, 6, ... Use this to cut runtime if the full
        sweep is too slow/crashes; set to 1 to analyse every rate.
    """
    all_rates = discover_rates(base_dir)
    rates = all_rates[::rate_stride]
    print(f"Found {len(all_rates)} rates, analysing {len(rates)} "
          f"(every {rate_stride}): {rates}")

    # results[pop_label][rate] = dict with freqs, psd_mean, psd_per_trial,
    # subharmonic_ratio, and (optionally) plv / bicoherence
    results = {label: {} for label in POP_LABELS}

    for rate in rates:
        rate_dir = os.path.join(base_dir, rate)
        trials = discover_trials(rate_dir)
        if not trials:
            print(f"  [!] no trials found under {rate_dir}, skipping")
            continue

        for pop_idx, pop_label in enumerate(POP_LABELS):
            try:
                psd_per_trial = []
                raw_signals = []
                freqs_ref = None
                fs_ref = None

                for trial in trials:
                    fpath = trial_pop_file(base_dir, rate, trial, pop_idx)
                    if not os.path.exists(fpath):
                        print(f"  [!] missing {fpath}")
                        continue

                    try:
                        signal, fs = load_pop_activity(fpath)
                        freqs, psd = welch_psd(signal, fs)
                    except Exception as e:
                        print(f"  [!] failed to process {fpath}: {e}")
                        continue

                    if freqs_ref is None:
                        freqs_ref = freqs
                        fs_ref = fs
                    elif len(freqs) != len(freqs_ref):
                        # different trial length -> interpolate onto reference grid
                        psd = np.interp(freqs_ref, freqs, psd)

                    psd_per_trial.append(psd)
                    raw_signals.append(signal)

                if not psd_per_trial:
                    continue

                psd_per_trial = np.array(psd_per_trial)
                psd_mean = psd_per_trial.mean(axis=0)

                p_low = band_power(freqs_ref, psd_mean, f0 / 2.0)
                p_high = band_power(freqs_ref, psd_mean, f0)
                subharmonic_ratio = p_low / p_high if p_high > 0 else np.nan

                entry = {
                    "freqs": freqs_ref,
                    "fs": fs_ref,
                    "psd_mean": psd_mean,
                    "psd_per_trial": psd_per_trial,
                    "n_trials": len(psd_per_trial),
                    "band_power_f0_2": p_low,
                    "band_power_f0": p_high,
                    "subharmonic_ratio": subharmonic_ratio,
                }

                if do_coupling_tests:
                    # concatenate trials for the PLV surrogate test; run
                    # bicoherence per trial-segment via nperseg = one trial
                    concat_signal = np.concatenate(raw_signals)
                    coupling = test_subharmonic_coupling(
                        concat_signal, fs_ref, f0,
                        n_surrogates=n_surrogates,
                        bicoh_nperseg=len(raw_signals[0]),
                        bicoh_noverlap=0,
                    )
                    entry.update(coupling)

                results[pop_label][rate] = entry

                print(f"  {pop_label:5s} rate={rate:>6s}  "
                      f"n_trials={entry['n_trials']:3d}  "
                      f"P(f0/2)/P(f0)={subharmonic_ratio:.3f}"
                      + (f"  PLV={entry.get('plv', float('nan')):.3f}"
                         f"  bicoh={entry.get('bicoherence', float('nan')):.3f}"
                         if do_coupling_tests else ""))

            except Exception as e:
                print(f"  [!] {pop_label} rate={rate}: unexpected failure, "
                      f"skipping ({type(e).__name__}: {e})")
                continue

    return results, rates


# ==============================================================================
# Plotting
# ==============================================================================

def _rate_to_float(rate_label):
    m = re.search(r"[-+]?\d*\.?\d+", rate_label)
    return float(m.group()) if m else np.nan


def plot_spectra_and_spectrograms(results, rates, f0=F0, freq_max=150.0,
                                   outpath="fig2_spectra.png"):
    """
    Fig. 2A-H / L-S style figure:
      - one PSD panel per population, one curve per drive rate
      - below it, one spectrogram (freq x rate heatmap) per population
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    n_pop = len(POP_LABELS)
    rate_vals = [_rate_to_float(r) for r in rates]
    norm = Normalize(vmin=min(rate_vals), vmax=max(rate_vals))
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(2, n_pop, figsize=(3.0 * n_pop, 6.5),
                              constrained_layout=True)

    for i, pop_label in enumerate(POP_LABELS):
        ax_psd = axes[0, i]
        ax_spec = axes[1, i]

        pop_results = results.get(pop_label, {})
        if not pop_results:
            ax_psd.set_visible(False)
            ax_spec.set_visible(False)
            continue

        # --- PSD panel (one line per rate) ---
        for rate in rates:
            if rate not in pop_results:
                continue
            entry = pop_results[rate]
            freqs = entry["freqs"]
            mask = freqs <= freq_max
            color = cmap(norm(_rate_to_float(rate)))
            ax_psd.semilogy(freqs[mask], entry["psd_mean"][mask],
                             color=color, lw=1.2)

        ax_psd.axvline(f0, color="k", ls=":", lw=0.8, alpha=0.5)
        ax_psd.axvline(f0 / 2, color="k", ls="--", lw=0.8, alpha=0.5)
        ax_psd.set_title(pop_label, fontsize=11)
        ax_psd.set_xlim(0, freq_max)
        if i == 0:
            ax_psd.set_ylabel("PSD (Welch)")
        ax_psd.set_xlabel("Frequency (Hz)")

        # --- spectrogram panel: freq (y) x rate (x) heatmap ---
        available_rates = [r for r in rates if r in pop_results]
        freqs_ref = pop_results[available_rates[0]]["freqs"]
        mask = freqs_ref <= freq_max
        spec = np.array([np.log10(pop_results[r]["psd_mean"][mask] + 1e-20)
                          for r in available_rates]).T  # freq x rate
        rate_x = [_rate_to_float(r) for r in available_rates]

        im = ax_spec.pcolormesh(rate_x, freqs_ref[mask], spec, shading="auto",
                                 cmap="magma")
        ax_spec.axhline(f0, color="w", ls=":", lw=0.8, alpha=0.6)
        ax_spec.axhline(f0 / 2, color="w", ls="--", lw=0.8, alpha=0.6)
        ax_spec.set_xlabel("Background rate (spikes/s)")
        if i == 0:
            ax_spec.set_ylabel("Frequency (Hz)")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes[0, :].tolist(), label="Background rate (spikes/s)",
                 shrink=0.8, pad=0.01)
    fig.colorbar(im, ax=axes[1, :].tolist(), label="log10 PSD",
                 shrink=0.8, pad=0.01)

    fig.suptitle("Population spectra across drive (top) and drive-resolved "
                  "spectrograms (bottom)", fontsize=12)
    fig.savefig(outpath, dpi=200)
    print(f"Saved {outpath}")
    return fig


def plot_ratio_and_coupling(results, rates, f0=F0, outpath="fig2_ratio_coupling.png"):
    """
    Fig. 2K/T style figure: subharmonic power ratio, 2:1 PLV, and
    bicoherence at (f0/2, f0/2), each vs. drive, one line per population.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    n_pop = len(POP_LABELS)
    cmap = plt.get_cmap("tab10" if n_pop <= 10 else "tab20")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    ax_ratio, ax_plv, ax_bic = axes

    for i, pop_label in enumerate(POP_LABELS):
        pop_results = results.get(pop_label, {})
        if not pop_results:
            continue
        available_rates = [r for r in rates if r in pop_results]
        rate_x = [_rate_to_float(r) for r in available_rates]
        color = cmap(i % cmap.N)

        ratio = [pop_results[r]["subharmonic_ratio"] for r in available_rates]
        ax_ratio.plot(rate_x, ratio, "-o", color=color, label=pop_label,
                       ms=3, lw=1.3)

        if "plv" in pop_results[available_rates[0]]:
            plv = [pop_results[r]["plv"] for r in available_rates]
            plv_p = [pop_results[r]["plv_p_value"] for r in available_rates]
            ax_plv.plot(rate_x, plv, "-o", color=color, ms=3, lw=1.3)
            sig = [p < 0.05 for p in plv_p]
            ax_plv.scatter(np.array(rate_x)[sig], np.array(plv)[sig],
                            marker="*", color=color, s=60, zorder=5)

            bic = [pop_results[r]["bicoherence"] for r in available_rates]
            bic_p99 = [pop_results[r]["bicoherence_p99"] for r in available_rates]
            ax_bic.plot(rate_x, bic, "-o", color=color, ms=3, lw=1.3)
            above = np.array(bic) > np.array(bic_p99)
            ax_bic.scatter(np.array(rate_x)[above], np.array(bic)[above],
                            marker="*", color=color, s=60, zorder=5)

    ax_ratio.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.6,
                      label="expected 0.5")
    ax_ratio.set_xlabel("Background rate (spikes/s)")
    ax_ratio.set_ylabel("P(f0/2) / P(f0)")
    ax_ratio.set_title("Subharmonic power ratio")
    ax_ratio.legend(fontsize=7, ncol=2)

    ax_plv.set_xlabel("Background rate (spikes/s)")
    ax_plv.set_ylabel("2:1 phase-locking value")
    ax_plv.set_title("Phase locking (* = p<0.05 vs. surrogate)")
    ax_plv.set_ylim(0, 1.05)

    ax_bic.set_xlabel("Background rate (spikes/s)")
    ax_bic.set_ylabel(f"Bicoherence at (f0/2, f0/2)")
    ax_bic.set_title("Bicoherence (* = above 99th pct null)")
    ax_bic.set_ylim(0, 1.05)

    fig.savefig(outpath, dpi=200)
    print(f"Saved {outpath}")
    return fig


if __name__ == "__main__":
    results, rates = run_pipeline(rate_stride=RATE_STRIDE)

    # Example: build the Fig. 2K/T style array (subharmonic ratio vs rate,
    # one line per population) once you're ready to plot.
    ratio_matrix = np.full((len(POP_LABELS), len(rates)), np.nan)
    for i, pop_label in enumerate(POP_LABELS):
        for j, rate in enumerate(rates):
            if rate in results[pop_label]:
                ratio_matrix[i, j] = results[pop_label][rate]["subharmonic_ratio"]

    np.save("subharmonic_ratio_matrix.npy", ratio_matrix)
    print("\nSaved subharmonic_ratio_matrix.npy, shape:", ratio_matrix.shape)
    print("Rows (populations):", POP_LABELS)
    print("Cols (rates):", rates)

    plot_spectra_and_spectrograms(results, rates)
    plot_ratio_and_coupling(results, rates)