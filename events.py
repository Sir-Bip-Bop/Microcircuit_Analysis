"""
Events-per-cycle ratio: N_events / (T_rec * f0)

Assumes directory structure:
  data_background_rate_fig_1/<rate>/trial_0/spike_recorder-7717<x>-0<y>.dat
    x = population index (0-7 -> L23E,L23I,L4E,L4I,L5E,L5I,L6E,L6I)
    y = virtual process / thread index (0-9)

Each file: 2 comment lines, then a 'sender  time_ms' header row, then data.
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
        f0 = 40.0
        n_events, expected_cycles, ratio = events_per_cycle(rate_trace, t_start, t_stop, f0)
        results[pop_name] = dict(f0=f0, n_events=n_events,
                                  expected_cycles=expected_cycles, ratio=ratio)
    return results


if __name__ == "__main__":
    base_dir = "data_background_rate_fig_1"
    for rate_label in ["06", "09", "12", "18"]:
        print(f"\n=== {rate_label} spikes/s ===")
        res = analyze_rate_condition(base_dir, rate_label)
        for pop, r in res.items():
            if r is None:
                print(f"  {pop}: no data")
            else:
                print(f"  {pop}: f0={r['f0']:.2f} Hz, "
                      f"N_events={r['n_events']}, "
                      f"expected_cycles={r['expected_cycles']:.2f}, "
                      f"ratio={r['ratio']:.2f}")