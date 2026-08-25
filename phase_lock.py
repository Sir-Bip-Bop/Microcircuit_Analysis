"""
phase_locking_bicoherence.py

Time-domain test for the ~40 Hz / ~80 Hz relation described in Sections
2-4 and 2-7: is the ~40 Hz component a subharmonic of a single ~80 Hz
rhythm (every-other-cycle failure to mount a full event), or are the two
bands independent oscillations from distinct sub-circuits (Bos et al.,
2016)? The spectrum alone can't decide this, because a locally generated
rhythm is inherited by every downstream population regardless of origin.

Two direct tests are implemented, both operating on population activity
in the time domain (not on the FFT amplitudes already stored in
Fourier_data_final):

  1. n:m phase-locking index (PLV)
         lambda = |< exp(i*(2*phi_low - phi_high)) >|
     between the Hilbert phase of the band-passed ~40 Hz component
     (phi_low) and the ~80 Hz component (phi_high). Under the
     subharmonic account phi_high should track 2*phi_low with a fixed
     offset (lambda large); under the two-oscillator account the two
     phases drift relative to one another (lambda ~ 0).

  2. Bicoherence at (f0/2, f0/2), which sums to f0. A single nonlinear
     process that mints the 80 Hz rhythm and lets 40 Hz leak out of it
     (or vice versa) produces consistent quadratic phase coupling
     between f0/2 and f0/2 -> f0; two independently-generated rhythms do
     not, even if their power spectra overlap in the same bands.

Both statistics carry a surrogate-based significance threshold, because
lambda and bicoherence are bounded in [0, 1] and can look deceptively
large just from finite-sample noise.

This module is meant to be dropped alongside the existing notebook code
(same super_name / list_dirs / addons.analysis_dict / population
loading logic as the compute_FFT cell) and either imported for its
functions, or run directly as the driver at the bottom.
"""

import os
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
import matplotlib.pyplot as plt

import addons  # same analysis_dict used by the existing compute_FFT cell


# ----------------------------------------------------------------------
# 1. Band-pass filtering + instantaneous phase
# ----------------------------------------------------------------------

def bandpass(x, f_lo, f_hi, fs, order=4):
    """Zero-phase Butterworth band-pass filter (SOS form, filtfilt)."""
    nyq = fs / 2.0
    sos = butter(order, [f_lo / nyq, f_hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def inst_phase(x):
    """Instantaneous phase via the analytic signal (Hilbert transform)."""
    return np.angle(hilbert(x))


def extract_phases(x, fs, low_band, high_band, order=4):
    """
    Band-pass a raw population-rate trace into the ~40 Hz and ~80 Hz
    components and return their instantaneous phases.

    low_band, high_band: (f_lo, f_hi) tuples in Hz, e.g. (30, 50) and
    (65, 95). Pick these around the peaks actually present in
    Fourier_data_final for the population/background rate in question
    (see `bands_from_spectrum` below) rather than fixed numbers, since
    both peaks can shift a few Hz with drive.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    low = bandpass(x, *low_band, fs=fs, order=order)
    high = bandpass(x, *high_band, fs=fs, order=order)
    return inst_phase(low), inst_phase(high)


def bands_from_spectrum(freqs, amp, low_search=(25, 55), high_search=(60, 100),
                         half_width=6.0):
    """
    Locate the low (~40 Hz) and high (~80 Hz) peaks in an already-computed
    amplitude spectrum (e.g. a row of Fourier_data_final) and return
    (low_band, high_band, f_low_peak, f_high_peak) with half_width Hz on
    each side of the peak. Falls back to the search-window centre if no
    clear peak is found.
    """
    freqs = np.asarray(freqs)
    amp = np.asarray(amp)

    def peak_in(window):
        lo, hi = window
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            return 0.5 * (lo + hi)
        sub_freqs = freqs[mask]
        sub_amp = amp[mask]
        return sub_freqs[np.argmax(sub_amp)]

    f_low = peak_in(low_search)
    f_high = peak_in(high_search)
    low_band = (max(1.0, f_low - half_width), f_low + half_width)
    high_band = (max(1.0, f_high - half_width), f_high + half_width)
    return low_band, high_band, f_low, f_high


# ----------------------------------------------------------------------
# 2. n:m phase-locking index
# ----------------------------------------------------------------------

def nm_plv(phi_low, phi_high, n=2, m=1):
    """
    lambda = |< exp(i*(n*phi_low - m*phi_high)) >|

    Default n=2, m=1 is the relation predicted by the subharmonic
    account: the ~80 Hz phase advances twice per ~40 Hz cycle, so
    2*phi_low - phi_high should sit near a fixed offset if the two
    bands are really one rhythm and its subharmonic.
    """
    phi_low = np.asarray(phi_low)
    phi_high = np.asarray(phi_high)
    L = min(len(phi_low), len(phi_high))
    diff = n * phi_low[:L] - m * phi_high[:L]
    return np.abs(np.mean(np.exp(1j * diff)))


def plv_surrogate_test(phi_low, phi_high, n=2, m=1, n_surr=500, alpha=0.05, rng=None):
    """
    Null distribution of the n:m PLV built by circularly shifting
    phi_high by a random lag before recombining with phi_low. This
    preserves each band's own autocorrelation/frequency content but
    destroys any fixed phase relation between them.

    Returns: observed_lambda, surrogate_values (n_surr,), threshold (alpha)
    """
    rng = np.random.default_rng(rng)
    L = min(len(phi_low), len(phi_high))
    phi_low = phi_low[:L]
    phi_high = phi_high[:L]

    observed = nm_plv(phi_low, phi_high, n=n, m=m)
    surrogate_vals = np.empty(n_surr)
    for k in range(n_surr):
        shift = rng.integers(1, L - 1)
        surrogate_vals[k] = nm_plv(phi_low, np.roll(phi_high, shift), n=n, m=m)
    threshold = np.quantile(surrogate_vals, 1 - alpha)
    return observed, surrogate_vals, threshold


# ----------------------------------------------------------------------
# 3. Bicoherence at a single frequency pair, e.g. (f0/2, f0/2)
# ----------------------------------------------------------------------

def _segment_epochs(x, fs, seg_len_s, overlap):
    seg_len = int(round(seg_len_s * fs))
    step = max(1, int(round(seg_len * (1 - overlap))))
    n_seg = 1 + (len(x) - seg_len) // step
    if n_seg < 1:
        raise ValueError("Signal shorter than one bicoherence segment; "
                          "reduce seg_len_s or check fs.")
    return np.stack([x[i * step: i * step + seg_len] for i in range(n_seg)])


def bicoherence_value(x, fs, f1, f2, seg_len_s=0.5, overlap=0.5, window=True):
    """
    Direct-method bicoherence at a single frequency pair (f1, f2):

        b(f1,f2) = |< X(f1) X(f2) X*(f1+f2) >|
                   / sqrt( <|X(f1)X(f2)|^2> * <|X(f1+f2)|^2> )

    averaged over overlapping segments. x is the raw (unfiltered)
    population-rate trace -- do not band-pass first, or the coupling
    the statistic is meant to detect gets filtered away.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    epochs = _segment_epochs(x, fs, seg_len_s, overlap)
    n_seg, seg_len = epochs.shape
    if window:
        epochs = epochs * np.hanning(seg_len)

    freqs = np.fft.rfftfreq(seg_len, d=1.0 / fs)
    X = np.fft.rfft(epochs, axis=1)

    def nearest(f):
        return int(np.argmin(np.abs(freqs - f)))

    i1, i2, i12 = nearest(f1), nearest(f2), nearest(f1 + f2)
    x1, x2, x12c = X[:, i1], X[:, i2], np.conj(X[:, i12])

    num = np.abs(np.mean(x1 * x2 * x12c))
    denom = np.sqrt(np.mean(np.abs(x1 * x2) ** 2) * np.mean(np.abs(x12c) ** 2))
    return num / denom if denom > 0 else np.nan


def _phase_randomize(x, rng):
    """Fourier phase-randomized surrogate: same amplitude spectrum, phases
    scrambled -> destroys any cross-frequency phase coupling."""
    n = len(x)
    X = np.fft.rfft(x)
    random_phases = rng.uniform(0, 2 * np.pi, size=X.shape)
    random_phases[0] = 0.0
    if n % 2 == 0:
        random_phases[-1] = 0.0
    X_surr = np.abs(X) * np.exp(1j * random_phases)
    return np.fft.irfft(X_surr, n=n)


def bicoherence_surrogate_test(x, fs, f1, f2, seg_len_s=0.5, overlap=0.5,
                                n_surr=200, alpha=0.05, rng=None):
    """
    Null distribution of bicoherence(f1,f2) from phase-randomized
    surrogates of the same signal (matched power spectrum, no
    cross-frequency phase structure).

    Returns: observed_bicoherence, surrogate_values (n_surr,), threshold (alpha)
    """
    rng = np.random.default_rng(rng)
    observed = bicoherence_value(x, fs, f1, f2, seg_len_s, overlap)
    surrogate_vals = np.empty(n_surr)
    for k in range(n_surr):
        x_surr = _phase_randomize(x, rng)
        surrogate_vals[k] = bicoherence_value(x_surr, fs, f1, f2, seg_len_s, overlap)
    threshold = np.quantile(surrogate_vals, 1 - alpha)
    return observed, surrogate_vals, threshold


# ----------------------------------------------------------------------
# 4. One-call convenience wrapper: raw signal -> both statistics
# ----------------------------------------------------------------------

def phase_locking_and_bicoherence(x, fs, freqs=None, amp=None,
                                   low_band=None, high_band=None,
                                   n=2, m=1, seg_len_s=0.5, overlap=0.5,
                                   n_surr_plv=500, n_surr_bic=200,
                                   alpha=0.05, rng=None):
    """
    Full pipeline for one population-rate trace x sampled at fs (Hz):

      - locate/accept the ~40 Hz and ~80 Hz bands
      - compute the n:m PLV (lambda) with a surrogate significance test
      - compute bicoherence at (f_low_peak, f_low_peak) -> f_high_peak
        with a surrogate significance test

    Either pass low_band/high_band explicitly, or pass freqs/amp (a
    precomputed amplitude spectrum, e.g. a row of Fourier_data_final)
    so the bands are located automatically with bands_from_spectrum.

    Returns a dict with lambda, lambda_threshold, lambda_significant,
    bicoherence, bicoherence_threshold, bicoherence_significant, and the
    band edges / peak frequencies actually used.
    """
    rng = np.random.default_rng(rng)

    if low_band is None or high_band is None:
        if freqs is None or amp is None:
            raise ValueError("Provide either (low_band, high_band) or (freqs, amp).")
        low_band, high_band, f_low_peak, f_high_peak = bands_from_spectrum(freqs, amp)
    else:
        f_low_peak = 0.5 * sum(low_band)
        f_high_peak = 0.5 * sum(high_band)

    phi_low, phi_high = extract_phases(x, fs, low_band, high_band)
    lam, lam_surr, lam_thr = plv_surrogate_test(
        phi_low, phi_high, n=n, m=m, n_surr=n_surr_plv, alpha=alpha, rng=rng)

    f0_half = f_low_peak  # bicoherence at (f0/2, f0/2)
    bic, bic_surr, bic_thr = bicoherence_surrogate_test(
        x, fs, f0_half, f0_half, seg_len_s=seg_len_s, overlap=overlap,
        n_surr=n_surr_bic, alpha=alpha, rng=rng)

    return {
        "low_band": low_band, "high_band": high_band,
        "f_low_peak": f_low_peak, "f_high_peak": f_high_peak,
        "lambda": lam, "lambda_threshold": lam_thr, "lambda_significant": lam > lam_thr,
        "bicoherence": bic, "bicoherence_threshold": bic_thr,
        "bicoherence_significant": bic > bic_thr,
    }


# ----------------------------------------------------------------------
# 5. Driver: loop over background rates / trials / populations
# ----------------------------------------------------------------------
# Mirrors the loading logic of the existing compute_FFT cell (same
# super_name, list_dirs, addons.analysis_dict, population file layout)
# so it can be dropped in right after that cell has already built
# list_dirs, names, Fourier_data_final and FFT_frequencies.

if __name__ == "__main__":

    # --- must match the earlier cells ------------------------------------
    super_name = "data_background_rate_big"
    names = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
    num_trials = 10
    fs_sampling = 5000.0  # Hz; matches the dt=0.2 ms binning used for
                           # analysis_interval_start/end in compute_FFT

    list_dirs = sorted(os.listdir(super_name))
    n_bg = len(list_dirs)
    n_pops = len(names)

    analysis_interval_start = int((addons.analysis_dict["analysis_start"] - 500) / 0.2)
    analysis_interval_end = int((addons.analysis_dict["analysis_end"] - 500) / 0.2)

    Lambda = np.full((n_bg, num_trials, n_pops), np.nan)
    Lambda_thr = np.full((n_bg, num_trials, n_pops), np.nan)
    Bicoh = np.full((n_bg, num_trials, n_pops), np.nan)
    Bicoh_thr = np.full((n_bg, num_trials, n_pops), np.nan)

    # Reuse the FFT amplitude spectrum already on disk/in memory to seed
    # the band search, if available; otherwise fall back to fixed bands.
    try:
        Fourier_data_final  # noqa: F821 (from the earlier notebook cell)
        FFT_frequencies      # noqa: F821
        have_spectrum = True
    except NameError:
        have_spectrum = False

    for i, bg_dir in enumerate(list_dirs):
        name = os.path.join(super_name, bg_dir)
        trial_dirs = sorted(d for d in os.listdir(name) if d != "results")

        for j, trial_dir in enumerate(trial_dirs):
            name_2 = os.path.join(name, trial_dir)
            neuron_id = np.loadtxt(os.path.join(name_2, "population_nodeids.dat"), dtype=int)
            n_pop_here = len(neuron_id)

            for p in range(n_pop_here):
                x = np.loadtxt(os.path.join(
                    name_2, "measurements", "pop_activities", f"pop_activity_{p}.dat"))
                x = x[analysis_interval_start:analysis_interval_end]
                if len(x) < fs_sampling * 0.5:
                    continue  # need at least one bicoherence segment

                if have_spectrum:
                    result = phase_locking_and_bicoherence(
                        x, fs=fs_sampling,
                        freqs=FFT_frequencies, amp=Fourier_data_final[i, p, :],
                        rng=i * 1000 + j * 10 + p)
                else:
                    result = phase_locking_and_bicoherence(
                        x, fs=fs_sampling,
                        low_band=(30, 50), high_band=(65, 95),
                        rng=i * 1000 + j * 10 + p)

                Lambda[i, j, p] = result["lambda"]
                Lambda_thr[i, j, p] = result["lambda_threshold"]
                Bicoh[i, j, p] = result["bicoherence"]
                Bicoh_thr[i, j, p] = result["bicoherence_threshold"]

    Lambda_mean = np.nanmean(Lambda, axis=1)       # (n_bg, n_pops)
    Lambda_thr_mean = np.nanmean(Lambda_thr, axis=1)
    Bicoh_mean = np.nanmean(Bicoh, axis=1)
    Bicoh_thr_mean = np.nanmean(Bicoh_thr, axis=1)

    np.savez("Figure2/phase_locking_bicoherence.npz",
             list_dirs=list_dirs, names=names,
             Lambda=Lambda, Lambda_thr=Lambda_thr,
             Bicoh=Bicoh, Bicoh_thr=Bicoh_thr)

    # --- plotting, matching the style of the existing Figure 2 cell -----
    fs_font = 26
    bg_rates = np.linspace(5, 20, n_bg)

    for stat_name, mean_vals, thr_vals, ylabel, fname in [
        ("lambda", Lambda_mean, Lambda_thr_mean,
         r"$\lambda = |\langle e^{i(2\phi_{low}-\phi_{high})}\rangle|$",
         "Figure2/nm_phase_locking.svg"),
        ("bicoherence", Bicoh_mean, Bicoh_thr_mean,
         r"bicoherence$(f_0/2,f_0/2)$",
         "Figure2/bicoherence.svg"),
    ]:
        plt.subplots(2, 4, figsize=(30, 15), sharex="col", sharey="col", dpi=300)
        for p in range(n_pops):
            plt.subplot(2, 4, p + 1)
            plt.plot(bg_rates, mean_vals[:, p], color="#3288bd", lw=2, label=stat_name)
            plt.plot(bg_rates, thr_vals[:, p], "--", color="#d53e4f",
                     lw=1.5, label="surrogate 95%")
            plt.ylim(0, 1)
            if p in [4, 5, 6, 7]:
                plt.xlabel("background rate [spikes/s]", fontsize=fs_font)
            if p in [0, 4]:
                plt.ylabel(ylabel, fontsize=fs_font)
            plt.title(names[p], fontsize=fs_font)
            if p == 0:
                plt.legend(loc="upper left", fontsize=fs_font - 8)
        plt.tight_layout()
        plt.savefig(fname, bbox_inches="tight", dpi=600)
        plt.close()

    print("Done. Results saved to Figure2/phase_locking_bicoherence.npz, "
          "Figure2/nm_phase_locking.svg, Figure2/bicoherence.svg")