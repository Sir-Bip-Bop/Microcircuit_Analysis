"""
subharmonic_coupling.py

Tests whether the ~40 Hz component is a true subharmonic of the ~80 Hz
rhythm (phase-locked, nonlinearly coupled) rather than an independently
generated oscillation that merely coexists with it.

Two complementary measures, both computed per population and per drive level:

1. n:m phase-locking value (here n:m = 2:1)

       lambda = | < exp( i * (2*phi_low - phi_high) ) > |

   phi_low  = instantaneous phase of the band-limited signal around f0/2 (~40 Hz)
   phi_high = instantaneous phase of the band-limited signal around f0   (~80 Hz)
   <.>      = average over time (and, if available, over trials)

   lambda = 1 means the low-band phase advances at exactly half the rate of
   the high-band phase at all times (perfect subharmonic locking).
   lambda ~ 0 means the two phases drift independently.

   Significance is assessed against a surrogate distribution built by
   circularly time-shifting one of the two phase series, which destroys
   any true cross-frequency relationship while preserving each signal's
   own autocorrelation structure.

2. Bicoherence at (f0/2, f0/2)

       B(f1,f2) = | < X(f1) X(f2) X*(f1+f2) > |
                  ----------------------------------------
                  sqrt( <|X(f1)X(f2)|^2> * <|X(f1+f2)|^2> )

   evaluated at f1 = f2 = f0/2, so f1+f2 = f0. This is the direct
   frequency-domain signature of quadratic phase coupling between the
   40 Hz component and the 80 Hz component: it is elevated above chance
   only if the phase of the 80 Hz component is consistently related to
   twice the phase of the 40 Hz component, which is exactly what
   "40 Hz is the subharmonic of 80 Hz" predicts.

Both quantities are complementary: PLV is a time-domain phase statistic
computed on band-filtered signals, bicoherence is computed directly in
the frequency domain from segmented FFTs (no filtering, no phase
extraction), so agreement between the two is fairly strong evidence
that the coupling is real and not a filtering artefact.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt, welch


# --------------------------------------------------------------------------
# 1. n:m (2:1) phase-locking value
# --------------------------------------------------------------------------

def bandpass(x, fs, f_lo, f_hi, order=4):
    """Zero-phase Butterworth band-pass filter."""
    nyq = fs / 2.0
    sos = butter(order, [f_lo / nyq, f_hi / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def instantaneous_phase(x):
    """Instantaneous phase via the analytic signal (Hilbert transform)."""
    return np.angle(hilbert(x))


def phase_locking_value_2_1(signal, fs, f0, band_halfwidth_low=6.0,
                             band_halfwidth_high=10.0, n=2, m=1):
    """
    n:m phase-locking value between the f0/2 band and the f0 band.

    Parameters
    ----------
    signal : 1D array
        Population activity time series (e.g. rate signal), single trial
        or a single concatenated/epoched trace.
    fs : float
        Sampling rate (Hz).
    f0 : float
        Fundamental (fast) frequency for this drive level, in Hz
        (e.g. the peak frequency of the ~80 Hz band from Fig. 2A-H).
    band_halfwidth_low, band_halfwidth_high : float
        Half-widths (Hz) of the pass-bands around f0/2 and f0 respectively.
    n, m : int
        Locking order; n applies to the low-frequency phase, m to the
        high-frequency phase. Default 2:1 for a true subharmonic.

    Returns
    -------
    lam : float
        Phase-locking value, in [0, 1].
    phase_diff : 1D array
        The instantaneous n:m phase difference time series (radians),
        useful for plotting/diagnostics (e.g. circular histogram).
    """
    f_low = f0 / 2.0
    x_low = bandpass(signal, fs, f_low - band_halfwidth_low, f_low + band_halfwidth_low)
    x_high = bandpass(signal, fs, f0 - band_halfwidth_high, f0 + band_halfwidth_high)

    phi_low = instantaneous_phase(x_low)
    phi_high = instantaneous_phase(x_high)

    phase_diff = n * phi_low - m * phi_high
    lam = np.abs(np.mean(np.exp(1j * phase_diff)))
    return lam, phase_diff


def plv_with_surrogates(signal, fs, f0, n_surrogates=500, min_shift_sec=1.0,
                         rng=None, **plv_kwargs):
    """
    2:1 PLV plus a circular-shift surrogate null distribution and p-value.

    The high-frequency phase series is circularly shifted by a random
    amount (at least min_shift_sec, to avoid trivial near-zero shifts)
    before recombining with the unshifted low-frequency phase. This
    preserves each band's own autocorrelation/spectral content while
    destroying any genuine cross-frequency phase relationship.

    Returns
    -------
    lam : float          observed PLV
    p_value : float      fraction of surrogates >= observed PLV
    surrogate_lams : 1D array   the surrogate null distribution
    """
    if rng is None:
        rng = np.random.default_rng()

    f_low = f0 / 2.0
    band_halfwidth_low = plv_kwargs.get("band_halfwidth_low", 6.0)
    band_halfwidth_high = plv_kwargs.get("band_halfwidth_high", 10.0)

    x_low = bandpass(signal, fs, f_low - band_halfwidth_low, f_low + band_halfwidth_low)
    x_high = bandpass(signal, fs, f0 - band_halfwidth_high, f0 + band_halfwidth_high)

    phi_low = instantaneous_phase(x_low)
    phi_high = instantaneous_phase(x_high)

    lam = np.abs(np.mean(np.exp(1j * (2 * phi_low - phi_high))))

    n_samp = len(signal)
    min_shift = int(min_shift_sec * fs)
    surrogate_lams = np.empty(n_surrogates)
    for i in range(n_surrogates):
        shift = rng.integers(min_shift, n_samp - min_shift)
        phi_high_shift = np.roll(phi_high, shift)
        surrogate_lams[i] = np.abs(np.mean(np.exp(1j * (2 * phi_low - phi_high_shift))))

    p_value = np.mean(surrogate_lams >= lam)
    return lam, p_value, surrogate_lams


# --------------------------------------------------------------------------
# 2. Bicoherence at (f0/2, f0/2)
# --------------------------------------------------------------------------

def bicoherence_at_freq(signal, fs, f1, f2, nperseg=None, noverlap=None,
                         window="hann", detrend="constant"):
    """
    Bicoherence b(f1, f2) estimated from segmented FFTs (direct analogue
    of Welch's method applied to the bispectrum), evaluated at a single
    frequency pair.

    b(f1,f2) = |<X(f1) X(f2) X*(f1+f2)>| / sqrt(<|X(f1)X(f2)|^2> <|X(f1+f2)|^2>)

    Bounded in [0, 1]. Values well above the surrogate/segment-count
    baseline indicate quadratic phase coupling between f1, f2 and f1+f2 --
    i.e. that the phase at f1+f2 is not independent of the phases at f1
    and f2, which is the frequency-domain signature expected if the f0
    component is generated as a nonlinear consequence of the f0/2 rhythm.

    Parameters
    ----------
    signal : 1D array
    fs : float
    f1, f2 : float
        Frequencies to test, e.g. f1 = f2 = f0/2 so that f1+f2 = f0.
    nperseg : int, optional
        Segment length in samples. Default: fs (i.e. 1-second segments),
        which gives 1 Hz frequency resolution -- adjust so f0/2 and f0
        fall close to an FFT bin.
    noverlap : int, optional
        Overlap in samples. Default: 50% of nperseg.

    Returns
    -------
    bic : float
        Bicoherence value at (f1, f2).
    n_segments : int
        Number of segments averaged over (relevant for interpreting
        significance -- more segments -> lower baseline under the null).
    """
    x = np.asarray(signal, dtype=float)
    if nperseg is None:
        nperseg = int(fs)
    if noverlap is None:
        noverlap = nperseg // 2

    step = nperseg - noverlap
    n_segments = 1 + (len(x) - nperseg) // step
    if n_segments < 1:
        raise ValueError("Signal too short for the requested nperseg.")

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)

    def nearest_bin(f):
        return int(np.argmin(np.abs(freqs - f)))

    i1 = nearest_bin(f1)
    i2 = nearest_bin(f2)
    i3 = nearest_bin(f1 + f2)

    if detrend == "constant":
        detrend_fn = lambda seg: seg - np.mean(seg)
    else:
        detrend_fn = lambda seg: seg

    if window == "hann":
        win = np.hanning(nperseg)
    else:
        win = np.ones(nperseg)

    num = 0.0 + 0.0j
    denom1 = 0.0
    denom2 = 0.0

    for k in range(n_segments):
        seg = x[k * step: k * step + nperseg]
        seg = detrend_fn(seg) * win
        X = np.fft.rfft(seg)

        term = X[i1] * X[i2] * np.conj(X[i3])
        num += term
        denom1 += np.abs(X[i1] * X[i2]) ** 2
        denom2 += np.abs(X[i3]) ** 2

    num /= n_segments
    denom1 /= n_segments
    denom2 /= n_segments

    bic = np.abs(num) / (np.sqrt(denom1 * denom2) + 1e-30)
    return bic, n_segments


def bicoherence_surrogate_baseline(n_segments, n_surrogates=2000, rng=None):
    """
    Analytic-free surrogate null for bicoherence: under the null of
    independent random phases, |mean of n_segments unit vectors| follows
    a known distribution. We estimate its 95th/99th percentile by Monte
    Carlo, which also serves as a rough significance threshold for
    bicoherence values obtained with the same segment count.
    """
    if rng is None:
        rng = np.random.default_rng()
    phases = rng.uniform(0, 2 * np.pi, size=(n_surrogates, n_segments))
    resultant = np.abs(np.mean(np.exp(1j * phases), axis=1))
    return {
        "p95": np.percentile(resultant, 95),
        "p99": np.percentile(resultant, 99),
        "distribution": resultant,
    }


# --------------------------------------------------------------------------
# 3. Convenience wrapper: run both tests for one (population, drive) pair
# --------------------------------------------------------------------------

def test_subharmonic_coupling(signal, fs, f0, n_surrogates=500,
                                bicoh_nperseg=None, bicoh_noverlap=None, rng=None):
    """
    Run both the 2:1 PLV test and the bicoherence test at (f0/2, f0/2)
    for a single population/drive combination.

    Parameters
    ----------
    signal : 1D array
        Population rate (or LFP-proxy) time series for one drive level.
    fs : float
    f0 : float
        Fundamental frequency (~80 Hz peak) at this drive level, e.g.
        read off from the Welch PSD used for Fig. 2A-H.
    n_surrogates : int
        Surrogate count for the PLV null.
    bicoh_nperseg : int, optional
        Segment length for the bicoherence estimate.

    Returns
    -------
    dict with keys: plv, plv_p_value, bicoherence, bicoherence_n_segments,
    bicoherence_p95, bicoherence_p99
    """
    if rng is None:
        rng = np.random.default_rng()

    lam, p_plv, _ = plv_with_surrogates(signal, fs, f0, n_surrogates=n_surrogates, rng=rng)

    bic, n_seg = bicoherence_at_freq(signal, fs, f0 / 2.0, f0 / 2.0,
                                      nperseg=bicoh_nperseg, noverlap=bicoh_noverlap)
    baseline = bicoherence_surrogate_baseline(n_seg, n_surrogates=n_surrogates, rng=rng)

    return {
        "plv": lam,
        "plv_p_value": p_plv,
        "bicoherence": bic,
        "bicoherence_n_segments": n_seg,
        "bicoherence_p95": baseline["p95"],
        "bicoherence_p99": baseline["p99"],
    }


# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Replace this block with your actual population rate signals and,
    # for each drive level, the f0 (~80 Hz peak) read off from the Welch
    # PSD already computed for Fig. 2A-H.
    #
    # Loop structure for the full figure:
    #
    # results = {}
    # for pop_name, pop_signals in population_data.items():      # 8 populations
    #     results[pop_name] = {}
    #     for drive, (signal, fs, f0) in pop_signals.items():     # drive-resolved
    #         results[pop_name][drive] = test_subharmonic_coupling(signal, fs, f0)
    #
    # Minimal synthetic sanity check: a signal with a genuine 2:1 phase
    # relationship (80 Hz clock, amplitude-modulated every other cycle)
    # should give high PLV and bicoherence; pure independent 40+80 Hz
    # tones should give low PLV/bicoherence close to the surrogate baseline.

    fs = 2000.0
    f0 = 80.0
    rng = np.random.default_rng(0)

    # Build both test signals as a concatenation of many short trials.
    # Within each trial the 80 Hz phase gets a fresh random offset (so the
    # trace as a whole is not one long deterministic tone). In the
    # "locked" case the 40 Hz phase is tied to that same offset every
    # trial (offset/2); in the "independent" case the 40 Hz phase gets
    # its OWN fresh random offset each trial. This is also the more
    # realistic setting for real data (many drive/trial repeats).
    trial_dur = 0.5
    n_trials = 400
    t_trial = np.arange(0, trial_dur, 1.0 / fs)

    locked_trials = []
    indep_trials = []
    for _ in range(n_trials):
        offset80 = rng.uniform(0, 2 * np.pi)
        phi80 = 2 * np.pi * f0 * t_trial + offset80
        locked_trials.append(np.sin(phi80) + 0.8 * np.sin(phi80 / 2.0))

        offset40_indep = rng.uniform(0, 2 * np.pi)
        phi40_indep = 2 * np.pi * 40.0 * t_trial + offset40_indep
        indep_trials.append(np.sin(phi80) + 0.8 * np.sin(phi40_indep))

    signal_locked = np.concatenate(locked_trials)
    signal_indep = np.concatenate(indep_trials)

    nperseg = t_trial.size  # one FFT segment per trial, no overlap

    for label, sig in [("locked (expect high PLV/bicoherence)", signal_locked),
                        ("independent (expect low PLV/bicoherence)", signal_indep)]:
        out = test_subharmonic_coupling(sig, fs, f0, n_surrogates=300,
                                         bicoh_nperseg=nperseg, bicoh_noverlap=0, rng=rng)
        print(f"\n{label}")
        print(f"  2:1 PLV        = {out['plv']:.3f}  (p = {out['plv_p_value']:.4f})")
        print(f"  bicoherence    = {out['bicoherence']:.3f}  "
              f"(95th pct null = {out['bicoherence_p95']:.3f}, "
              f"99th pct null = {out['bicoherence_p99']:.3f}, "
              f"n_segments = {out['bicoherence_n_segments']})")