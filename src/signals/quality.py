import numpy as np
from scipy.signal import coherence
from src.signals.preprocess import fill_nans_linear, zscore

def best_lag_seconds(x_ref_z, x_vid_z, fs, max_lag_s=5.0):
    """
    Estimate the relative time lag between two signals using cross-correlation.
    Computes the cross-correlation of the two (z-scored) signals, restricts the 
    search to [-max_lag_s, +mag_lag_s], and returns the lag that maximizes correlation. 

    Args:
        x_ref_z (array-like): Reference signal.
        x_vid_z (array-like): Video-derived signal.
        fs (float): Sampling frequency in Hz.
        max_lag_s (float): Maximum absolute lag to consider in seconds.

    Returns:
        float: Estimated lag in seconds (positive means the video signal is delayed relative
        to the reference, given the chosen correlation ordering).
    """
    x_ref_z = np.asarray(x_ref_z, float)
    x_vid_z = np.asarray(x_vid_z, float)
    n = min(len(x_ref_z), len(x_vid_z))
    x_ref_z = x_ref_z[:n]
    x_vid_z = x_vid_z[:n]

    c = np.correlate(x_ref_z, x_vid_z, mode="full")
    lags = np.arange(-n + 1, n)

    max_lag = int(round(max_lag_s * fs))
    keep = (lags >= -max_lag) & (lags <= max_lag)
    c_keep = c[keep]
    l_keep = lags[keep]

    k = int(np.argmax(c_keep))
    return float(l_keep[k] / fs)

def coherence_band_stats(x_ref_z, x_vid_z, fs, lo_hz=0.07, hi_hz=1.0):
    """
    Compute magnitude-squared coherence and summarize it inside a frequency band.
    Uses segmented coherence estimation (Welch-style) to avoid degenerate behavior where
    using a single segment (nperseg == n) can yield coherence values close to 1 across
    all frequencies. Returns the full coherence spectrum, plus the mean and peak coherence
    within [lo_hz, hi_hz].

    Args:
        x_ref_z (array-like): Reference signal.
        x_vid_z (array-like): Video-derived signal.
        fs (float): Sampling frequency in Hz.
        lo_hz (float): Lower frequency bound (Hz) for band summary.
        hi_hz (float): Upper frequency bound (Hz) for band summary.

    Returns:
        f (np.ndarray): Frequency axis in Hz.
        Cxy (np.ndarray) : Coherence values for each frequency.
        mean (float): Mean coherence within the band [lo_hz, hi_hz], or np.nan if invalid.
        peak (float): Peak coherence within the band [lo_hz, hi_hz], or np.nan if invalid.

    """
    x_ref_z = np.asarray(x_ref_z, float)
    x_vid_z = np.asarray(x_vid_z, float)
    n = min(len(x_ref_z), len(x_vid_z))
    x_ref_z = x_ref_z[:n]
    x_vid_z = x_vid_z[:n]

    nyq = 0.5 * fs
    hi_hz = min(hi_hz, 0.99 * nyq)
    if hi_hz <= lo_hz or n < 64:
        return np.array([]), np.array([]), np.nan, np.nan
        
    nperseg = int(np.clip(n // 8, 64, 512))
    nperseg = min(nperseg, max(64, n // 2))
    noverlap = nperseg // 2

    f, Cxy = coherence(x_ref_z, x_vid_z, fs=fs, nperseg=nperseg, noverlap=noverlap)

    band = (f >= lo_hz) & (f <= hi_hz)
    if not np.any(band):
        return f, Cxy, np.nan, np.nan

    c_band = Cxy[band]

    mean = float(np.mean(c_band))
    peak = float(np.max(c_band))
    return f, Cxy, mean, peak
    
def signal_quality_metrics(ref, vid, fs, lo_hz=0.07, hi_hz=1.0, max_lag_s=5.0, forced_lag_s=None):
    """
    Compute alignment and similarity metrics between a reference and a video-derived signal.

    This function cleans NaNs, z-scores both signals, computes a zero-lag Pearson correlation,
    estimates a best time lag via cross-correlation (unless `forced_lag_s` is provided), and
    computes coherence statistics (mean and peak) inside a target frequency band.

    Args:
        ref (array-like): Reference signal (e.g., ground-truth respiration waveform).
        vid (array-like): Video-derived signal to compare against the reference.
        fs (float): Sampling frequency in Hz.
        lo_hz (float): Lower frequency bound (Hz) for coherence band summary.
        hi_hz (float): Upper frequency bound (Hz) for coherence band summary.
        max_lag_s (float): Maximum absolute lag to search for via cross-correlation.
        forced_lag_s (float | None): If provided, use this lag (seconds) instead of estimating it.

    Returns:
        dict: Dictionary containing quality metrics and intermediate signals:
            - "corr_z" (float): Zero-lag correlation between z-scored signals (or np.nan).
            - "best_lag_s" (float): Estimated (or forced) best lag in seconds.
            - "coh_mean_band" (float): Mean coherence within [lo_hz, hi_hz].
            - "coh_peak_band" (float): Peak coherence within [lo_hz, hi_hz].
            - "coh_f" (np.ndarray): Frequency axis for coherence spectrum.
            - "coh_Cxy" (np.ndarray): Coherence spectrum values.
            - "ref_z" (np.ndarray): Z-scored reference signal (aligned length).
            - "vid_z" (np.ndarray): Z-scored video signal (aligned length).

    """
    ref = fill_nans_linear(ref)
    vid = fill_nans_linear(vid)
    n = min(len(ref), len(vid))
    ref = ref[:n]
    vid = vid[:n]

    ref_z = zscore(ref)
    vid_z = zscore(vid)

    corr_z = np.nan
    if np.std(ref_z) > 1e-12 and np.std(vid_z) > 1e-12:
        corr_z = float(np.corrcoef(ref_z, vid_z)[0, 1])
    if forced_lag_s is None:
        if abs(corr_z) < 0.1: 
            lag_s = 0
        else:
            lag_s = best_lag_seconds(ref_z, vid_z, fs=fs, max_lag_s=max_lag_s)
    else:
        lag_s = float(forced_lag_s)
    f, Cxy, coh_mean, coh_peak = coherence_band_stats(ref_z, vid_z, fs=fs, lo_hz=lo_hz, hi_hz=hi_hz)

    return {
        "corr_z": corr_z,
        "best_lag_s": lag_s,
        "coh_mean_band": coh_mean,
        "coh_peak_band": coh_peak,
        "coh_f": f,
        "coh_Cxy": Cxy,
        "ref_z": ref_z,
        "vid_z": vid_z,
    }

def corrcoef_safe(a, b):
    """
    Compute Pearson correlation safely, handling NaNs and near-constant inputs.

    The function keeps only finite (a, b) pairs, returns np.nan if there are too few samples
    or if either signal has near-zero variance, otherwise returns the standard Pearson
    correlation coefficient.

    Args:
        a (array): First input signal.
        b (array): Second input signal.

    Returns:
        corr (float): Pearson correlation coefficient in [-1, 1], or np.nan if undefined.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]; b = b[m]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    corr = float(np.corrcoef(a, b)[0, 1])
    return corr