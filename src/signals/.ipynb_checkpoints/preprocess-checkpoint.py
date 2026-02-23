import numpy as np
from scipy.signal import butter, filtfilt

def bandpass(sig, fs, lo_hz=0.08, hi_hz=1.0, order=2):
    """
    Band-pass filter a 1D signal with a Butterworth filter.
    The cutoff frequencies are clamped to valid ranges (below Nyquist and above ~0 Hz).
    If the requested band is invalid (hi_hz <= lo_hz), the input signal is returned unchanged.

    Args:
        sig (array-like): 1D input signal to filter.
        fs (float): Sampling frequency in Hz.
        lo_hz (float): Low cutoff frequency in Hz.
        hi_hz (float): High cutoff frequency in Hz.
        order (int): Butterworth filter order.

    Returns:
        filtered (np.ndarray): Filtered signal. If the band is invalid, returns
        the input signal unchanged.
    """
    nyq = 0.5 * fs
    hi_hz = min(hi_hz, 0.99 * nyq)
    lo_hz = max(lo_hz, 0.001)
    if hi_hz <= lo_hz:
        return sig  # fallback: no filtering
    b, a = butter(N=order, Wn=[lo_hz/nyq, hi_hz/nyq], btype="bandpass")
    filtered = filtfilt(b, a, sig)
    return filtered

def interp_extrapolate(x_new, x, y):
    """
    Resample y(x) onto a new grid using 1D linear interpolation.

    Args:
        x_new (array-like): New x-coordinates where the signal should be evaluated.
        x (array-like): Original x-coordinates of the input samples.
        y (array-like): Original y-values sampled at `x`.

    Returns:
        out (np.ndarray): Interpolated (and edge-extrapolated) values y(x_new).
    """
    x_new, x, y = np.asarray(x_new), np.asarray(x), np.asarray(y)
    out = np.interp(x_new, x, y)
    return out

def fill_nans_linear(x):
    """
    Fill NaNs/infs in a 1D array using linear interpolation over valid samples.

    Args:
        x (array-like): 1D input signal containing possible NaNs/Infs.

    Returns:
        x (np.ndarray): Cleaned signal with invalid values filled. If all samples are invalid,
        returns zeros.
    """
    x = np.asarray(x, float).copy()
    good = np.isfinite(x)
    if good.all():
        return x
    if not np.any(good):
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x

def zscore(x, eps=1e-12):
    """
    Standardize a 1D signal to zero mean and unit variance (z-score normalization).

    If the standard deviation is too small (near-constant signal), returns zeros
    to avoid division by tiny values.

    Args:
        x (array-like): 1D input signal to normalize.
        eps (float): Minimum standard deviation threshold.

    Returns:
        zs (np.ndarray): Z-scored signal with mean ~0 and std ~1. If the input is near-constant,
        returns an array of zeros.
    """
    x = np.asarray(x, float)
    mu = np.mean(x)
    sd = np.std(x)
    if sd < eps:
        return x * 0.0
    zs = (x - mu) / sd
    return zs

def align_video_waveform_to_ref(df_run_time_s, t_video_s, sig_video, lag_s):
    """
    Time-shift and resample a video-derived signal onto a reference time grid.

    The video waveform is sampled at times (df_run_time_s + lag_s) so that it aligns
    with the reference timeline. Values outside the video time range are extrapolated
    using the endpoint samples to avoid NaNs.

    Args:
        df_run_time_s (array-like): Reference time grid (seconds) to resample onto.
        t_video_s (array-like): Video signal time stamps (seconds).
        sig_video (array-like): Video-derived waveform sampled at t_video_s.
        lag_s (float): Time shift applied to align video with reference (seconds).

    Returns:
        y (np.ndarray): Video waveform resampled onto df_run_time_s after applying lag_s.
    """
    df_run_time_s = np.asarray(df_run_time_s, float)
    t_video_s = np.asarray(t_video_s, float)
    sig_video = np.asarray(sig_video, float)

    y = np.interp(df_run_time_s + lag_s, t_video_s, sig_video, left=sig_video[0], right=sig_video[-1])
    return y
