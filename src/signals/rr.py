import numpy as np
from scipy.signal import detrend, welch

def pick_fundamental_harmonic_safe(f_hz, P, lo_hz, hi_hz):
    """
    Select the most likely fundamental frequency from a PSD while reducing harmonic errors.
    The function finds the strongest peak in the frequency band [lo_hz, hi_hz], then checks
    whether f/2 or f/3 has comparable power (which would suggest the top peak is a harmonic).
    If so, the lower candidate is preferred. Returns NaN if the band is too small.

    Args:
        f_hz (np.ndarray): Frequency axis in Hz (as returned by Welch).
        P (np.ndarray): Power spectral density values corresponding to f_hz.
        lo_hz (float): Lower frequency bound (Hz) for valid peak search.
        hi_hz (float): Upper frequency bound (Hz) for valid peak search.

    Returns:
        float: Estimated fundamental frequency in Hz, or np.nan if estimation is not possible.
    """
    band = (f_hz >= lo_hz) & (f_hz <= hi_hz)
    f = f_hz[band]
    p = P[band]
    if len(f) < 5:
        return np.nan

    k = int(np.argmax(p))
    f1 = float(f[k])

    cands = [f1]
    if f1/2 >= lo_hz: cands.append(f1/2)
    if f1/3 >= lo_hz: cands.append(f1/3)

    def local_power(freq):
        j = int(np.argmin(np.abs(f - freq)))
        j0 = max(0, j-1); j1 = min(len(p), j+2)
        return float(np.sum(p[j0:j1]))

    scores = [(local_power(fc), fc) for fc in cands]
    scores.sort(reverse=True)

    best_score, best_f = scores[0]
    for sc, fc in scores[1:]:
        if sc >= 0.85 * best_score:   
            best_f = min(best_f, fc)
    return float(best_f)


def estimate_rr_robust(sig, fs, lo_hz=0.07, hi_hz=1.0,
                       win_s=20.0, hop_s=2.0,
                       nperseg_max=512, min_valid=6):
    """
    Estimate respiratory rate (RR) robustly using windowed Welch PSD and median aggregation.
    The signal is split into overlapping time windows. For each window, the RR is estimated
    from the dominant spectral peak in the target band, using a harmonic-safe fundamental
    selection. The final RR is the median of all valid window estimates, improving stability
    against noise and outliers.

    Args:
        sig (array-like): 1D input signal containing respiratory motion measurements.
        fs (float): Sampling frequency in Hz.
        lo_hz (float): Lower frequency bound (Hz) for valid breathing rates.
        hi_hz (float): Upper frequency bound (Hz) for valid breathing rates.
        win_s (float): Window duration in seconds used for each PSD estimate.
        hop_s (float): Hop duration in seconds between consecutive windows.
        nperseg_max (int): Maximum nperseg used by Welch.
        min_valid (int): Minimum number of valid window estimates required.

    Returns:
        float: Estimated respiratory rate in breaths-per-minute (bpm), or np.nan if there
        are not enough valid windows to produce a stable estimate.
    """
    sig = np.asarray(sig, float)
    sig = sig[np.isfinite(sig)]
    if len(sig) < int(win_s * fs):
        return np.nan

    win = int(win_s * fs)
    hop = int(hop_s * fs)
    if hop < 1: hop = 1

    rr_list = []
    for start in range(0, len(sig) - win + 1, hop):
        x = detrend(sig[start:start+win])

        nperseg = min(nperseg_max, max(64, win // 4))
        noverlap = nperseg // 2
        nfft = int(max(16384, 2 ** int(np.ceil(np.log2(nperseg * 8)))))

        f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        f0 = pick_fundamental_harmonic_safe(f, Pxx, lo_hz, hi_hz)
        if np.isfinite(f0):
            rr_list.append(60.0 * f0)

    if len(rr_list) < min_valid:
        return np.nan
    return float(np.median(rr_list))
