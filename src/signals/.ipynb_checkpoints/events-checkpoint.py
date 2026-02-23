import numpy as np
from scipy.signal import find_peaks

def extract_breath_amplitudes(sig, fps, prominence_factor=0.3, hi_hz=1.0):
    """
    Extract per-breath amplitudes and their timestamps.

    Detects inhale/exhale cycles by finding peaks and valleys in the signal, then 
    pairs each peak with the most recent preceding valley. The amplitude is computed 
    as (peak - valley), and the timestamp is taken as the midpoint time between them.

    Args:
        sig (array-like): 1D input respiratory signal (waveform).
        fps (float): Sampling frequency of the signal in Hz (frames per second).
        prominence_factor (float): Peak/valley prominence threshold expressed as a fraction
            of the signal standard deviation.
        hi_hz (float): Maximum expected breathing frequency in Hz, used to set the minimum
            peak distance constraint.

    Returns:
        tuple:
            - np.ndarray: Breath timestamps in seconds (midpoint between valley and peak).
            - np.ndarray: Breath amplitudes (peak - valley) for each detected breath.
    """
    sig = sig - np.mean(sig)

    min_period_s = 1.0 / hi_hz  
    distance = int(0.6 * min_period_s * fps)
    distance = max(distance, 2)

    peaks, _ = find_peaks(sig, distance=distance, prominence=np.std(sig)*prominence_factor)
    
    valleys, _ = find_peaks(-sig, distance=distance, prominence=np.std(sig)*prominence_factor)
    
    if len(peaks) < 2 or len(valleys) < 2:
        return np.array([]), np.array([])

    t_amp = []
    amp_vals = []
    
    for p in peaks:
        past_valleys = valleys[valleys < p]
        if len(past_valleys) == 0:
            continue
            
        v = past_valleys[-1]
        
        height = sig[p] - sig[v]
        
        t_sec = (p + v) / 2.0 / fps
        
        t_amp.append(t_sec)
        amp_vals.append(height)
        
    return np.array(t_amp), np.array(amp_vals)

def breath_times_from_rf(time_s, rf_bpm):
    """
    Convert an instantaneous respiratory rate trace into discrete breath timestamps.

    Treats rf_bpm(t) as a continuous breathing rate, converts it to Hz, integrates it 
    over time to obtain cumulative breath count, then returns the times when the cumulative 
    count crosses each integer breath number.

    Args:
        time_s (array-like): Time axis in seconds.
        rf_bpm (array-like): Instantaneous respiratory rate in breaths-per-minute (bpm),
            sampled at `time_s`.

    Returns:
        out (np.ndarray): Array of breath timestamps in seconds. Returns an empty array if the
        input is too short or does not span at least one full breath.

    """
    t = np.asarray(time_s, float)
    rf = np.asarray(rf_bpm, float)
    m = np.isfinite(t) & np.isfinite(rf)
    t = t[m]; rf = rf[m]
    if len(t) < 5:
        return np.array([])

    r_hz = rf / 60.0

    cum = np.zeros_like(t)
    dt_ = np.diff(t)
    cum[1:] = np.cumsum(0.5 * (r_hz[1:] + r_hz[:-1]) * dt_)

    n0 = int(np.ceil(cum[0]))
    n1 = int(np.floor(cum[-1]))
    if n1 <= n0:
        return np.array([])

    targets = np.arange(n0, n1 + 1, dtype=float)

    bt = []
    j = 0
    for target in targets:
        while j < len(cum) - 1 and cum[j+1] < target:
            j += 1
        if j >= len(cum) - 1:
            break
        c0, c1 = cum[j], cum[j+1]
        if c1 <= c0:
            continue
        a = (target - c0) / (c1 - c0)
        t_cross = t[j] + a * (t[j+1] - t[j])
        bt.append(t_cross)

    out = np.asarray(bt, float)
    return out

def match_events_nearest(t_ref, t_vid, y_vid, max_dt=1.0):
    """
    Match each reference event time to the nearest video event value within a tolerance.

    For every timestamp in t_ref, finds the closest time in sorted t_vid. If the nearest 
    event is farther than max_dt seconds away, the output for that reference time is NaN.

    Args:
        t_ref (array-like): Reference event timestamps in seconds.
        t_vid (array-like): Video event timestamps in seconds (must be sorted ascending).
        y_vid (array-like): Video event values corresponding to `t_vid` (same length).
        max_dt (float): Maximum allowed time difference (seconds) for a valid match
            (default: 1.0).

    Returns:
        out (np.ndarray): Array of matched video values aligned to `t_ref`. Entries are NaN when
        no acceptable match is found.
    """
    t_ref = np.asarray(t_ref, float)
    t_vid = np.asarray(t_vid, float)
    y_vid = np.asarray(y_vid, float)

    if len(t_ref) == 0 or len(t_vid) == 0:
        return np.full(len(t_ref), np.nan)

    out = np.full(len(t_ref), np.nan)
    idx = np.searchsorted(t_vid, t_ref)

    for i, t in enumerate(t_ref):
        cands = []
        if 0 <= idx[i] < len(t_vid):
            cands.append(idx[i])
        if 0 <= idx[i]-1 < len(t_vid):
            cands.append(idx[i]-1)

        best_j = None
        best_dt = None
        for j in cands:
            d = abs(t_vid[j] - t)
            if best_dt is None or d < best_dt:
                best_dt = d
                best_j = j

        if best_j is not None and best_dt <= max_dt:
            out[i] = y_vid[best_j]

    return out

def nearest_dt(t_ref, t_evt):
    """
    Compute the time distance from each reference time to the nearest event time.

    For each entry in t_ref, finds the closest timestamp in sorted t_evt and returns
    the absolute difference in seconds. If t_evt is empty, returns all-NaN.

    Args:
        t_ref (array-like): Reference timestamps in seconds.
        t_evt (array-like): Event timestamps in seconds (must be sorted ascending).

    Returns:
        out (np.ndarray): Absolute time difference (seconds) from each entry in t_ref to the
        nearest event in t_evt, or NaNs if t_evt is empty.
    """
    t_ref = np.asarray(t_ref, float)
    t_evt = np.asarray(t_evt, float)
    if len(t_evt) == 0:
        return np.full(len(t_ref), np.nan)
    idx = np.searchsorted(t_evt, t_ref)
    out = np.full(len(t_ref), np.nan)
    for i, t in enumerate(t_ref):
        cands = []
        if 0 <= idx[i] < len(t_evt): cands.append(idx[i])
        if 0 <= idx[i]-1 < len(t_evt): cands.append(idx[i]-1)
        out[i] = np.min([abs(t_evt[j]-t) for j in cands]) if cands else np.nan
    return out