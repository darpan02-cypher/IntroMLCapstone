
"""features.py
Helper functions for feature extraction from wearable signals.
Fill in or adapt functions based on dataset file formats (WESAD uses per-subject files).
"""
import numpy as np
import pandas as pd
from scipy.signal import welch

def rolling_features(series, window=30, fs=1):
    """Compute simple rolling features for a 1D signal series.
    Args:
        series: pd.Series or 1D array
        window: window size in seconds (assumes fs samples per second)
        fs: sampling frequency (samples per second)
    Returns:
        DataFrame with mean, std, min, max for each window (non-overlapping)
    """
    step = window * fs
    n = len(series)
    features = []
    for i in range(0, n, step):
        win = series[i:i+step]
        if len(win) == 0:
            break
        features.append({
            'mean': np.nanmean(win),
            'std': np.nanstd(win),
            'min': np.nanmin(win),
            'max': np.nanmax(win),
            'median': np.nanmedian(win)
        })
    return pd.DataFrame(features)

def compute_hrv_metrics(rr_intervals_ms):
    """Compute simple HRV metrics from R-R intervals in milliseconds.
    Args:
        rr_intervals_ms: array-like of successive RR intervals in ms
    Returns: dict of HRV features (RMSSD, SDNN, meanHR)
    """
    rr = np.array(rr_intervals_ms)
    if len(rr) < 2:
        return {'rmssd': np.nan, 'sdnn': np.nan, 'mean_hr': np.nan}
    diff = np.diff(rr)
    rmssd = np.sqrt(np.nanmean(diff**2))
    sdnn = np.nanstd(rr)
    mean_hr = 60000.0 / np.mean(rr) if np.mean(rr) > 0 else np.nan
    return {'rmssd': rmssd, 'sdnn': sdnn, 'mean_hr': mean_hr}

def psd_bandpower(signal, fs=4, band=(0.04,0.15)):
    """Estimate band power using Welch's method (simple)."""
    f, Pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    mask = (f >= band[0]) & (f <= band[1])
    return np.trapz(Pxx[mask], f[mask]) if np.any(mask) else np.nan
