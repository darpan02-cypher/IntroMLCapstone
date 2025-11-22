"""features.py
Comprehensive feature extraction for WESAD physiological signals.
Supports mindfulness prediction via stress proxy mapping.

Features extracted:
- HRV (time-domain, frequency-domain)
- EDA (phasic/tonic decomposition, SCR metrics)
- Respiratory (rate, depth, variability)
- Temperature (mean, slope, variability)
- Activity (accelerometer magnitude)
- Statistical (rolling windows)
"""

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.signal import welch, find_peaks
from scipy.stats import skew, kurtosis
import warnings

# Optional imports with fallbacks
try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False
    warnings.warn("neurokit2 not installed. Some HRV features will use fallback methods.")

try:
    import biosppy
    from biosppy.signals import ecg as ecg_biosppy
    HAS_BIOSPPY = True
except ImportError:
    HAS_BIOSPPY = False

try:
    import cvxopt
    HAS_CVXOPT = True
except ImportError:
    HAS_CVXOPT = False
    warnings.warn("cvxopt not installed. EDA decomposition will use simplified method.")


# ============================================================================
# TARGET VARIABLE MAPPING
# ============================================================================

def map_labels_to_mindfulness(labels):
    """
    Convert WESAD labels to continuous mindfulness index.
    
    Mapping:
        0 (Transient) -> NaN (excluded)
        1 (Baseline) -> 1.0 (high mindfulness)
        2 (Stress/TSST) -> 0.0 (low mindfulness)
        3 (Amusement) -> 0.7 (medium-high mindfulness)
        4 (Meditation) -> 0.9 (high mindfulness)
    
    Args:
        labels: array of WESAD labels
    
    Returns:
        array of mindfulness scores (0-1 scale)
    """
    label_map = {
        0: np.nan,  # Transient - exclude
        1: 1.0,     # Baseline
        2: 0.0,     # Stress
        3: 0.7,     # Amusement
        4: 0.9,     # Meditation
    }
    
    mindfulness = np.array([label_map.get(int(l), np.nan) for l in labels])
    return mindfulness


# ============================================================================
# HRV FEATURES (ECG-based)
# ============================================================================

def extract_rpeaks(ecg_signal, fs=700):
    """
    Detect R-peaks in ECG signal.
    
    Args:
        ecg_signal: 1D array of ECG values
        fs: sampling frequency (Hz)
    
    Returns:
        array of R-peak indices
    """
    if HAS_NEUROKIT:
        # Use NeuroKit2 for robust peak detection
        _, info = nk.ecg_peaks(ecg_signal, sampling_rate=fs)
        return info['ECG_R_Peaks']
    elif HAS_BIOSPPY:
        # Use BioSPPy as fallback
        out = ecg_biosppy.ecg(ecg_signal, sampling_rate=fs, show=False)
        return out['rpeaks']
    else:
        # Simple peak detection fallback
        # Bandpass filter ECG
        sos = sp_signal.butter(4, [0.5, 40], btype='band', fs=fs, output='sos')
        ecg_filtered = sp_signal.sosfilt(sos, ecg_signal)
        
        # Find peaks
        peaks, _ = find_peaks(ecg_filtered, distance=int(0.6*fs), prominence=0.5*np.std(ecg_filtered))
        return peaks


def compute_hrv_time_domain(rr_intervals_ms):
    """
    Compute time-domain HRV features.
    
    Args:
        rr_intervals_ms: array of RR intervals in milliseconds
    
    Returns:
        dict of time-domain HRV features
    """
    rr = np.array(rr_intervals_ms)
    
    if len(rr) < 2:
        return {
            'rmssd': np.nan,
            'sdnn': np.nan,
            'pnn50': np.nan,
            'mean_hr': np.nan,
            'std_hr': np.nan,
            'min_hr': np.nan,
            'max_hr': np.nan,
            'hr_range': np.nan
        }
    
    # RR interval differences
    diff_rr = np.diff(rr)
    
    # RMSSD: Root mean square of successive differences
    rmssd = np.sqrt(np.mean(diff_rr**2))
    
    # SDNN: Standard deviation of NN intervals
    sdnn = np.std(rr, ddof=1)
    
    # pNN50: Percentage of successive RR intervals differing by > 50ms
    pnn50 = 100 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr)
    
    # Heart rate statistics
    hr = 60000.0 / rr  # Convert RR (ms) to HR (bpm)
    mean_hr = np.mean(hr)
    std_hr = np.std(hr, ddof=1)
    min_hr = np.min(hr)
    max_hr = np.max(hr)
    hr_range = max_hr - min_hr
    
    return {
        'rmssd': rmssd,
        'sdnn': sdnn,
        'pnn50': pnn50,
        'mean_hr': mean_hr,
        'std_hr': std_hr,
        'min_hr': min_hr,
        'max_hr': max_hr,
        'hr_range': hr_range
    }


def compute_hrv_frequency_domain(rr_intervals_ms, fs_rr=4):
    """
    Compute frequency-domain HRV features using Welch's method.
    
    Args:
        rr_intervals_ms: array of RR intervals in milliseconds
        fs_rr: sampling frequency of RR intervals (Hz) - typically 4Hz after resampling
    
    Returns:
        dict of frequency-domain HRV features
    """
    rr = np.array(rr_intervals_ms)
    
    if len(rr) < 10:
        return {
            'lf_power': np.nan,
            'hf_power': np.nan,
            'lf_hf_ratio': np.nan,
            'total_power': np.nan,
            'lf_norm': np.nan,
            'hf_norm': np.nan
        }
    
    # Resample RR intervals to uniform sampling
    # Create time vector
    time_rr = np.cumsum(rr) / 1000.0  # Convert to seconds
    time_uniform = np.arange(0, time_rr[-1], 1/fs_rr)
    
    # Interpolate to uniform sampling
    rr_uniform = np.interp(time_uniform, time_rr, rr)
    
    # Compute PSD using Welch's method
    nperseg = min(256, len(rr_uniform))
    f, psd = welch(rr_uniform, fs=fs_rr, nperseg=nperseg)
    
    # Define frequency bands
    vlf_band = (0.0, 0.04)   # Very low frequency
    lf_band = (0.04, 0.15)   # Low frequency (sympathetic + parasympathetic)
    hf_band = (0.15, 0.4)    # High frequency (parasympathetic)
    
    # Calculate band powers
    lf_power = np.trapz(psd[(f >= lf_band[0]) & (f < lf_band[1])], 
                        f[(f >= lf_band[0]) & (f < lf_band[1])])
    hf_power = np.trapz(psd[(f >= hf_band[0]) & (f < hf_band[1])], 
                        f[(f >= hf_band[0]) & (f < hf_band[1])])
    total_power = np.trapz(psd[f < 0.4], f[f < 0.4])
    
    # LF/HF ratio
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan
    
    # Normalized powers
    lf_norm = lf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else np.nan
    hf_norm = hf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else np.nan
    
    return {
        'lf_power': lf_power,
        'hf_power': hf_power,
        'lf_hf_ratio': lf_hf_ratio,
        'total_power': total_power,
        'lf_norm': lf_norm,
        'hf_norm': hf_norm
    }


def extract_hrv_features(ecg_signal, fs=700):
    """
    Extract comprehensive HRV features from ECG signal.
    
    Args:
        ecg_signal: 1D array of ECG values
        fs: sampling frequency (Hz)
    
    Returns:
        dict of HRV features (time + frequency domain)
    """
    # Detect R-peaks
    rpeaks = extract_rpeaks(ecg_signal, fs)
    
    if len(rpeaks) < 2:
        # Return NaN features if insufficient peaks
        features = {}
        features.update(compute_hrv_time_domain([]))
        features.update(compute_hrv_frequency_domain([]))
        return features
    
    # Calculate RR intervals in milliseconds
    rr_intervals = np.diff(rpeaks) / fs * 1000.0
    
    # Filter physiologically implausible RR intervals (300-2000 ms = 30-200 bpm)
    valid_mask = (rr_intervals >= 300) & (rr_intervals <= 2000)
    rr_intervals = rr_intervals[valid_mask]
    
    if len(rr_intervals) < 2:
        features = {}
        features.update(compute_hrv_time_domain([]))
        features.update(compute_hrv_frequency_domain([]))
        return features
    
    # Extract time-domain features
    features = compute_hrv_time_domain(rr_intervals)
    
    # Extract frequency-domain features
    features.update(compute_hrv_frequency_domain(rr_intervals))
    
    return features


# ============================================================================
# EDA FEATURES
# ============================================================================

def decompose_eda_simple(eda_signal, fs=700):
    """
    Simple EDA decomposition into tonic (SCL) and phasic (SCR) components.
    Uses low-pass filtering approach.
    
    Args:
        eda_signal: 1D array of EDA values
        fs: sampling frequency (Hz)
    
    Returns:
        tuple of (tonic, phasic) components
    """
    # Low-pass filter for tonic component (cutoff ~0.05 Hz)
    sos = sp_signal.butter(4, 0.05, btype='low', fs=fs, output='sos')
    tonic = sp_signal.sosfilt(sos, eda_signal)
    
    # Phasic = original - tonic
    phasic = eda_signal - tonic
    
    return tonic, phasic


def extract_eda_features(eda_signal, fs=700):
    """
    Extract EDA features including phasic/tonic decomposition.
    
    Args:
        eda_signal: 1D array of EDA values (microsiemens)
        fs: sampling frequency (Hz)
    
    Returns:
        dict of EDA features
    """
    # Decompose into tonic and phasic
    tonic, phasic = decompose_eda_simple(eda_signal, fs)
    
    # Detect SCR peaks in phasic component
    # Threshold: peaks above 0.01 microsiemens
    threshold = 0.01
    min_distance = int(1.0 * fs)  # Minimum 1 second between peaks
    
    peaks, properties = find_peaks(phasic, height=threshold, distance=min_distance)
    
    # SCR metrics
    scr_count = len(peaks)
    scr_mean_amplitude = np.mean(properties['peak_heights']) if len(peaks) > 0 else 0.0
    scr_max_amplitude = np.max(properties['peak_heights']) if len(peaks) > 0 else 0.0
    
    # Statistical features
    features = {
        # Tonic (SCL) features
        'scl_mean': np.mean(tonic),
        'scl_std': np.std(tonic),
        'scl_min': np.min(tonic),
        'scl_max': np.max(tonic),
        'scl_range': np.max(tonic) - np.min(tonic),
        
        # Phasic (SCR) features
        'scr_count': scr_count,
        'scr_mean_amplitude': scr_mean_amplitude,
        'scr_max_amplitude': scr_max_amplitude,
        'scr_rate': scr_count / (len(eda_signal) / fs / 60),  # SCRs per minute
        
        # Overall EDA statistics
        'eda_mean': np.mean(eda_signal),
        'eda_std': np.std(eda_signal),
        'eda_min': np.min(eda_signal),
        'eda_max': np.max(eda_signal),
        'eda_range': np.max(eda_signal) - np.min(eda_signal),
        
        # Slope (linear trend)
        'eda_slope': np.polyfit(np.arange(len(eda_signal)), eda_signal, 1)[0]
    }
    
    return features


# ============================================================================
# RESPIRATORY FEATURES
# ============================================================================

def extract_respiratory_features(resp_signal, fs=700):
    """
    Extract respiratory features (rate, depth, variability).
    
    Args:
        resp_signal: 1D array of respiration values
        fs: sampling frequency (Hz)
    
    Returns:
        dict of respiratory features
    """
    # Bandpass filter to isolate breathing (0.1-0.5 Hz = 6-30 breaths/min)
    sos = sp_signal.butter(4, [0.1, 0.5], btype='band', fs=fs, output='sos')
    resp_filtered = sp_signal.sosfilt(sos, resp_signal)
    
    # Detect breathing peaks (inhalation peaks)
    min_distance = int(1.5 * fs)  # Minimum 1.5 seconds between breaths
    peaks, properties = find_peaks(resp_filtered, distance=min_distance)
    
    # Detect troughs (exhalation)
    troughs, _ = find_peaks(-resp_filtered, distance=min_distance)
    
    if len(peaks) < 2:
        return {
            'resp_rate': np.nan,
            'resp_depth_mean': np.nan,
            'resp_depth_std': np.nan,
            'resp_variability': np.nan,
            'ie_ratio': np.nan
        }
    
    # Respiratory rate (breaths per minute)
    breath_intervals = np.diff(peaks) / fs  # in seconds
    resp_rate = 60.0 / np.mean(breath_intervals)
    
    # Respiratory depth (amplitude)
    # Get peak heights - either from properties or calculate manually
    if 'peak_heights' in properties and len(properties['peak_heights']) > 0:
        amplitudes = properties['peak_heights']
    else:
        # Calculate peak heights manually
        amplitudes = resp_filtered[peaks]
    
    resp_depth_mean = np.mean(amplitudes)
    resp_depth_std = np.std(amplitudes)
    
    # Respiratory variability (std of inter-breath intervals)
    resp_variability = np.std(breath_intervals)
    
    # Inhalation/Exhalation ratio (simplified)
    ie_ratio = len(peaks) / len(troughs) if len(troughs) > 0 else np.nan
    
    features = {
        'resp_rate': resp_rate,
        'resp_depth_mean': resp_depth_mean,
        'resp_depth_std': resp_depth_std,
        'resp_variability': resp_variability,
        'ie_ratio': ie_ratio
    }
    
    return features


# ============================================================================
# TEMPERATURE FEATURES
# ============================================================================

def extract_temperature_features(temp_signal, fs=700):
    """
    Extract temperature features.
    
    Args:
        temp_signal: 1D array of temperature values (Celsius)
        fs: sampling frequency (Hz)
    
    Returns:
        dict of temperature features
    """
    # Statistical features
    temp_mean = np.mean(temp_signal)
    temp_std = np.std(temp_signal)
    temp_min = np.min(temp_signal)
    temp_max = np.max(temp_signal)
    temp_range = temp_max - temp_min
    
    # Slope (linear trend over window)
    time = np.arange(len(temp_signal)) / fs
    slope, _ = np.polyfit(time, temp_signal, 1)
    
    features = {
        'temp_mean': temp_mean,
        'temp_std': temp_std,
        'temp_min': temp_min,
        'temp_max': temp_max,
        'temp_range': temp_range,
        'temp_slope': slope
    }
    
    return features


# ============================================================================
# ACTIVITY/ACCELEROMETER FEATURES
# ============================================================================

def extract_activity_features(acc_signal, fs=700):
    """
    Extract activity features from 3-axis accelerometer.
    
    Args:
        acc_signal: 2D array of shape (n_samples, 3) for x, y, z axes
        fs: sampling frequency (Hz)
    
    Returns:
        dict of activity features
    """
    if acc_signal.ndim == 1:
        # Single axis - convert to 2D
        acc_signal = acc_signal.reshape(-1, 1)
    
    # Compute magnitude of acceleration vector
    if acc_signal.shape[1] == 3:
        magnitude = np.sqrt(np.sum(acc_signal**2, axis=1))
    else:
        magnitude = np.abs(acc_signal.flatten())
    
    # Statistical features
    activity_mean = np.mean(magnitude)
    activity_std = np.std(magnitude)
    activity_max = np.max(magnitude)
    
    # Activity level (percentage of high-activity samples)
    # High activity = magnitude > mean + 1*std
    threshold = activity_mean + activity_std
    activity_level = 100 * np.sum(magnitude > threshold) / len(magnitude)
    
    # Posture stability (lower variance = more stable)
    posture_stability = 1.0 / (1.0 + activity_std)  # Normalized inverse of std
    
    features = {
        'activity_mean': activity_mean,
        'activity_std': activity_std,
        'activity_max': activity_max,
        'activity_level': activity_level,
        'posture_stability': posture_stability
    }
    
    return features


# ============================================================================
# STATISTICAL FEATURES
# ============================================================================

def compute_statistical_features(signal, prefix='signal'):
    """
    Compute comprehensive statistical features for a signal.
    
    Args:
        signal: 1D array
        prefix: string prefix for feature names
    
    Returns:
        dict of statistical features
    """
    features = {
        f'{prefix}_mean': np.mean(signal),
        f'{prefix}_std': np.std(signal),
        f'{prefix}_min': np.min(signal),
        f'{prefix}_max': np.max(signal),
        f'{prefix}_median': np.median(signal),
        f'{prefix}_range': np.max(signal) - np.min(signal),
        f'{prefix}_q25': np.percentile(signal, 25),
        f'{prefix}_q75': np.percentile(signal, 75),
        f'{prefix}_iqr': np.percentile(signal, 75) - np.percentile(signal, 25),
        f'{prefix}_skewness': skew(signal),
        f'{prefix}_kurtosis': kurtosis(signal)
    }
    
    return features


# ============================================================================
# WINDOW-BASED FEATURE EXTRACTION
# ============================================================================

def extract_window_features(chest_signals, window_start, window_end, fs_dict=None):
    """
    Extract all features from a time window of chest signals.
    
    Args:
        chest_signals: dict with keys 'ECG', 'EDA', 'Resp', 'Temp', 'ACC'
        window_start: start index of window
        window_end: end index of window
        fs_dict: dict of sampling frequencies for each signal (default: all 700 Hz)
    
    Returns:
        dict of all extracted features
    """
    if fs_dict is None:
        fs_dict = {'ECG': 700, 'EDA': 700, 'Resp': 700, 'Temp': 700, 'ACC': 700}
    
    features = {}
    
    # Extract ECG/HRV features
    if 'ECG' in chest_signals:
        ecg_window = chest_signals['ECG'][window_start:window_end].flatten()
        hrv_features = extract_hrv_features(ecg_window, fs=fs_dict.get('ECG', 700))
        features.update(hrv_features)
    
    # Extract EDA features
    if 'EDA' in chest_signals:
        eda_window = chest_signals['EDA'][window_start:window_end].flatten()
        eda_features = extract_eda_features(eda_window, fs=fs_dict.get('EDA', 700))
        features.update(eda_features)
    
    # Extract respiratory features
    if 'Resp' in chest_signals:
        resp_window = chest_signals['Resp'][window_start:window_end].flatten()
        resp_features = extract_respiratory_features(resp_window, fs=fs_dict.get('Resp', 700))
        features.update(resp_features)
    
    # Extract temperature features
    if 'Temp' in chest_signals:
        temp_window = chest_signals['Temp'][window_start:window_end].flatten()
        temp_features = extract_temperature_features(temp_window, fs=fs_dict.get('Temp', 700))
        features.update(temp_features)
    
    # Extract activity features
    if 'ACC' in chest_signals:
        acc_window = chest_signals['ACC'][window_start:window_end]
        activity_features = extract_activity_features(acc_window, fs=fs_dict.get('ACC', 700))
        features.update(activity_features)
    
    return features


def create_feature_matrix(subject_data, window_size_sec=60, overlap=0.0):
    """
    Create feature matrix from entire subject data.
    
    Args:
        subject_data: dict from load_wesad_subject() with 'signal' and 'label' keys
        window_size_sec: window size in seconds (default: 60)
        overlap: overlap fraction (0.0 = non-overlapping, 0.5 = 50% overlap)
    
    Returns:
        DataFrame with features (rows=windows, columns=features+target)
    """
    chest = subject_data['signal']['chest']
    labels = subject_data['label']
    
    # Determine sampling frequency (assume 700 Hz for chest)
    fs = 700
    window_size = int(window_size_sec * fs)
    step_size = int(window_size * (1 - overlap))
    
    # Get signal length
    signal_length = len(labels)
    
    # Extract features for each window
    feature_list = []
    
    for start_idx in range(0, signal_length - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # Extract features
        window_features = extract_window_features(chest, start_idx, end_idx)
        
        # Get majority label for this window
        window_labels = labels[start_idx:end_idx]
        unique, counts = np.unique(window_labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        
        # Convert to mindfulness score
        mindfulness_score = map_labels_to_mindfulness([majority_label])[0]
        
        # Add metadata
        window_features['window_start'] = start_idx
        window_features['window_end'] = end_idx
        window_features['label'] = majority_label
        window_features['mindfulness_index'] = mindfulness_score
        
        feature_list.append(window_features)
    
    # Create DataFrame
    df = pd.DataFrame(feature_list)
    
    # Remove windows with NaN target (transient states)
    df = df.dropna(subset=['mindfulness_index'])
    
    return df
