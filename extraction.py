from scipy import signal
import pandas as pd
import numpy as np


def split_to_windows(data, n_windows):
    merged_windows = []
    for sig in data:
        windows = np.array_split(sig, n_windows)
        merged_windows.append(windows)
    return merged_windows


def calculate_band_psd(merged_windows, sampling_freq, bands):
    merged_bands = []
    for channel in merged_windows:
        channel_bands = []
        for window in channel:
            band_psd_values = []
            freqs, psd = signal.welch(window, sampling_freq)
            for band in bands:
                idx_band = np.logical_and(
                    freqs >= bands[band][0], freqs <= bands[band][1])
                avg_psd_band = np.mean(psd[idx_band])
                band_psd_values.append(avg_psd_band)
            channel_bands.append(band_psd_values)
        merged_bands.append(channel_bands)

    return np.array(merged_bands)


def extract_features(data, n_windows, sampling_freq, bands, col_names):
    merged_windows = split_to_windows(data, n_windows)

    merged_bands = calculate_band_psd(merged_windows, sampling_freq, bands)

    reshaped_features = merged_bands.transpose(
        1, 0, 2).reshape(merged_bands.shape[1], -1)

    return pd.DataFrame(reshaped_features, columns=col_names)
