import numpy as np
from scipy import signal
from scipy.integrate import simps


def calculate_band_power(sig, low, high, sf, window):
    freqs, psd = signal.welch(sig, sf, nperseg=window)
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    freq_res = freqs[1] - freqs[0]
    band_power = simps(psd[idx_band], dx=freq_res)
    total_power = simps(psd, dx=freq_res)
    band_rel_power = band_power / total_power
    # return float("{:.3f}".format(band_rel_power))
    return band_rel_power


def get_band_power_values(channels, bands, sf, window):
    band_power_values = []
    for channel in channels:
        for band in bands:
            band_power_values.append(calculate_band_power(
                channel, bands[band][0], bands[band][1], sf, window))
    return band_power_values


def get_band_power_differences(pairs, bands, row):
    differences = []
    for pair in pairs:
        for band in bands:
            if band != 'Slow Alpha':
                cl_name_1 = pair[0] + '_' + band
                cl_name_2 = pair[1] + '_' + band
                diff = abs(row[cl_name_1] - row[cl_name_2])
                differences.append(float("{:.3f}".format(diff)))
    return differences
