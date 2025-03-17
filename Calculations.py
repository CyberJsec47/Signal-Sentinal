import csv
import numpy as np 
from scipy.stats import entropy
from scipy.signal import welch
import colorama
from colorama import Fore
import os
import pandas as pd


colorama.init(autoreset=True)
# settings for file location, converting the raw data to a complex64 number, sample rate and rtl gain
# rtl is experimental needs to be adjusted when dongle is back in use
file = 'iq_samples.dat'
iq_data = np.fromfile(file, dtype=np.complex64)
fs = 1_000_000
rtl_gain = 28.0
jam_file = 'Jamming_raw_iq.dat'
jam_iq = np.fromfile(jam_file, dtype=np.complex64)


def calculate_snr(sample_file):
    signal_power = np.mean(np.abs(sample_file)**2)
    noise_power = np.mean(np.abs(sample_file - np.mean(sample_file))**2)

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def max_mag(sample_file):
    magnitude = np.max(np.abs(sample_file))
    return magnitude


def sig_phase(sample_file):
    instantaneous_phase = np.angle(sample_file)
    unwrapped_phase = np.unwrap(instantaneous_phase)
    average_phase = np.mean(unwrapped_phase)
    return average_phase


def find_entropy(sample_file):
    mag = np.abs(sample_file)
    hist, bin_edges = np.histogram(mag, bins=256, density=True)
    prob_distro = hist / hist.sum()
    iq_entropy = entropy(prob_distro)
    return iq_entropy


def find_psd(sample_file, sample_rate):
    if np.all(sample_file == 0):
        return -np.inf
    sample_file = sample_file / np.max(np.abs(sample_file))
    f, psd = welch(sample_file, sample_rate, nperseg=1024, return_onesided=True)
    avg_psd = np.mean(np.log10(psd + 1e-10))  
    return avg_psd


def find_amplitude(sample_file):
    mag = np.abs(sample_file)
    amplitude = (np.max(mag) - np.min(mag)) / 2
    return amplitude


def find_rms(sample_file):
    rms = np.sqrt(np.mean(np.abs(sample_file)**2))
    return rms


def find_dBm(sample_file, gain_dB=0):
    gain_dB = float(gain_dB) 
    mag = np.abs(sample_file)
    power_linear = mag**2
    power_linear[power_linear == 0] = 1e-12
    calibration_factor = -30 + gain_dB
    power_dBm = 10 * np.log10(power_linear) + calibration_factor
    avg_power_dBm = 10 * np.log10(np.mean(power_linear)) + calibration_factor
    return round(avg_power_dBm, 2)



def feature_extraction(sample_file, frequency, fs, rtl_gain):
    """
    Extract features from the sample_file and return them as a list/array
    to be standardized and input into the model.
    """
    Freq = frequency
    SNR = calculate_snr(sample_file)
    Mag = max_mag(sample_file)
    Phase = sig_phase(sample_file)
    Entropy = find_entropy(sample_file)
    PSD = find_psd(sample_file, fs)
    Amplitude = find_amplitude(sample_file)
    RMS = find_rms(sample_file)
    avg_dBm = find_dBm(sample_file, rtl_gain)

    print(Fore.BLUE + f"Frequency: {Fore.GREEN}{Freq}\n{Fore.BLUE}Signal to Noise ratio: {Fore.GREEN}{SNR}\n{Fore.BLUE}Magnitude: {Fore.GREEN}{Mag}\n{Fore.BLUE}Phase: {Fore.GREEN}"
                      f"{Phase}\n{Fore.BLUE}Entropy: {Fore.GREEN}{Entropy}\n{Fore.BLUE}Power Spectral Density: {Fore.GREEN}{PSD}\n{Fore.BLUE}Amplitude: {Fore.GREEN}{Amplitude}"
                      f"\n{Fore.BLUE}RMS: {Fore.GREEN}{RMS}\n{Fore.BLUE}dBm: {Fore.GREEN}{avg_dBm:.2f}")

    features = pd.DataFrame([[SNR, Mag, Phase, Entropy, PSD, Amplitude, RMS, avg_dBm]], 
                            columns=["Signal To Noise", "Max Magnitude", "Avg dBm", "Average Phase", "Entropy", 
                                     "PSD", "Amplitude", "RMS"])

    return features


def export_csv(sample_file, frequency, fs, classification):

    output_file='Training_data.csv'

    features = {
        'Frequency': frequency,
        'Signal To Noise': calculate_snr(sample_file),
        'Max Magnitude': max_mag(sample_file),
        'Avg dBm': find_dBm(sample_file, rtl_gain),
        'Average Phase': sig_phase(sample_file),
        'Entropy': find_entropy(sample_file),
        'PSD': find_psd(sample_file, fs),
        'Amplitude': find_amplitude(sample_file),
        'RMS': find_rms(sample_file),
        'Signal': classification
    }

    file_exists = os.path.isfile(output_file)

    try:
         with open(output_file, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=features.keys())

            if not file_exists:
                writer.writeheader()  

            writer.writerow(features)
            print(Fore.GREEN + f"Data saved to {output_file}")

    except Exception as e:
        print(f"Error writing to CSV: {e}")
