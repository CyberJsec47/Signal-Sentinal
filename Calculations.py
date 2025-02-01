import csv
import numpy as np 
from scipy.stats import entropy
from scipy.signal import welch
import colorama
from colorama import Fore
from SigCapture import freq_select

colorama.init(autoreset=True)

file = 'iq_samples.dat'
iq_data = np.fromfile(file, dtype=np.complex64)
fs = 1_000_000


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
    psd = welch(sample_file, sample_rate, nperseg=1024, return_onesided=False)
    avg_mag = np.mean(psd)
    return avg_mag


def find_amplitude(sample_file):
    mag = np.abs(sample_file)
    amplitude = (np.max(mag) - np.min(mag)) / 2
    return amplitude


def feature_extraction(sample_file, frequency):

    Freq = frequency
    SNR = calculate_snr(sample_file)
    Mag = max_mag(sample_file)
    Phase = sig_phase(sample_file)
    Entropy = find_entropy(sample_file)
    PSD = find_psd(sample_file, fs)
    Amplitude = find_amplitude(sample_file)

    print(Fore.BLUE + f"Frequency:{Fore.GREEN}{Freq}\n{Fore.BLUE}Signal to Noise ratio:{Fore.GREEN}{SNR}\n{Fore.BLUE}Magnitude:{Fore.GREEN}{Mag}\n{Fore.BLUE}Phase:{Fore.GREEN}"
                      f"{Phase}\n{Fore.BLUE}Entropy:{Fore.GREEN}{Entropy}\n{Fore.BLUE}Power Spectral Density:{Fore.GREEN}{PSD}\n{Fore.BLUE}Amplitude:{Fore.GREEN}{Amplitude}")


def export_csv(sample_file, output_file='Features.csv'):
    frequency = freq_select()
    features = {
        'Frequency': frequency,
        'Signal To Noise': calculate_snr(sample_file),
        'Max Magnitude': max_mag(sample_file),
        'Average Phase': sig_phase(sample_file),
        'Entropy': find_entropy(sample_file),
        'PSD': find_psd(sample_file, fs),
        'Amplitude': find_amplitude(sample_file)
    }

    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=features.keys())
        writer.writeheader()
        writer.writerow(features)
