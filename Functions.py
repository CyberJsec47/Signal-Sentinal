import subprocess
import numpy as np
from rtlsdr import RtlSdr
import time
from colorama import Fore
import csv
from Calculations import *
import matplotlib.pyplot as plt 


def freq_select():
    frequency = float(input(Fore.GREEN + "Choose a frequency to use (MHz): "))
    freq_mhz = float(frequency)
    freq_in_hz = freq_mhz * 1e6
    print(Fore.GREEN + f"Using {freq_in_hz} Hz")
    return freq_in_hz



def get_signal(seconds, frequency):
    sdr = RtlSdr()

    try:
        sdr.center_freq = frequency  
        sdr.sample_rate = 1e6  
        sdr.gain = 'auto'  

        total_samples = int(seconds * sdr.sample_rate)  
        chunk_size = 256 * 1024 
        captured_samples = []

        sdr.read_samples(2048)

        while len(captured_samples) < total_samples:
            samples_needed = total_samples - len(captured_samples)
            chunk = sdr.read_samples(min(chunk_size, samples_needed))  
            captured_samples.extend(chunk)

        captured_samples_np = np.array(captured_samples, dtype=np.complex64)
        output_path = "iq_samples.dat"
        captured_samples_np.tofile(output_path)

        print(Fore.GREEN + f"Capture complete. Collected {len(captured_samples)} samples\nSaving to {output_path}")

    finally:
        sdr.close()

def check_rtl_sdr():
    try:
        result = subprocess.run(["lsusb"], capture_output=True, text=True)
        return any("Realtek" in line or "RTL2838" in line for line in result.stdout.splitlines())
    except Exception as e:
        print(Fore.RED + f"Error checking RTL-SDR with lsusb: {e}")
        return False
    

def view_CSV(file_path, num_rows):

    with open (file_path) as file:
        
        reader_obj = csv.reader(file, delimiter=',', quotechar='|')
        for i, row in enumerate(reader_obj):
            print(', '.join(row))
            if i + 1 >= num_rows: 
                break
 

            


def rolling_window(seconds, frequency, classification):

    fs = 1_000_000
    sdr = RtlSdr()

    try:
        freq = sdr.center_freq = frequency
        fs = sdr.fs = 1e6
        sdr.gain = 'auto'

        total_samples = int(seconds * fs)
        chunk_size = 256 * 1024
        captured_samples = []
        start_time = time.time()

        while len(captured_samples) < total_samples:
            sampled_needed = total_samples - len(captured_samples)
            chunk = sdr.read_samples(min(chunk_size, sampled_needed))
            captured_samples.extend(chunk)

            if time.time() - start_time >= seconds:
                break

        captured_samples_np = np.array(captured_samples, dtype=np.complex64)
        output_path = 'iq_samples.dat'
        captured_samples_np.tofile(output_path)

    finally:
        sdr.close()

        file = 'iq_samples.dat'
        iq_data = np.fromfile(file, dtype=np.complex64)
        feature_extraction(iq_data, frequency)
        export_csv(iq_data, frequency, fs, seconds, classification)


def visualise_signal(file, freq_hz):

    file_path = file

    samples = np.fromfile(file, dtype=np.complex64)

    fs = 1e6
    fc = freq_hz

    N = len(samples)
    fft_data = np.fft.fftshift(np.abs(np.fft.fft(samples))**2)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs)) + fc  

    plt.figure(figsize=(10, 5))
    plt.plot(freqs / 1e6, 10 * np.log10(fft_data))  
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB)")
    plt.title(f"FFT Spectrum (Centered at {fc/1e6} MHz)")
    plt.grid()
    plt.show()
