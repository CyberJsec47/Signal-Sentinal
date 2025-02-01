import numpy as np
from rtlsdr import RtlSdr
import time
from colorama import Fore


def freq_select():
<<<<<<< HEAD
    frequency = float(input(Fore.GREEN + "Frequency(MHz): "))
=======
    frequency = float(input(Fore.BLUE + "Frequency(MHz): "))
>>>>>>> 5cbbe08eea3435f52a1064e9962addf3288a7430
    freq_mhz = float(frequency)
    freq_in_hz = freq_mhz * 1e6
    return freq_in_hz


def get_signal(seconds, frequency):
    sdr = RtlSdr()

    try:
        freq = sdr.center_freq = frequency
        fs = sdr.fs = 1e6
        sdr.gain = 'auto'

<<<<<<< HEAD
=======
        print(f"Starting signal capture on frequency {freq} for {seconds} seconds")

>>>>>>> 5cbbe08eea3435f52a1064e9962addf3288a7430
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
<<<<<<< HEAD
        output_path = 'iq_samples.dat'
        captured_samples_np.tofile(output_path)

        print(Fore.GREEN + f"Capture complete. collected {len(captured_samples)} samples\n Saving to {output_path}")
=======
        output_path = '/home/kali/Documents/SignalSentinal/iq_samples.dat'
        captured_samples_np.tofile(output_path)

        print(f"Capture complete. collected {len(captured_samples)} samples\n Saving to {output_path}")
>>>>>>> 5cbbe08eea3435f52a1064e9962addf3288a7430
    finally:
        sdr.close()
