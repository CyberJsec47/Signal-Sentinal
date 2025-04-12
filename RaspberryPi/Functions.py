import subprocess
import numpy as np
from rtlsdr import RtlSdr
import time
from colorama import Fore
from Calculations import *
import pickle
from pyfiglet import Figlet
from termcolor import colored
from RPLCD.i2c import CharLCD


lcd = CharLCD(
    i2c_expander='PCF8574',
    address=0x27,
    port=1,
    cols=16,
    rows=2,
    charmap='A00',
    auto_linebreaks=True,
    backlight_enabled=True
)
lcd.clear()

def freq_select():
    frequency = float(input(Fore.GREEN + "Choose a frequency to use (MHz): "))
    freq_mhz = float(frequency)
    freq_in_hz = freq_mhz * 1e6
    print(Fore.GREEN + f"Using {freq_in_hz} Hz")
    return freq_in_hz


def check_rtl_sdr():
    try:
        result = subprocess.run(["lsusb"], capture_output=True, text=True)
        return any("Realtek" in line or "RTL2838" in line for line in result.stdout.splitlines())
    except Exception as e:
        print(Fore.RED + f"Error checking RTL-SDR with lsusb: {e}")
        return False


def signalCapture(seconds, frequency):
    
    fs = 1e6
    sdr = RtlSdr()

    try:
        sdr.center_freq = frequency
        sdr.sample_rate = fs
        sdr.gain = 28.0
        time.sleep(0.5)

        end_time = time.time() + seconds
        captured_samples = []

        while time.time() < end_time:
            chunk = sdr.read_samples(2048)
            captured_samples.extend(chunk)
            time.sleep(0.1)

        captured_samples_np = np.array(captured_samples, dtype=np.complex64)
        captured_samples_np.tofile('iq_samples.dat')

    finally:
        sdr.close()

    features = feature_extraction(captured_samples_np, frequency, fs, sdr.gain)
    return features


def modelTest(sample_file, frequency, fs, rtl_gain):
    """
    Function to load the trained model and scaler, then predict on new sample data.
    """
    f = Figlet(font='big')

    try:
        with open("naive_bayes_model.pkl", "rb") as model_file:
            naiveBayes = pickle.load(model_file)

        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        new_features = feature_extraction(sample_file, frequency, fs, rtl_gain)
        if new_features is None:
            print(Fore.RED + "Feature extraction returned None.")
            lcd.clear() 
            lcd.write_string('Feature Extraction Error')
            return "Error"

        new_features_selected = new_features[['RMS', 'Max Magnitude', 'Amplitude', 'PSD', 'Signal To Noise']]
        new_features_scaled = scaler.transform(new_features_selected)

        prediction = naiveBayes.predict(new_features_scaled)
        prediction_text = "Safe" if prediction[0] == 0 else "Jamming"

        lcd.clear() 
        time.sleep(0.1)  

        if prediction[0] == 0:
            print(Fore.GREEN + f.renderText(f"{prediction_text}"))
            lcd.write_string('Safe        ')
        else:
            print(Fore.RED + f.renderText(f"{prediction_text}"))
            lcd.write_string('Jamming     ')

        return prediction_text

    except Exception as e:
        print(Fore.RED + f"Error in modelTest: {e}")
        lcd.clear() 
        lcd.write_string('Error        ')
        return "Error"
