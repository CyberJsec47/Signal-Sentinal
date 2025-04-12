import time
from rtlsdr import RtlSdr
from Functions import * 
import numpy as np
import warnings
from pyfiglet import Figlet

warnings.filterwarnings("ignore", message="Input data is complex")


def opening_script():
    f = Figlet(font='slant')
    print(Fore.RED + f.renderText("Signal Sentinel"))

    if check_rtl_sdr():
        print(Fore.GREEN + "RTL_SDR Device found")
    else:
        print(Fore.RED + "No RTL-SDR device found, please reinstall device and start again")

    freq_hz = freq_select()
    return freq_hz


def main(frequency):
    f = Figlet(font='slant')
    freq_hz = frequency

    try:
        while True:
            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| Main Menu             |")
            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| Run scan          [1] |")
            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| Change frequency  [2] |")
            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| Exit              [3] |")
            print(Fore.BLUE + "-" * 25)

            option = input(Fore.GREEN + "| Choose an option: ")
            print(Fore.BLUE + "-" * 25)

            try:
                option = int(option)
            except ValueError:
                print(Fore.RED + "Invalid input. Please enter a valid number.")
                time.sleep(0.1)
                continue


            if option == 1:
                print(Fore.YELLOW + "Starting continuous scan...")
                try:
                    while True:
                        seconds = 5
                        features = signalCapture(seconds, freq_hz)
                        if features is None:
                            print(Fore.RED + "Scan was interrupted or stopped.")
                            break

                        file = 'iq_samples.dat'
                        iq_data = np.fromfile(file, dtype=np.complex64)
                        rtl_gain = 30
                        modelTest(iq_data, freq_hz, 1e6, rtl_gain)

                        time.sleep(1.0) 
                except KeyboardInterrupt:
                    print(Fore.RED + "\nScan interrupted by user. Returning to menu.")
                    continue


            elif option == 2:
                print(Fore.YELLOW + "Changing frequency...")
                freq_hz = freq_select()
                print(Fore.GREEN + f"New frequency set: {freq_hz} MHz")
                continue 

            elif option == 3:
                print(Fore.RED + f.renderText("Exiting"))
                quit()  

            else:
                print(Fore.RED + "Select a valid option.")
                continue 

    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
        exit()



if __name__ == "__main__":
    freq = opening_script()
    main(freq)
