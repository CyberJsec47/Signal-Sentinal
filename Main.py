from Calculations import *
from pyfiglet import Figlet
from colorama import Fore
import numpy as np
from Functions import *

file = 'iq_samples.dat'
iq_data = np.fromfile(file, dtype=np.complex64)
fs = 1_000_000

def opening_script():
    f = Figlet(font='slant')
    print(Fore.RED + f.renderText("Signal Sentinel"))
    print("A machine learning project aimed at passively detecting RF jamming attacks")
    print("Designed to work on small form embedded systems for remote detection and response automation")
    print("Josh Perryman Bcs(Hons) Cyber Security 2025\n")
    if check_rtl_sdr():
        print(Fore.GREEN + "RTL_SDR Device found")
    else:
        print(Fore.RED + "No RTL-SDR device found, please reinstall device and start again")
        exit()
    freq_hz = freq_select()
    return freq_hz
    

def main(frequency):

    f = Figlet(font='slant')
    freq_hz = frequency


    try: 
        while True:

            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| Main Menu               |")
            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| Capture a signal    [1] |")
            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| Change frequency    [2] |")
            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| View CSV sample     [3] |")
            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| Exit                [4] |")
            print(Fore.BLUE + "-" * 25)

            option = int(input(Fore.GREEN + "| Choose an option: "))
            print(Fore.BLUE + "-" * 25)

            if option == 1:
                print(Fore.GREEN + "Choose how long to capture for: (seconds)")
                seconds = int(input(Fore.GREEN + "Time in Seconds: "))
                print(Fore.GREEN + f"Capturing on frequency {freq_hz} MHz for {seconds} seconds")
                rolling_window(seconds, freq_hz)

                new = input(Fore.GREEN + "Redo signal capture? 'y' or 'n': ")
                if new == ('y' or 'Y'):
                    seconds = int(input(Fore.GREEN + "Time in Seconds: "))
                    print(Fore.GREEN + f"Capturing on frequency {freq_hz} MHZ for {seconds} seconds")
                    get_signal(seconds, freq_hz)
                    feature_extraction(iq_data, freq_hz)
                    continue
                else:
                    main(freq)

            elif option == 2:
                print(Fore.YELLOW + "Changing frequency...")
                freq_hz = freq_select()
                print(Fore.GREEN + f"New frequency set: {freq_hz} MHz")
                continue 

            elif option == 3:
                print("Getting CSV file samples...")
                view_CSV('Safe_signals.csv', 20)
                input(Fore.GREEN + "\nPress Enter to return to the menu...")
                continue


            elif option == 4:
                print(Fore.RED + f.renderText("Exiting"))
                quit()

            else:
                print(Fore.RED + "Select an option")

    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
        exit()


if __name__ == "__main__":
    freq = opening_script()
    main(freq)