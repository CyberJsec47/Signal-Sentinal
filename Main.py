from Calculations import *
from pyfiglet import Figlet
from colorama import Fore
from SigCapture import *
from Functions import *
import numpy as np

file = 'iq_samples.dat'
iq_data = np.fromfile(file, dtype=np.complex64)
fs = 1_000_000


def main():

    try: 
        while True:

            f = Figlet(font='slant')
            print(Fore.RED + f.renderText("Signal Sentinel"))
            print("A machine learning project aimed at passively detecting RF jamming attacks")
            print("Designed to work on small form embedded systems for remote detection and response automation")
            print("Josh Perryman Bcs(Hons) Cyber Security 2025\n")
            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| Main Menu              |")
            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| Capture a signal   [1] |")
            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| View signal data   [2] |")
            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| Analyse Signal     [3] |")
            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| Generate data      [4] |")
            print(Fore.BLUE + "-" * 25)
            print(Fore.GREEN + "| Exit               [5] |")
            print(Fore.BLUE + "-" * 25)
            option = int(input(Fore.GREEN + "| Choose an option: "))
            print(Fore.BLUE + "-" * 25)

            if option == 1:
                print(Fore.GREEN + "Choose a frequency and capture time")
                freq_hz = freq_select()
                seconds = int(input(Fore.GREEN + "Time in Seconds: "))
                print(Fore.GREEN + f"Capturing on frequency {freq_hz} MHz for {seconds} seconds")
                get_signal(seconds, freq_hz)
                choice = input(Fore.GREEN + "View captured data? 'y' or 'n': ")

                if choice == ('y' or 'Y'):
                    feature_extraction(iq_data, freq_hz)

                    new = input(Fore.GREEN + "Redo signal capture? 'y' or 'n': ")
                    if new == ('y' or 'Y'):
                        freq_hz = freq_select()
                        seconds = int(input(Fore.GREEN + "Time in Seconds: "))
                        print(Fore.GREEN + f"Capturing on frequency {freq_hz} MHZ for {seconds} seconds")
                        get_signal(seconds, freq_hz)
                        feature_extraction(iq_data, freq_hz)
                        main()
                else:
                    main()
            elif option == 2:
                print(Fore.RED + "Opt 2 placeholder")
                again = input("Return to main or exit (M or E): ")
                if again == ('m' or 'M'):
                    main()
                else:
                    print(Fore.RED + f.renderText("Exiting"))
                    quit()
            elif option == 3:
                print(Fore.RED + "Placeholder")
                again = input("Return to main or exit (M or E): ")
                if again == ('m' or 'M'):
                    main()
                else:
                    print(Fore.RED + f.renderText("Exiting"))
                    quit()
                main()
            elif option == 4:
                print("Generating data placeholder")
                again = input("Return to main or exit (M or E): ")
                if again == ('m' or 'M'):
                    main()
                else:
                    print(Fore.RED + f.renderText("Exiting"))
                    quit()
            elif option == 5:
                print(Fore.RED + f.renderText("Exiting"))
                quit()
            else:
                print(Fore.RED + "Select an option")

    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
        exit()


if __name__ == "__main__":
    main()