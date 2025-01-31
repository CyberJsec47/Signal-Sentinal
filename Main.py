from Calculations import *
from pyfiglet import Figlet
from colorama import Fore
from SigCapture import *
from Functions import *

file = '/home/kali/Documents/SignalSentinel/iq_samples.dat'
iq_data = np.fromfile(file, dtype=np.complex64)
fs = 1_000_000


def main():
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
            print(Fore.BLUE + "Choose a frequency and capture time")
            freq_hz = freq_select()
            seconds = int(input(Fore.BLUE + "Time in Seconds: "))
            get_signal(seconds, freq_hz)
            main()
        elif option == 2:
            feature_extraction(iq_data, fs)
            again = input("Return to main or exit (M or E): ")
            if again == ('m' or 'M'):
                main()
            else:
                print(Fore.RED + f.renderText("Exiting"))
                quit()
        elif option == 3:
            print(Fore.RED + "Placeholder")
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


if __name__ == "__main__":
    main()
