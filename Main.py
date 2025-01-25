from calculations import *
from pyfiglet import Figlet
from colorama import Fore
from rawIQ import *


def main():
    while True:

        f = Figlet(font='slant')
        print(Fore.RED + f.renderText("Signal Sentry"))
        print("A machine learning project aimed at passively detecting RF jamming attacks")
        print("Designed to work on small form embedded systems for remote detection and response automation")
        print("Josh Perryman Bcs(Hons) Cyber Security 2025\n")
        print(Fore.BLUE + "-"*25)
        print(Fore.BLUE + "| Main Menu              |")
        print(Fore.BLUE + "-"*25)
        print(Fore.BLUE + "| Capture a signal   [1] |")
        print(Fore.BLUE + "-"*25)
        print(Fore.BLUE + "| Add signal to CSV  [2] |")
        print(Fore.BLUE + "-"*25)
        print(Fore.BLUE + "| Analyse Signal     [3] |")
        print(Fore.BLUE + "-"*25)
        print(Fore.BLUE + "| Exit               [4] |")
        print(Fore.BLUE + "-"*25)
        option = int(input(Fore.BLUE + "| Choose an option:"))
        if option == 1:
            print(Fore.BLUE + "Capturing signal...")
            capture_signal(top_block_cls=rawIQ, options=None)







main()
