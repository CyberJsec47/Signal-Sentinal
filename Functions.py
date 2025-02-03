import subprocess
from colorama import Fore

def check_rtl_sdr():
    try:
        result = subprocess.run(["lsusb"], capture_output=True, text=True)
        return any("Realtek" in line or "RTL2838" in line for line in result.stdout.splitlines())
    except Exception as e:
        print(Fore.RED + f"Error checking RTL-SDR with lsusb: {e}")
        return False

 
