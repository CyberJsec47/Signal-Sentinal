import os
from math import ceil

from colorama import Fore, Style

folder_path = "/home/kali/Documents/SignalSentinal"
fs = 1_000_000


def capture_menu():
    folder = folder_path

    extensions = ('*.dat', '*.csv')

    files = [
        file for file in os.listdir(folder_path)
        if file.endswith(('.csv', '.dat'))
    ]

    print("Files in folder:", files)

    num_columns = 1
    num_rows = ceil(len(files) / num_columns)

    data_rows = []
    max_lengths = []

    for i in range(num_rows):
        row = files[i * num_columns:(i + 1) * num_columns]
        data_rows.append(row)
    for col in range(num_columns):
        max_len = max([len(data_rows[row][col]) for row in range(len(data_rows)) if col < len(data_rows[row])],
                      default=0)
        max_lengths.append(max_len)

    print(Fore.GREEN + "Files found:" + Style.RESET_ALL)
    border = Fore.BLUE + "+" + "+".join([Fore.BLUE + "-" * (length + 2) for length in max_lengths]) + Fore.BLUE + "+"
    print(border)
    for row in data_rows:
        row_display = Fore.BLUE + "|"
        for col in range(num_columns):
            if col < len(row):
                row_display += " " + Fore.GREEN + row[col].ljust(max_lengths[col]) + Fore.BLUE + " |"
            else:
                row_display += " " * (max_lengths[col] + 2) + Fore.BLUE + "|"
        print(row_display)
        print(border)
