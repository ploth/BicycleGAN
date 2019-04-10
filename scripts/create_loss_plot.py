#!/usr/bin/env python3

import re
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="create loss plot",
        formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument("loss_log_file", help="The loss log file.")
    arg_parser.add_argument("output_folder", help="The output folder")
    args = arg_parser.parse_args()
    return Path(args.loss_log_file), Path(args.output_folder)


def parse_file(loss_log):
    caption = re.compile(r'^=')
    key_value_pair_pattern = re.compile(r'(\w+):\s*(\d*.\d+)')

    data = np.array([])
    identifiers = []
    for line in loss_log:
        if re.match(caption, line):
            continue

        tuples = re.findall(key_value_pair_pattern, line)
        identifiers = [pair[0] for pair in tuples]
        values = [float(pair[1]) for pair in tuples]
        data = np.append(data, values)

    data = np.reshape(data, (-1, len(identifiers)))
    return identifiers, data

def plot(identifiers, data):
    fig, ax = plt.subplots()
    x = range(0, data.shape[0])
    ax.plot(x, data, linewidth=1)
    ax.legend(identifiers)
    plt.show()

if __name__ == '__main__':
    loss_log_file, output_folder = parse_args()

    with open(str(loss_log_file), 'r') as file:
        loss_log = file.readlines()

    # Parse file
    identifiers, data = parse_file(loss_log)

    # Filter list
    ignore = ['epoch', 'iters', 'time', 'data']

    # Create list of indexes to filter
    indexes_to_ignore = [identifiers.index(element) for element in ignore]

    # Create new identifier list
    identifiers = [element for element in identifiers if element not in ignore]

    # Delete unrelevant columns in data
    data = np.delete(data, indexes_to_ignore, 1)

    plot(identifiers, data)
