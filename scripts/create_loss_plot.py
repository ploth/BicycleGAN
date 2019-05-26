#!/usr/bin/env python3

import re
from collections import defaultdict
from pathlib import Path
from math import sqrt
default_width = 5.78853 # in inches
default_ratio = (sqrt(5.0) - 1.0) / 2.0 # golden mean
import matplotlib as mpl
mpl.use('pgf')
mpl.rcParams.update({
    "text.usetex": True, # use inline math for ticks
    "pgf.texsystem": "lualatex",
    "pgf.rcfonts": False, # don't setup fonts from rc parameters
    "font.size": 12,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "figure.figsize": [default_width, default_width * default_ratio],
    #  "pgf.preamble": [
    #      r"\usepackage{metalogo}",
    #  ],
})
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, resample
import numpy as np
import seaborn as sns


def parse_args():
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="create loss plot",
        formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument(
        "-k", "--kernel", type=int, help="Kernel size used for low-pass.")
    arg_parser.add_argument(
        "-p",
        "--polynomial",
        type=int,
        help="The polynomial order used for low-pass.")
    arg_parser.add_argument(
        "-m", "--ymax", type=float, help="Maximum value of y axis.")
    arg_parser.add_argument("-i", "--input", type=str, help="Log file to plot.")
    arg_parser.add_argument("-o", "--output", type=str, help="Path to output figure (without suffix).")
    args = arg_parser.parse_args()
    return args


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


def get_number_of_epochs(identifiers, data):
    index = identifiers.index('epoch')
    return int(np.amax(data[:, index]))


def plot(identifiers, data, epochs, output_path):
    fig, ax = plt.subplots()
    x = range(0, epochs)
    for i in range(0, data.shape[1]):
        y = data[:,i]
        y = resample(y, epochs)
        y = savgol_filter(y, args.kernel, args.polynomial, mode='nearest')
        ax.plot(x, y, linewidth=1)

    ax.legend(identifiers)
    plt.xlabel(r'Epoche')
    plt.ylabel(r'Verlust')
    plt.xlim(0, epochs)
    plt.ylim(0, args.ymax)
    plt.tight_layout()
    #  plt.show()
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.savefig(output_path.with_suffix('.pgf'))


if __name__ == '__main__':
    args = parse_args()

    loss_log_file = Path(args.input)
    with open(str(loss_log_file), 'r') as file:
        loss_log = file.readlines()

    # Parse file
    identifiers, data = parse_file(loss_log)

    epochs = get_number_of_epochs(identifiers, data)

    # Filter list
    ignore = ['epoch', 'iters', 'time', 'data']
    ignore.append('G_IL1')

    # Create list of indexes to filter
    indexes_to_ignore = [identifiers.index(element) for element in ignore]

    # Create new identifier list
    identifiers = [element for element in identifiers if element not in ignore]

    # Latex identifiers
    for i, identifier in enumerate(identifiers):
        if identifier == 'G_GAN':
            identifiers[i] = '$\mathcal{L}_{GAN}^{VAE}(G)$'
        if identifier == 'D':
            identifiers[i] = '$\mathcal{L}_{GAN}^{VAE}(D)$'
        if identifier == 'G_GAN2':
            identifiers[i] = '$\mathcal{L}_{GAN}(G)$'
        if identifier == 'D2':
            identifiers[i] = '$\mathcal{L}_{GAN}(D)$'
        if identifier == 'G_L1':
            identifiers[i] = '$\mathcal{L}_{1}^{VAE}(G,E)$'
        if identifier == 'G_IL1':
            identifiers[i] = '$1 / (\mathcal{L}_{2}^{latent}(E)\cdot\mathcal{L}_{1}^{LR}(G)$)'
        if identifier == 'z_L1':
            identifiers[i] = '$\mathcal{L}_{1}^{latent}(G,E)$'
        if identifier == 'kl':
            identifiers[i] = '$\mathcal{L}_{KL}(E)$'

    # Delete unrelevant columns in data
    data = np.delete(data, indexes_to_ignore, 1)

    add_totals = False
    if add_totals:
        # Create column for total loss
        totals = np.empty([data.shape[0], 1])
        # Calculate total loss
        totals = np.sum(data, axis=1)
        totals = totals[..., np.newaxis]
        # Append total loss to data array
        data = np.concatenate((data, totals), axis=1)
        # Add identifier
        identifiers.append('totals')

    # Plot
    sns.set_palette("bright")
    plot(identifiers, data, epochs, Path(args.output))
