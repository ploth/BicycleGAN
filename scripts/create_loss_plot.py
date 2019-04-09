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


if __name__ == '__main__':
    loss_log_file, output_folder = parse_args()

    with open(str(loss_log_file), 'r') as file:
        loss_log = file.readlines()

    values = defaultdict(lambda: [])
    point = re.compile(
        r'\((epoch):\s*(\d+),\s*(iters):\s*(\d+),\s*(time):\s*(\d*\.\d+),\s*(data):\s*(\d*\.\d+)\)\s*(G_GAN):\s*(\d*.\d+)\s*(D):\s*(\d*\.\d+)\s*(G_GAN2):\s*(\d*\.\d+)\s*(D2):\s*(\d*.\d+)\s*(G_L1):\s*(\d*\.\d+)\s*(G_IL1):\s*(\d*\.\d+)\s*(z_L1):\s*(\d*\.\d+)\s*(kl):\s*(\d*\.\d+)\s*'
    )
    for line in loss_log:
        match = re.match(point, line)
        if match:
            for i in range(1, len(match.groups()), 2):
                values[match[i]].append(float(match[i + 1]))


    x = range(0, len(values['G_GAN']))
    data = []
    for key, value in values.items():
        if key != 'epoch' and key != 'iters':
            data.append(np.array(value))
    #  data1 = np.array(values['G_GAN'])
    #  data2 = np.array(values['G_GAN2'])
    fig, ax = plt.subplots()
    for entry in data:
        ax.plot(x, entry)
    plt.show()
