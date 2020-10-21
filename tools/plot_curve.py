import matplotlib.pyplot as plt
import numpy as np
import re
import argparse


def plot_acc(log_name, color="r"):
    train_name = log_name.replace("log.txt", " train")
    with open(log_name) as f:
        lines = f.readlines()

    val_c = []
    val_p = []
    val_r = []
    x = []

    training_start = False
    idx = 0
    for l in lines:
        if training_start and idx >= 10:
            values = l.replace(' ', '').split('|')
            val_c.append(float(values[1]) + float(values[2]))
            val_p.append(float(values[3]) + float(values[4]))
            val_r.append(float(values[5]) + float(values[6]))
            x.append(idx)
        if training_start:
            idx += 1
        if 'progress' in l:
            training_start = True

    plt.subplot(3, 1, 1)
    plt.plot(x, val_c, '-', linestyle='--', color=color, linewidth=2, label=train_name)
    plt.subplot(3, 1, 2)
    plt.plot(x, val_p, '-', linestyle='--', color=color, linewidth=2, label=train_name)
    plt.subplot(3, 1, 3)
    plt.plot(x, val_r, '-', linestyle='--', color=color, linewidth=2, label=train_name)


def main():
    plt.figure(figsize=(32, 18))
    plt.xlabel("epoch")
    plt.ylabel("Top-1 accuracy")
    color = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'o']
    log_files = args.logs
    color = color[:len(log_files)]
    for c in range(len(log_files)):
        plot_acc(log_files[c], color[c])
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(args.out, format='svg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parses log file and generates train/val curves, using like: \n'
                                     'python -u plot_curve.py --log=resnet-18.log,resnet-50.log')
    parser.add_argument('--logs', nargs='*', help='the path of log file, --logs=resnet-50.log,resnet-101.log')
    parser.add_argument('--out', type=str, default="val.svg", help='the name of output curve ')
    parser.add_argument('--accuracy', type=bool, default=True, help='True: plot accuracy False: plot error rate')
    args = parser.parse_args()
    main()
