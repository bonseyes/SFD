#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


if __name__ == "__main__":
    logfile = sys.argv[1]
    with open(logfile, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    test_interv = [int(l.split()[1]) for l in lines if "test_interval" in l][0]
    last_iter = [int(l.split()[5].strip(',')) for l in lines if "Iteration" in l][-1]
    display = [int(l.split()[1]) for l in lines if "display" in l][0]

    train_acc = [l for l in lines if "Train net output #0: accuracy" in l]
    test_acc = [l for l in lines if "Test net output #0: accuracy" in l]
    train_loss = [l for l in lines if " loss = " in l]

    train_acc = np.array([float(l.split()[10]) for l in train_acc])
    test_acc = np.array([float(l.split()[10]) for l in test_acc])
    train_loss = np.array([float(l.split()[8]) for l in train_loss])

    train_acc_x = np.arange(0, last_iter+1, display)
    test_acc_x = np.arange(0, last_iter+1, test_interv)

    # Reduce train points. Comment to use all the points (more noisy)
    idxs = (test_acc_x / display).astype(int)
    train_acc_x = train_acc_x[idxs]
    train_acc = train_acc[idxs]
    train_loss = train_loss[idxs]

    gridsize = (1, 2)
    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid(gridsize, (0, 1), colspan=1, rowspan=1)

    ax1.plot(train_acc_x, train_loss, label="train", color="royalblue")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(train_acc_x, train_acc, label="train", color="salmon")
    ax2.plot(test_acc_x, test_acc, label="test", color="mediumseagreen")
    ax2.set_title("Accuracy")
    ax2.legend()
    plt.show()
    fig.savefig("training.png")
