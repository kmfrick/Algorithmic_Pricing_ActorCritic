import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pandas as pd
import sys
import os

def plot_array_rolling_avg(arr, low_line, high_line, title, rolling_interval, fnames, points = False):
    tot0 = np.sum(np.array([r[0] for r in arr]), axis = 0) / len(arr)
    tot1 = np.sum(np.array([r[1] for r in arr]), axis = 0) / len(arr)
    tot0_series = pd.Series(tot0).rolling(rolling_interval).mean()
    tot1_series = pd.Series(tot1).rolling(rolling_interval).mean()
    if (points):
        plt.scatter(list(range(len(tot0_series))), tot0_series)
        plt.scatter(list(range(len(tot1_series))), tot1_series)
    plt.plot(tot0_series)
    plt.plot(tot1_series)
    plt.axhline(low_line)
    plt.axhline(high_line)
    plt.title(f"Average {title}")
    plt.show()
    for i, r in enumerate(arr):
        r0_series = pd.Series(r[0]).rolling(rolling_interval).mean()
        r1_series = pd.Series(r[1]).rolling(rolling_interval).mean()
        if (points):
            plt.scatter(list(range(len(r0_series))), r0_series)
            plt.scatter(list(range(len(r1_series))), r1_series)
        plt.plot(r0_series)
        plt.plot(r1_series)
        plt.axhline(low_line)
        plt.axhline(high_line)
        plt.title(f"Session {i} {title}: {fnames[i]}")
        plt.show()

def plot_heatmap(arr, title, fnames):
    for i, a in enumerate(arr):
        ax = plt.subplot()
        im = ax.imshow(a.reshape([100, 100]), cmap="Reds", extent=(1, 3, 1, 3), aspect="auto")
        ax.set_xlabel("p1")
        ax.set_ylabel("p2")
        plt.colorbar(im)
        ax.set_title(f"Session {i} {title}: {fnames[i]}")
        plt.show()

def main():
    nash_price = 1.4729273733327568
    coop_price = 1.9249689958811602
    nash = 0.22292696
    coop = 0.33749046
    mpl.rcParams['axes.prop_cycle'] = cycler(color=['b', 'r', 'g', 'y'])
    out_dir = sys.argv[1]
    files = os.listdir(out_dir)
    lr = [f for f in files if "net_lr" in f]
    lr = lr[0]
    plt.plot(np.load(f"{out_dir}/{lr}")[0, :])
    plt.title("LR")
    plt.show()
    rewards = [np.load(f"{out_dir}/{f}") for f in files if "session" in f]
    ir_profits = [np.load(f"{out_dir}/{f}") for f in files if "ir_profits" in f]
    ir_prices = [np.load(f"{out_dir}/{f}") for f in files if "ir_prices" in f]
    grad_norm = [np.load(f"{out_dir}/{f}") for f in files if "grad_norm" in f]
    actions = [np.load(f"{out_dir}/{f}") for f in files if "actions" in f]
    values = [np.load(f"{out_dir}/{f}") for f in files if "values" in f]
    plot_heatmap(actions, "Actions", [f for f in files if "actions" in f])
    plot_heatmap(values, "Values", [f for f in files if "actions" in f])
    plot_array_rolling_avg(rewards, nash, coop, "reward", 100, [f for f in files if "session" in f])
    plot_array_rolling_avg(ir_profits, nash, coop, "IR", 1, [f for f in files if "ir_profits" in f], points=True)
    plot_array_rolling_avg(ir_prices, nash_price, coop_price, "IR Prices", 1, [f for f in files if "ir_prices" in f], points=True)
    plot_array_rolling_avg(grad_norm, 0, 0, "Gradient Norm", 1, [f for f in files if "grad_norm" in f], points=False)



if __name__ == "__main__":
    main()
