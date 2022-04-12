import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pandas as pd
import sys
import os

def plot_reward_array(rewards, nash, coop, title, rolling_interval, fnames, points = False):
    tot0 = np.sum(np.array([r[0] for r in rewards]), axis = 0) / len(rewards)
    tot1 = np.sum(np.array([r[1] for r in rewards]), axis = 0) / len(rewards)
    tot0_series = pd.Series(tot0).rolling(rolling_interval).mean()
    tot1_series = pd.Series(tot1).rolling(rolling_interval).mean()
    if (points):
        plt.scatter(list(range(len(tot0_series))), tot0_series)
        plt.scatter(list(range(len(tot1_series))), tot1_series)
    plt.plot(tot0_series)
    plt.plot(tot1_series)
    plt.axhline(nash)
    plt.axhline(coop)
    plt.title(f"Average {title}")
    plt.show()
    for i, r in enumerate(rewards):
        r0_series = pd.Series(r[0]).rolling(rolling_interval).mean()
        r1_series = pd.Series(r[1]).rolling(rolling_interval).mean()
        if (points):
            plt.scatter(list(range(len(r0_series))), r0_series)
            plt.scatter(list(range(len(r1_series))), r1_series)
        plt.plot(r0_series)
        plt.plot(r1_series)
        plt.axhline(nash)
        plt.axhline(coop)
        plt.title(f"Session {i} {title}: {fnames[i]}")
        plt.show()

def main():
    nash_price = 1.4729273733327568
    coop_price = 1.9249689958811602
    nash = 0.22292696
    coop = 0.33749046
    mpl.rcParams['axes.prop_cycle'] = cycler(color=['b', 'r', 'g', 'y'])
    out_dir = sys.argv[1]
    files = os.listdir(out_dir)
    rewards = [np.load(f"{out_dir}/{f}") for f in files if "session" in f]
    ir_profits = [np.load(f"{out_dir}/{f}") for f in files if "ir_profits" in f]
    ir_prices = [np.load(f"{out_dir}/{f}") for f in files if "ir_prices" in f]
    plot_reward_array(rewards, nash, coop, "reward", 100, [f for f in files if "session" in f])
    plot_reward_array(ir_profits, nash, coop, "IR", 1, [f for f in files if "ir_profits" in f], points=True)
    plot_reward_array(ir_prices, nash_price, coop_price, "IR Prices", 1, [f for f in files if "ir_prices" in f], points=True)


if __name__ == "__main__":
    main()
