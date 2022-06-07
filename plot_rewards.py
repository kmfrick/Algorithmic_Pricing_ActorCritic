#!/usr/bin/env python3

import argparse
import itertools
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cycler import cycler

from utils import Pi

def main():
    parser = argparse.ArgumentParser(description="Plot impulse responses")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--out_dir", type=str, help="Random seed")
    parser.add_argument("--t_max", type=int)
    args = parser.parse_args()
    seed = args.seed
    out_dir = args.out_dir
    t_max = args.t_max
    nash_price = 1.4729273733327568
    coop_price = 1.9249689958811602
    nash = 0.22292696
    coop = 0.33749046
    N_AGENTS = 2
    mpl.rcParams["axes.prop_cycle"] = cycler(color=["b", "r", "g", "y"])

    def pg(x):
        return (x - nash) / (coop - nash)
    rewards = np.load(f"{out_dir}/session_reward_{seed}.npy")
    # Post processing
    if np.min(rewards) > 1: # It's prices
        rewards = np.apply_along_axis(Pi, 0, rewards)
    rewards = pg(rewards)
    start_t = t_max - t_max // 10
    end_t = t_max
    pg_end = rewards[:, start_t:end_t].mean()
    # Plot
    for i in range(N_AGENTS):
        r_series = pd.Series(rewards[i, :]).ewm(span=np.max(t_max) // 10)
        plt.plot(r_series.mean())
        plt.fill_between(
            range(len(r_series.mean())),
            r_series.mean() - r_series.std(),
            r_series.mean() + r_series.std(),
            alpha=0.2,
        )
    plt.axhline(0)
    plt.axhline(1)
    plt.show()


if __name__ == "__main__":
    main()
