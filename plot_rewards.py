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

def plot_profits_and_variance(n_agents, profits, pg, t_max):
    profits = pg(profits)
    start_t = t_max - t_max // 10
    end_t = t_max
    pg_end = profits[:, start_t:end_t].mean()
    for i in range(n_agents):
        r_series = pd.Series(profits[i, :]).rolling(window=1000)#.ewm(span=np.max(t_max) // 10)
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
    profit_variance = r_series.std() ** 2
    profit_variance_d = np.zeros_like(profit_variance)
    for i in range(1, len(profit_variance)):
        profit_variance_d[i] = (profit_variance[i] - profit_variance[i - 1]) / profit_variance[i - 1]
    plt.plot(profit_variance_d)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot profits")
    parser.add_argument("--seeds", type=int, help="Random seeds", nargs="+")
    parser.add_argument("--out_dir", type=str, help="Random seed")
    parser.add_argument("--t_max", type=int)
    parser.add_argument("--plot_intermediate", action="store_const", const=True, default=False)
    args = parser.parse_args()
    out_dir = args.out_dir
    t_max = args.t_max
    nash_price = 1.4729273733327568
    coop_price = 1.9249689958811602
    nash = 0.22292696
    coop = 0.33749046
    def pg(x):
        return (x - nash) / (coop - nash)
    n_agents = 2
    mpl.rcParams["axes.prop_cycle"] = cycler(color=["b", "r", "g", "y"])
    r = []
    for seed in args.seeds:
        profits_cur = np.load(f"{out_dir}/session_prices_{seed}.npy")
        if np.min(profits_cur) > 1: # It's prices
            profits_cur = np.apply_along_axis(Pi, 0, profits_cur)
        if args.plot_intermediate:
            plot_profits_and_variance(n_agents, profits_cur, pg, args.t_max)
        r.append(profits_cur)
    profits = np.stack(r, axis=0).mean(axis=0)
    plot_profits_and_variance(n_agents, profits, pg, args.t_max)
    print(profits.shape)
    profit_gains = np.apply_along_axis(pg, 0, profits)
    pg_series = pd.Series(profit_gains.mean(axis=0)).rolling(window=1000)
    plt.plot(pg_series.mean(), "b-")
    plt.fill_between(
        range(len(pg_series.mean())),
        pg_series.mean() - pg_series.std(),
        pg_series.mean() + pg_series.std(),
        alpha=0.2,
    )
    plt.axhline(0)
    plt.axhline(1)
    plt.show()

if __name__ == "__main__":
    main()
