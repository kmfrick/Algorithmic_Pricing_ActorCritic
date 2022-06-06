#!/usr/bin/env python3

import argparse
import itertools
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cycler import cycler

from main import (
    SquashedGaussianMLPActor,
    MLPQFunction,
    MLPValueFunction,
    MLPActorCritic,
    scale_price,
)
from utils import df, grad_desc

ai = 2.0
a0 = 0
mu = 0.25
c = 1


def Pi(p):
    q = np.exp((ai - p) / mu) / (np.sum(np.exp((ai - p) / mu)) + np.exp(a0 / mu))
    pi = (p - c) * q
    return pi


def main():
    parser = argparse.ArgumentParser(description="Plot impulse responses")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--t_max", type=int, help="Time periods elapsed")
    parser.add_argument("--out_dir", type=str, help="Directory")
    parser.add_argument("--start_prof_gains", type=float)
    args = parser.parse_args()
    start_prof_gains = args.start_prof_gains
    out_dir = args.out_dir
    t_max = args.t_max
    seed = args.seed
    nash_price = 1.4729273733327568
    coop_price = 1.9249689958811602
    nash = 0.22292696
    coop = 0.33749046
    N_AGENTS = 2
    mpl.rcParams["axes.prop_cycle"] = cycler(color=["b", "r", "g", "y"])

    ir_periods = 60
    ir_arrays_compliant = []
    ir_arrays_defector = []
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    actor = [
        torch.load(
            f"{out_dir}/actor_weights_{seed}_t{t_max}_agent{i}.pth",
            map_location=torch.device(device),
        )
        for i in range(N_AGENTS)
    ]
    print("Computing IRs...")
    ir_profit_periods = 1000
    with torch.inference_mode():
        for j in range(N_AGENTS):
            # Impulse response
            price_history = np.zeros([N_AGENTS, ir_profit_periods])
            state = torch.rand(N_AGENTS).to(device) * c + c
            for i in range(0, N_AGENTS):
                state[i] = (coop_price - nash_price) * start_prof_gains + nash_price
            print(f"Initial state = {state}")
            price = state.clone()
            initial_state = state.clone()
            # First compute non-deviation profits
            DISCOUNT = 0.99
            nondev_profit = 0
            leg = ["Non-deviating agent"] * N_AGENTS
            leg[j] = "Deviating agent"
            for t in range(ir_profit_periods):
                for i in range(N_AGENTS):
                    price[i] = scale_price(actor[i](state.unsqueeze(0))[0], c)
                if t >= (ir_periods / 2):
                    nondev_profit += Pi(price.numpy())[j] * DISCOUNT ** (t - ir_periods / 2)
                price_history[:, t] = price.detach()
                state = price
            # Now compute deviation profits
            dev_profit = 0
            state = initial_state.clone()
            for t in range(ir_profit_periods):
                for i in range(N_AGENTS):
                    price[i] = scale_price(actor[i](state.unsqueeze(0))[0], c)
                if t == (ir_periods / 2):
                    br = grad_desc(Pi, price.numpy(), j)
                    price[j] = torch.tensor(br)
                if t >= (ir_periods / 2):
                    dev_profit += Pi(price.numpy())[j] * DISCOUNT ** (t - ir_periods / 2)
                price_history[:, t] = price
                state = price
            dev_gain = (dev_profit / nondev_profit - 1) * 100
            print(
                f"Non-deviation profits = {nondev_profit:.3f}; Deviation profits = {dev_profit:.3f}; Deviation gain = {dev_gain:.3f}%"
            )
            for i in range(N_AGENTS):
                plt.scatter(list(range(ir_periods)), price_history[i, :ir_periods])
            plt.legend(leg)
            for i in range(N_AGENTS):
                plt.plot(price_history[i, :ir_periods])
            plt.ylim(1.5, 2)
            plt.show()
            ir_arrays_defector.append(price_history[j, :ir_periods])
            for i in range(N_AGENTS):
                if i != j:
                    ir_arrays_compliant.append(price_history[i, :ir_periods])
    ir_arrays_compliant = np.stack(ir_arrays_compliant).mean(axis=0)
    ir_arrays_defector = np.stack(ir_arrays_defector).mean(axis=0)

    leg = ["Non-deviating agent", "Deviating agent"]
    plt.plot(ir_arrays_compliant)
    plt.plot(ir_arrays_defector)
    plt.scatter(list(range(ir_periods)), ir_arrays_compliant)
    plt.scatter(list(range(ir_periods)), ir_arrays_defector)
    plt.legend(leg)
    plt.ylim(1.5, 2)
    plt.show()


if __name__ == "__main__":
    main()
