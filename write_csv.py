#!/usr/bin/env python3

import torch
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

from cycler import cycler

from utils import Pi, scale_price, grad_desc
from main import SquashedGaussianMLPActor

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
    min_price = nash_price - 0.1
    max_price = coop_price + 0.1
    n_agents = 2
    FNAME = "experiments.csv"

    def pg(x):
        return (x - nash) / (coop - nash)
    df = pd.read_csv(FNAME, index_col=0)
    rewards = np.load(f"{out_dir}/session_reward_{seed}.npy")
    # Post processing
    pivot = out_dir.find("lr")
    exp_name = out_dir[:pivot]
    param_str = out_dir[pivot:]
    print(out_dir)
    if np.min(rewards) > 1: # It's prices
        rewards = np.apply_along_axis(Pi, 0, rewards)
    rewards = pg(rewards)
    start_t = t_max - t_max // 10
    end_t = t_max
    pg_end = rewards[:, start_t:end_t].mean()
    print(f"Average profit gains at the end: {pg_end}")
    # Get parameters
    params = ["".join(filter(str.isdigit, i)) for i in param_str.split("_") if "-" not in i]
    params = [i for i in params if len(i) > 0]
    name_params = ["".join(filter(str.isalpha, i)) for i in param_str.split("_") if "-" not in i]
    name_params = [i for i in name_params if len(i) > 0]
    lrs = [[float("".join(filter(lambda x : not str.isalpha(x), i))) for i in j.split("-")] for j in param_str.split("_") if "lr" in j][0]
    print(name_params)
    print(params)
    print(lrs)
    # IR
    ir_periods = 60
    ir_arrays_compliant = []
    ir_arrays_defector = []
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    actor = [
        torch.load(
            f"{out_dir}/actor_weights_{seed}_t{t_max}_agent{i}.pth",
            map_location=torch.device(device),
        )
        for i in range(n_agents)
    ]
    print("Computing IRs...")
    ir_profit_periods = 1000
    avg_dev_gain = 0
    DISCOUNT = 0.95
    with torch.inference_mode():
        for j in range(n_agents):
            # Impulse response
            price_history = np.zeros([n_agents, ir_profit_periods])
            state = torch.zeros(n_agents)
            for i in range(0, n_agents):
                state[i] = (coop_price - nash_price) * pg_end + nash_price
            print(f"Initial state = {state}")
            price = state.clone()
            initial_state = state.clone()
            # First compute non-deviation profits
            nondev_profit = 0
            leg = ["Non-deviating agent"] * n_agents
            leg[j] = "Deviating agent"
            for t in range(ir_profit_periods):
                for i in range(n_agents):
                    action, _ = actor[i](state.unsqueeze(0), deterministic=True, with_logprob=False)
                    price[i] = scale_price(action, min_price, max_price)
                if t >= (ir_periods / 2):
                    nondev_profit += Pi(price.numpy())[j] * DISCOUNT ** (t - ir_periods / 2)
                price_history[:, t] = price.detach()
                state = price
            # Now compute deviation profits
            dev_profit = 0
            state = initial_state.clone()
            for t in range(ir_profit_periods):
                for i in range(n_agents):
                    action, _ = actor[i](state.unsqueeze(0), deterministic=True, with_logprob=False)
                    price[i] = scale_price(action, min_price, max_price)
                if t == (ir_periods / 2):
                    br = grad_desc(Pi, price.numpy(), j)
                    price[j] = torch.tensor(br)
                if t >= (ir_periods / 2):
                    dev_profit += Pi(price.numpy())[j] * DISCOUNT ** (t - ir_periods / 2)
                price_history[:, t] = price
                state = price
            dev_gain = (dev_profit / nondev_profit - 1) * 100
            avg_dev_gain += dev_gain
            print(
                f"Non-deviation profits = {nondev_profit:.3f}; Deviation profits = {dev_profit:.3f}; Deviation gain = {dev_gain:.3f}%"
            )
    df.loc[len(df)] = {
        "id": len(df),
        "experiment_name": exp_name,
        "seed": int(seed),
        "buf_size": int(params[name_params.index("buf")]),
        "actor_lr": float(lrs[0]),
        "critic_lr": float(lrs[1]),
        "profit_gain": pg_end,
        "t": t_max,
        "deviation_profit_percent": dev_gain / n_agents,
    }
    df.to_csv(FNAME)


if __name__ == "__main__":
    main()
