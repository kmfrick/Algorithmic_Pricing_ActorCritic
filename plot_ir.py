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
import torch.nn as nn

from cycler import cycler

from main import (
    SquashedGaussianMLPActor,
    MLPQFunction,
    MLPActorCritic,
    scale_price,
)
from utils import df, grad_desc, Pi

def main():
    parser = argparse.ArgumentParser(description="Plot impulse responses")
    parser.add_argument("--seeds", type=int, help="Random seeds", nargs="+")
    parser.add_argument("--t_max", type=int, help="Time periods elapsed")
    parser.add_argument("--out_dir", type=str, help="Directory")
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--start_prof_gains", type=float)
    parser.add_argument("--actor_hidden_size", type=int)
    parser.add_argument("--plot_intermediate", action="store_const", const=True, default=False)
    parser.add_argument("--defect_to_nash", action="store_const", const=True, default=False)
    parser.add_argument("--defect_to_c", action="store_const", const=True, default=False)
    parser.add_argument("--undershoot", action="store_const", const=True, default=False)
    args = parser.parse_args()
    start_prof_gains = args.start_prof_gains
    out_dir = args.out_dir
    t_max = args.t_max
    nash_price = 1.4729273733327568
    coop_price = 1.9249689958811602
    min_price = nash_price - 0.1
    max_price = coop_price + 0.1
    nash = 0.22292696
    coop = 0.33749046
    n_agents = 2
    c = 1
    mpl.rcParams["axes.prop_cycle"] = cycler(color=["b", "r", "g", "y"])

    ir_periods = 30
    dev_t = ir_periods // 2
    avg_dev_gain = 0
    avg_dev_diff_profit = 0
    for seed in args.seeds:
        ir_arrays_compliant = []
        ir_arrays_defector = []
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        actor = []
        for i in range(n_agents):
            a = SquashedGaussianMLPActor(n_agents, args.actor_hidden_size, nn.Tanh)
            a.load_state_dict(torch.load(f"{out_dir}/actor_weights_{seed}_t{t_max}_agent{i}.pth",  map_location=torch.device('cpu')))
            actor.append(a)
        ir_profit_periods = 1000
        convergence_arrays = np.zeros([n_agents, ir_profit_periods])
        with torch.inference_mode():
            for j in range(n_agents):
                # Impulse response
                price_history = np.zeros([n_agents, ir_profit_periods])
                state = torch.zeros(n_agents)
                for i in range(0, n_agents):
                    state[i] = (coop_price - nash_price) * start_prof_gains + nash_price
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
                        nondev_profit += Pi(price.numpy())[j] * args.discount ** (t - ir_periods / 2)
                    price_history[:, t] = price.detach()
                    state = price
                    conv_price = price.clone().numpy()
                    avg_rew = Pi(conv_price)

                convergence_arrays = price_history.copy()
                # Now compute deviation profits
                dev_profit = 0
                state = initial_state.clone()
                for t in range(ir_profit_periods):
                    for i in range(n_agents):
                        action, _ = actor[i](state.unsqueeze(0), deterministic=True, with_logprob=False)
                        price[i] = scale_price(action, min_price, max_price)
                        orig_price = price[j].item()
                    if t == (ir_periods / 2):
                        if args.defect_to_nash:
                            br = nash_price
                        elif args.defect_to_c:
                            br = c
                        else:
                            br = grad_desc(Pi, price.clone().numpy(), j)
                        if args.undershoot:
                            br = orig_price + (br - orig_price) / 2
                        price[j] = torch.tensor(br)
                    if t >= (ir_periods / 2):
                        dev_profit += Pi(price.numpy())[j] * args.discount ** (t - ir_periods / 2)
                    price_history[:, t] = price
                    state = price
                dev_gain = (dev_profit / nondev_profit - 1) * 100
                avg_dev_gain += dev_gain
                dev_diff_profit = (np.apply_along_axis(Pi, 0, price_history).T - avg_rew).T.sum(axis=1)[j]
                avg_dev_diff_profit += dev_diff_profit
                print(f"Deviation differential profits = {dev_diff_profit} (non-dev is 0)")
                print(
                    f"Non-deviation discounted profits = {nondev_profit:.3f}; Deviation profits = {dev_profit:.3f}; Deviation gain = {dev_gain:.3f}%"
                )
                if args.plot_intermediate:
                    for i in range(n_agents):
                        plt.scatter(list(range(ir_periods)), price_history[i, :ir_periods])
                    plt.legend(leg)
                    for i in range(n_agents):
                        plt.plot(price_history[i, :ir_periods])
                    plt.ylim(min_price, max_price)
                    plt.show()
                ir_arrays_defector.append(price_history[j, :ir_periods])
                for i in range(n_agents):
                    if i != j:
                        ir_arrays_compliant.append(price_history[i, :ir_periods])
    avg_dev_gain /= (n_agents * len(args.seeds))
    avg_dev_diff_profit /= (n_agents * len(args.seeds))
    print(f"Average deviation gain: {avg_dev_gain:.3f}%")
    print(f"Average deviation differential profit: {avg_dev_diff_profit:.3f} (non-deviation is 0)")
    ir_stack_compliant = np.stack(ir_arrays_compliant, axis = 0)
    ir_stack_defector = np.stack(ir_arrays_defector, axis = 0)
    ir_box_compliant = [ir_stack_compliant[:, t] - ir_stack_compliant[:, dev_t - 1] for t in range(dev_t - 1, ir_periods)]
    ir_box_defector = [ir_stack_defector[:, t] - ir_stack_defector[:, dev_t - 1] for t in range(dev_t - 1, ir_periods)]
    plt.boxplot(ir_box_compliant)
    plt.show()
    plt.boxplot(ir_box_defector)
    plt.show()
    ir_mean_compliant = ir_stack_compliant.mean(axis=0)
    ir_mean_defector = ir_stack_defector.mean(axis=0)

    ir_mean_compliant = ir_mean_compliant[(dev_t - 1):]
    ir_mean_defector = ir_mean_defector[(dev_t - 1):]

    leg = ["Non-deviating agent", "Deviating agent"]
    plt.plot(ir_mean_compliant, "bo-")
    plt.plot(ir_mean_defector, "ro-")
    plt.legend(leg)
    plt.ylim(min_price, max_price)
    plt.axhline(nash_price)
    plt.axhline(coop_price)
    plt.show()

    plt.show()


if __name__ == "__main__":
    main()
