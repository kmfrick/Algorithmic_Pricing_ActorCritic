#!/usr/bin/env python3

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cycler import cycler

from main import MLPActorCritic, scale_price


def save_state_action_map(actor, n_agents, c):
    grid_size = 100
    w = torch.linspace(c, c + 1, grid_size, requires_grad=False)
    A = [torch.zeros([grid_size, grid_size], requires_grad=False) for i in range(n_agents)]
    for i in range(n_agents):
        for ai, p1 in enumerate(w):
            for aj, p2 in enumerate(w):
                state = torch.tensor([[p1, p2]])
                a = scale_price(actor[i].act(state, deterministic=True), c)
                A[i][ai, aj] = a
    return A


def plot_heatmap(arr, title, c=1, w=100):
    for i, a in enumerate(arr):
        ax = plt.subplot()
        im = ax.imshow(
            a.reshape([w, w]),
            cmap="Reds",
            extent=(c, 2 * c, c, 2 * c),
            aspect="auto",
            origin="lower",
        )
        ax.set_xlabel("p0")
        ax.set_ylabel("p1")
        plt.colorbar(im)
        ax.set_title(f"{title}; Agent {i}")
        plt.show()


def main():
    nash_price = 1.4729273733327568
    coop_price = 1.9249689958811602
    nash = 0.22292696
    coop = 0.33749046
    n_agents = 2
    c = 1
    HIDDEN_SIZE = 256
    mpl.rcParams["axes.prop_cycle"] = cycler(color=["b", "r", "g", "y"])
    root_dir = sys.argv[1]
    fname = "actions_final"
    SEEDS = [12345, 54321, 464738, 250917]
    ir_periods = 20
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    actor = [
        MLPActorCritic(n_agents, 1, hidden_sizes=(HIDDEN_SIZE,) * n_agents) for i in range(n_agents)
    ]
    for fpostfix in SEEDS:
        out_dir = f"{root_dir}/weights_{fpostfix}"
        MAX_T = int(1e2)
        for i, a in enumerate(actor):
            a.pi.load_state_dict(
                torch.load(
                    f"{out_dir}/actor_weights_{fpostfix}_t{MAX_T-1}_agent{i}.pth",
                    map_location=torch.device(device),
                )
            )
        arr = save_state_action_map(actor, n_agents, c)
        # plot_heatmap(arr, f"Actions for seed {fpostfix}")
        price_history = np.zeros([n_agents, ir_periods])
        # Impulse response
        state = torch.rand(n_agents).to(device) + c
        price = torch.zeros([n_agents]).to(device)
        for t in range(ir_periods):
            for i in range(n_agents):
                price[i] = scale_price(actor[i].act(state, deterministic=True), c)
            if t == (ir_periods / 2):
                price[0] = nash_price
            price_history[:, t] = price
            # profits = compute_profit(ai, a0, mu, c, price)
            state = price
        for i in range(n_agents):
            plt.scatter(list(range(ir_periods)), price_history[i, :])
        plt.legend([f"p{i}" for i in range(n_agents)])
        for i in range(n_agents):
            plt.plot(price_history[i, :])
        plt.show()


if __name__ == "__main__":
    main()
