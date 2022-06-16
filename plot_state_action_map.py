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


def plot_heatmap(arr, title, min_price, max_price, w=100):
    plt.tight_layout()
    for i, a in enumerate(arr):
        ax = plt.subplot(1, len(arr), i + 1)
        im = plt.imshow(
            a.reshape([w, w]),
            cmap="Reds",
            extent=(min_price, max_price, min_price, max_price),
            aspect="auto",
            origin="lower",
        )
        ax.set_xlabel("p0")
        ax.set_ylabel("p1")
        ax.set_title(f"{title}; Agent {i}")
        plt.colorbar(orientation="horizontal")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot state-action maps")
    parser.add_argument("--seeds", type=int, help="Random seeds", nargs="+")
    parser.add_argument("--t_max", type=int, help="Time periods elapsed")
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--out_dir", type=str, help="Directory")
    parser.add_argument("--actor_hidden_size", type=int)
    parser.add_argument("--plot_intermediate", action="store_const", const=True, default=False)
    args = parser.parse_args()
    grid_size = args.grid_size
    out_dir = args.out_dir
    t_max = args.t_max
    nash_price = 1.4729273733327568
    coop_price = 1.9249689958811602
    min_price = nash_price - 0.1
    max_price = coop_price + 0.1
    nash = 0.22292696
    coop = 0.33749046
    n_agents = 2
    mpl.rcParams["axes.prop_cycle"] = cycler(color=["b", "r", "g", "y"])

    ir_periods = 60
    ir_arrays_compliant = []
    ir_arrays_defector = []
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    ai = 2.0
    a0 = 0
    mu = 0.25
    c = 1

    def Pi(p):
        q = np.exp((ai - p) / mu) / (np.sum(np.exp((ai - p) / mu)) + np.exp(a0 / mu))
        pi = (p - c) * q
        return pi

    state_action_maps = []
    for seed in args.seeds:
        actor = []
        for i in range(n_agents):
            a = SquashedGaussianMLPActor(n_agents, args.actor_hidden_size, nn.Tanh)
            a.load_state_dict(torch.load(f"{out_dir}/actor_weights_{seed}_t{t_max}_agent{i}.pth",  map_location=torch.device('cpu')))
            actor.append(a)
        with torch.inference_mode():
            # Create and plot state-action map
            w = torch.linspace(min_price, max_price, grid_size, requires_grad=False)
            A = [np.zeros([grid_size, grid_size]) for i in range(n_agents)]
            print("Computing state-action map...")
            for i in range(n_agents):
                for a_i, p1 in enumerate(w):
                    for a_j, p2 in enumerate(w):
                        state = torch.tensor([[p1, p2]])
                        action, _ = actor[i](state, deterministic=True, with_logprob = False)
                        a = scale_price(action, min_price, max_price).detach()
                        # print(f"{state} -> {a}")
                        A[i][a_i, a_j] = a.numpy()
            state_action_maps.append(A)
            if args.plot_intermediate:
                plot_heatmap(A, f"Actions for seed {seed}", min_price, max_price, w=grid_size)
    average_state_action_map = np.stack(state_action_maps, axis=0).mean(axis=0)
    plot_heatmap(average_state_action_map, f"Mean action across {len(args.seeds)} seeds", min_price, max_price, w=grid_size)


if __name__ == "__main__":
    main()
