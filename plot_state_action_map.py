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


def plot_heatmap(arr, title, c=1, w=100):
    plt.tight_layout()
    for i, a in enumerate(arr):
        ax = plt.subplot(1, len(arr), i + 1)
        im = plt.imshow(
            a.reshape([w, w]),
            cmap="Reds",
            extent=(c, 2 * c, c, 2 * c),
            aspect="auto",
            origin="lower",
        )
        ax.set_xlabel("p0")
        ax.set_ylabel("p1")
        ax.set_title(f"{title}; Agent {i}")
        plt.colorbar(orientation="horizontal")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot impulse responses")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--t_max", type=int, help="Time periods elapsed")
    parser.add_argument("--out_dir", type=str, help="Directory")
    args = parser.parse_args()
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
    ai = 2.0
    a0 = 0
    mu = 0.25
    c = 1

    def Pi(p):
        q = np.exp((ai - p) / mu) / (np.sum(np.exp((ai - p) / mu)) + np.exp(a0 / mu))
        pi = (p - c) * q
        return pi

    actor = [
        torch.load(
            f"{out_dir}/actor_weights_{seed}_t{t_max}_agent{i}.pth",
            map_location=torch.device(device),
        )
        for i in range(N_AGENTS)
    ]
    with torch.inference_mode():
        # Create and plot state-action map
        grid_size = 50
        w = torch.linspace(c, 2 * c, grid_size, requires_grad=False)
        A = [torch.zeros([grid_size, grid_size], requires_grad=False) for i in range(N_AGENTS)]
        print("Computing state-action map...")
        for i in range(N_AGENTS):
            for a_i, p1 in enumerate(w):
                for a_j, p2 in enumerate(w):
                    state = torch.tensor([[p1, p2]])
                    a = scale_price(actor[i](state)[0], c).detach()
                    # print(f"{state} -> {a}")
                    A[i][a_i, a_j] = a
        plot_heatmap(A, f"Actions for seed {seed}", w=grid_size)


if __name__ == "__main__":
    main()
