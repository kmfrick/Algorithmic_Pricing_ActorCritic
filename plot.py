#!/usr/bin/env python3

import itertools
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cycler import cycler

from main import SquashedGaussianMLPActor, MLPQFunction, MLPValueFunction, MLPActorCritic, scale_price


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


# https://cs231n.github.io/optimization-1/#gradcompute
def df(f, x):
    """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """
    fx = f(x)  # evaluate function value at original point
    grad = np.zeros(x.shape)
    h = 0.00001
    # iterate over all indexes in x
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h  # increment by h
        fxh = f(x)  # evalute f(x + h)
        x[ix] = old_value  # restore to previous value (very important!)
        x[ix] = old_value - h
        fxmh = f(x)
        x[ix] = old_value  # restore to previous value (very important!)
        # compute the partial derivative
        grad[ix] = (fxh[ix] - fxmh[ix]) / (2 * h)  # the slope
        it.iternext()  # step to next dimension
    return grad


def grad_desc(f, x, i, thresh=1e-7, lr=1e-4):
    while np.abs(df(f, x))[i] > thresh:
        x[i] += lr * df(f, x)[i]
    return x[i]


def main():
    nash_price = 1.4729273733327568
    coop_price = 1.9249689958811602
    nash = 0.22292696
    coop = 0.33749046
    N_AGENTS = 2
    STARTING_PROFIT_GAIN = 0.9
    SEEDS = [250917]
    T_MAX = [10000, 90000]
    mpl.rcParams["axes.prop_cycle"] = cycler(color=["b", "r", "g", "y"])
    root_dir = sys.argv[1]

    ir_periods = 40
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    ai = 2.0
    a0 = 0
    mu = 0.25
    c = 1

    def Pi(p):
        q = np.exp((ai - p) / mu) / (np.sum(np.exp((ai - p) / mu)) + np.exp(a0 / mu))
        pi = (p - c) * q
        return pi

    for seed in SEEDS:
        out_dir = f"{root_dir}"
        rewards = np.load(f"{out_dir}/session_reward_{seed}.npy")
        for i in range(N_AGENTS):
            r_series = pd.Series(rewards[i, :]).ewm(span=np.max(T_MAX)//50)
            plt.plot(r_series.mean())
            plt.fill_between(range(len(r_series.mean())), r_series.mean() - r_series.std(), r_series.mean() + r_series.std(), alpha=0.2)
        plt.axhline(nash)
        plt.axhline(coop)
        plt.show()
    with torch.no_grad():
        for seed, t_max in itertools.product(SEEDS, T_MAX):
            np.random.seed(seed)
            torch.manual_seed(seed)
            actor = [torch.load(f"{out_dir}/actor_weights_{seed}_t{t_max}_agent{i}.pth", map_location=torch.device(device)) for i in range(N_AGENTS)]
            print(actor)
            # Create and plot state-action map
            grid_size = 100
            w = torch.linspace(c, 2 * c, grid_size, requires_grad=False)
            A = [torch.zeros([grid_size, grid_size], requires_grad=False) for i in range(N_AGENTS)]
            print("Computing state-action map...")
            for i in range(N_AGENTS):
                for a_i, p1 in enumerate(w):
                    for a_j, p2 in enumerate(w):
                        state = torch.tensor([[p1, p2]])
                        a = scale_price(actor[i](state)[0], c).detach()
                        print(f"{state} -> {a}")
                        A[i][a_i, a_j] = a
            plot_heatmap(A, f"Actions for seed {seed}")
            print("Computing IRs...")
            # Impulse response
            price_history = np.zeros([N_AGENTS, ir_periods])
            state = torch.rand(N_AGENTS).to(device) * c + c
            for i in range(0, N_AGENTS):
                state[i] = (coop_price - nash_price) * STARTING_PROFIT_GAIN + nash_price
            print(f"Initial state = {state}")
            price = state.clone()
            initial_state = state.clone()
            # First compute non-deviation profits
            DISCOUNT = 0.99
            nondev_profit = 0
            j = 1
            leg = ["Non-deviating agent"] * N_AGENTS
            leg[j] = "Deviating agent"
            for t in range(ir_periods):
                for i in range(N_AGENTS):
                    price[i] = scale_price(actor[i](state.unsqueeze(0))[0], c)
                if t >= (ir_periods / 2):
                    nondev_profit += Pi(price.numpy())[j] * DISCOUNT ** (t - ir_periods / 2)
                price_history[:, t] = price.detach()
                state = price
            # Now compute deviation profits
            dev_profit = 0
            state = initial_state.clone()
            for t in range(ir_periods):
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
            print(f"Non-deviation profits = {nondev_profit:.3f}; Deviation profits = {dev_profit:.3f}; Deviation gain = {dev_gain:.3f}%")
            for i in range(N_AGENTS):
                plt.scatter(list(range(ir_periods)), price_history[i, :])
            plt.legend(leg)
            for i in range(N_AGENTS):
                plt.plot(price_history[i, :])
            plt.show()


if __name__ == "__main__":
    main()
