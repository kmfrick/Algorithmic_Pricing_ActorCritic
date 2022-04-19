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

# https://cs231n.github.io/optimization-1/#gradcompute
def df(f, x):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """
  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001
  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)
    x[ix] = old_value - h
    fxmh = f(x)
    x[ix] = old_value # restore to previous value (very important!)
    # compute the partial derivative
    grad[ix] = (fxh[ix] - fxmh[ix]) / (2*h) # the slope
    it.iternext() # step to next dimension
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
    HIDDEN_SIZE = 256
    MAX_T = int(1e5)
    STARTING_PROFIT_GAIN = 0.9
    SEEDS = [250917]
    mpl.rcParams["axes.prop_cycle"] = cycler(color=["b", "r", "g", "y"])
    root_dir = sys.argv[1]

    ir_periods = 40
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    actor = [
        MLPActorCritic(N_AGENTS, 1, hidden_sizes=(HIDDEN_SIZE,) * N_AGENTS) for i in range(N_AGENTS)
    ]
    ai = 2.0
    a0 = 0
    mu = 0.25
    c = 1

    def Pi(p):
        q = np.exp((ai - p) / mu) / (np.sum(np.exp((ai - p) / mu)) + np.exp(a0 / mu))
        pi = (p - c) * q
        return pi

    for fpostfix in SEEDS:
        out_dir = f"{root_dir}/weights_{fpostfix}"

        for i, a in enumerate(actor):
            a.pi.load_state_dict(
                torch.load(
                    f"{out_dir}/actor_weights_{fpostfix}_t{MAX_T-1}_agent{i}.pth",
                    map_location=torch.device(device),
                )
            )
        # Create and plot state-action map
        grid_size = 100
        w = torch.linspace(c, 2 * c, grid_size, requires_grad=False)
        A = [torch.zeros([grid_size, grid_size], requires_grad=False) for i in range(N_AGENTS)]
        for i in range(N_AGENTS):
            for a_i, p1 in enumerate(w):
                for a_j, p2 in enumerate(w):
                    state = torch.tensor([[p1, p2]])
                    a = scale_price(actor[i].act(state, deterministic=True), c)
                    A[i][a_i, a_j] = a
        plot_heatmap(A, f"Actions for seed {fpostfix}")
        # Impulse response
        price_history = np.zeros([N_AGENTS, ir_periods])
        state = torch.rand(N_AGENTS).to(device) * c + c
        for i in range(0, N_AGENTS):
            state[i] = (coop_price - nash_price) * STARTING_PROFIT_GAIN + nash_price
        price = state.clone()
        initial_state = state.clone()
        # First compute non-deviation profits
        DISCOUNT = 0.99
        nondev_profit = 0
        for t in range(ir_periods):
            for i in range(N_AGENTS):
                price[i] = scale_price(actor[i].act(state, deterministic=True), c)
            if t >= (ir_periods / 2):
                nondev_profit += Pi(price.numpy())[0] * DISCOUNT ** (t - ir_periods/2)
            price_history[:, t] = price
            state = price
        # Now compute deviation profits
        dev_profit = 0
        state = initial_state.clone()
        for t in range(ir_periods):
            for i in range(N_AGENTS):
                price[i] = scale_price(actor[i].act(state, deterministic=True), c)
            if t == (ir_periods / 2):
                price[0] = torch.tensor(grad_desc(Pi, price.numpy(), 0))
            if t >= (ir_periods / 2):
                dev_profit += Pi(price.numpy())[0] * DISCOUNT ** (t - ir_periods/2)
            price_history[:, t] = price
            state = price
        print(f"Non-deviation profits = {nondev_profit:.3f}; Deviation profits = {dev_profit:.3f}")
        for i in range(N_AGENTS):
            plt.scatter(list(range(ir_periods)), price_history[i, :])
        plt.legend(["Deviating agent", "Non-deviating agent"])
        for i in range(N_AGENTS):
            plt.plot(price_history[i, :])
        plt.show()


if __name__ == "__main__":
    main()
