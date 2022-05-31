#!/usr/bin/env python3

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cycler import cycler


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
        ret = it.iternext()  # step to next dimension
        if not ret:
            break
    return grad


def grad_desc(f, x, i, thresh=1e-5, lr=1e-4, maxit=int(1e5)):
    it = 0
    while np.abs(df(f, x))[i] > thresh and it < maxit:
        x[i] += lr * df(f, x)[i]
        it += 1
    return x[i]


def scale_price(price, c, d=None):
    if d is None:
        return price * c + c
    else:
        if c > d:
            c, d = d, c
        return price * (d - c) + c


def impulse_response(n_agents, agents, price, ir_periods, c, Pi):
    avg_dev_gain = 0
    with torch.no_grad():
        # Impulse response
        state = price.squeeze().clone().detach()
        print(f"Initial state = {state}")
        initial_state = state.clone()
        ir_profit_periods = 1000
        for j in range(n_agents):
            # Impulse response
            price = state.clone()
            # First compute non-deviation profits
            DISCOUNT = 0.99
            nondev_profit = 0
            for t in range(ir_profit_periods):
                for i in range(n_agents):
                    price[i] = scale_price(agents[i].act(state.unsqueeze(0))[0], c)
                if t >= (ir_periods / 2):
                    nondev_profit += Pi(price.cpu().numpy())[j] * DISCOUNT ** (
                            t - ir_periods / 2
                    )
                state = price
            # Now compute deviation profits
            dev_profit = 0
            state = initial_state.clone()
            for t in range(ir_profit_periods):
                for i in range(n_agents):
                    price[i] = scale_price(agents[i].act(state.unsqueeze(0))[0], c)
                if t == (ir_periods / 2):
                    br = grad_desc(Pi, price.cpu().numpy(), j)
                    price[j] = torch.tensor(br)
                if t >= (ir_periods / 2):
                    dev_profit += Pi(price.cpu().numpy())[j] * DISCOUNT ** (t - ir_periods / 2)
                state = price
            dev_gain = (dev_profit / nondev_profit - 1) * 100
            avg_dev_gain += dev_gain
            print(
                f"Agent {j}: Non-deviation profits = {nondev_profit:.3f}; Deviation profits = {dev_profit:.3f}; Deviation gain = {dev_gain:.3f}%"
            )
    return avg_dev_gain