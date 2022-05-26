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
