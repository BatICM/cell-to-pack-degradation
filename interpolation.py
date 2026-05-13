# -*- coding: utf-8 -*-
"""
Interpolation helpers for voltage-aligned battery samples.
"""

from typing import Callable

import numpy as np
from scipy.interpolate import interp1d


def enforce_monotonicity(x: np.ndarray, increasing: bool = True, eps: float = 1e-6) -> np.ndarray:
    """Make a 1D sequence strictly monotonic."""
    x = np.asarray(x, dtype=float).copy()

    if len(x) <= 1:
        return x

    if increasing:
        for i in range(1, len(x)):
            if x[i] <= x[i - 1]:
                x[i] = x[i - 1] + eps
    else:
        for i in range(len(x) - 2, -1, -1):
            if x[i] <= x[i + 1]:
                x[i] = x[i + 1] - eps

    return x


def custom_interp1d(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Linear interpolation with nearest-boundary values outside the data range."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    f = interp1d(
        x,
        y,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
        assume_sorted=False,
    )

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = y[0], y[-1]

    def interp_fn(x_new: np.ndarray) -> np.ndarray:
        x_new = np.asarray(x_new, dtype=float)
        y_new = f(x_new)

        y_new[x_new < x_min] = y_min
        y_new[x_new > x_max] = y_max

        return y_new

    return interp_fn