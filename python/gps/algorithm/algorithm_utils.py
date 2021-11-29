""" This file defines utility classes and functions for algorithms. """
from typing import Optional

import numpy as np


def compute_step_schedule_exp(num_iterations: int, y: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
    return np.exp(compute_step_schedule_lin(num_iterations, np.log(y), x))


def compute_step_schedule_lin(num_iterations: int, y: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
    if x is None:
        # Assume equal distribution
        x = np.linspace(0, 1, len(y)) * num_iterations
    else:
        assert np.amin(x) == 0, "x has to start with 0"
        assert np.amax(x) == num_iterations, f"x has to end with {num_iterations - 1}"
    return np.interp(np.arange(num_iterations), x, y)
