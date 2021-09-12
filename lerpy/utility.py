"""
utility
~~~~~~~

Utility functions for the lerpy module.
"""
from functools import wraps
from typing import Callable

import numpy as np


# Debugging functions.
def print_array(a: np.ndarray, depth: int = 0, dec_round: int = 4) -> None:
    """Write the values of the given array to stdout."""
    if len(a.shape) > 1:
        print(' ' * (4 * depth) + '[')
        for i in range(a.shape[0]):
            print_array(a[i], depth + 1, dec_round)
        print(' ' * (4 * depth) + '],')

    else:
        if a.dtype == np.float32 or a.dtype == np.float64:
            tmp = '{:>1.' + str(dec_round) + 'f}'
        else:
            tmp = '{}'
        nums = [tmp.format(n) for n in a]
        print(' ' * (4 * depth) + '[' + ', '.join(nums) + '],')


# Decorators.
def preserves_type(fn: Callable) -> Callable:
    """Ensure the datatype of the result is the same as the
    first parameter.
    """
    @wraps(fn)
    def wrapper(a: np.ndarray, *args, **kwargs) -> np.ndarray:
        a_dtype = a.dtype
        result = fn(a, *args, **kwargs)
        return result.astype(a_dtype)
    return wrapper
