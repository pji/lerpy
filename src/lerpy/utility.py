"""
utility
~~~~~~~

Utility functions for the lerpy module.
"""
from functools import wraps
from typing import Callable, TypeVar, Union

import numpy as np
from numpy.typing import NDArray


# Types.
T = TypeVar('T', bound=Union[np.int_, np.float_, np.bool_])
Interpolator = Callable[
    [NDArray[T], NDArray[T], NDArray[np.float_]],
    NDArray[T]
]


# Debugging functions.
def print_array(a: NDArray[T], depth: int = 0, dec_round: int = 4) -> None:
    """Write the values of the given array to stdout.

    :param a: The array to print.
    :param depth: (Optional.) The nesting layer of the current array.
        This is primarily for internal use.
    :param dec_round: (Optional.) The number of characters to display
        when printing an array of floats.
    :return: None.
    :rtype: NoneType

    Usage::

        >>> import numpy as np
        >>>
        >>> a = np.arange(9, dtype=float).reshape((3, 3))
        >>> a /= 3.0
        >>> print_array(a)
        [
            [0.0000, 0.3333, 0.6667],
            [1.0000, 1.3333, 1.6667],
            [2.0000, 2.3333, 2.6667],
        ],

    The `dec_round` parameter allows you to change the number of
    numbers after the decimal point.

        >>> a = np.arange(9, dtype=float).reshape((3, 3))
        >>> a /= 3.0
        >>> print_array(a, dec_round=2)
        [
            [0.00, 0.33, 0.67],
            [1.00, 1.33, 1.67],
            [2.00, 2.33, 2.67],
        ],

    """
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
def preserves_type(fn: Interpolator) -> Interpolator:
    """Ensure the datatype of the result is the same as the
    first parameter.
    """
    @wraps(fn)
    def wrapper(a: np.ndarray, *args, **kwargs) -> np.ndarray:
        a_dtype = a.dtype
        result = fn(a, *args, **kwargs)
        return result.astype(a_dtype)
    return wrapper
