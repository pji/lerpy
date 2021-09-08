"""
lerpy
~~~~~

Linear interpolation functions.
"""
import numpy as np


# Public functions.
def linear_interpolation(a: np.ndarray,
                         b: np.ndarray,
                         x: np.ndarray) -> np.ndarray:
    """Perform a linear interpolation on the values of two arrays

    :param a: The "left" values.
    :param b: The "right" values.
    :param x: An array of how close the location of the final value
        should be to the "left" value.
    :return: A :class:ndarray object
    :rtype: numpy.ndarray

    Usage::

        >>> import numpy as np
        >>>
        >>> a = np.array([1, 2, 3])
        >>> b = np.array([3, 4, 5])
        >>> x = np.array([.5, .5, .5])
        >>> linear_interpolation(a, b, x)
        array([2., 3., 4.])
    """
    return a * (1 - x) + b * x


# Function aliases.
lerp = linear_interpolation
