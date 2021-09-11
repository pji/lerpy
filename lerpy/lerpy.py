"""
lerpy
~~~~~

Linear interpolation functions.
"""
import numpy as np

from lerpy.utility import print_array


# Public interpolation functions.
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


def n_dimensional_interpolation(a: np.ndarray,
                                b: np.ndarray,
                                x: np.ndarray) -> np.ndarray:
    """Interpolate the values of each pixel of image data."""
    if len(x) > 1:
        lerped = lerp(a, b, x[-1])
        a = lerped[::2]
        b = lerped[1::2]
        return n_dimensional_interpolation(a, b, x[:-1])

    result = lerp(a, b, x[0])
    return result[0]


# Public utility functions.
def resize_array(a: np.ndarray, size: tuple[int, ...]) -> np.ndarray:
    """Resize an two dimensional array using linear interpolation.

    :param a: The array to resize. The array is expected to have at
        least two dimensions.
    :param size: The shape for the resized array.
    :return: A :class:ndarray object.
    :rtype: numpy.ndarray
    """
    # Perform defensive actions to prevent unneeded processing if
    # the array won't actually change and to make sure any changes
    # to the array won't have unexpected side effects.
    if size == a.shape:
        return a
    a = a.copy()

    # Map out the relationship between the old space and the
    # new space.
    whole, x = _map_resized_array(a, size)
    a, b = _build_sides(a, size, whole)

    # Perform the interpolation using the mapped space and return.
    return n_dimensional_interpolation(a, b, x)


# Private functions.
def _build_sides(a: np.ndarray,
                 size: tuple[int, ...],
                 whole: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the grids used in an n-dimensional interpolation."""
    # Linear interpolation determines the value of a new pixel by
    # comparing the values of the eight old pixels that surround it.
    # The hashes are the keys to the dictionary that contains those
    # old pixel values. The key indicates the position of the pixel
    # on each axis, with one meaning the position is ahead of the
    # new pixel, and zero meaning the position is behind it.
    num_dim = len(size)
    axes = range(num_dim)
    tmp = '{:>0' + str(num_dim) + 'b}'
    hashes = [tmp.format(n)[::-1] for n in range(2 ** num_dim)]
    size_sides = (int(len(hashes) / 2), *size)
    a_sides = np.zeros(size_sides, dtype=a.dtype)
    b_sides = np.zeros(size_sides, dtype=a.dtype)

    # The original array needs to be made one dimensional for the
    # numpy.take operation that will occur as we build the tables.
    orig_shape = a.shape
    raveled = np.ravel(a)

    # Build the table that contains the old pixel values to
    # interpolate.
    for hash in hashes:
        hash_whole = whole.copy()

        # Use the hash key to adjust the which old pixel we are
        # looking at.
        for axis in axes:
            if hash[axis] == '1':
                hash_whole[axis] += 1

                # Handle the pixels that were pushed off the far
                # edge of the original array by giving them the
                # value of the last pixel along that axis in the
                # original array.
                m = np.zeros(hash_whole[axis].shape, dtype=bool)
                m[hash_whole[axis] >= a.shape[axis]] = True
                hash_whole[axis][m] = a.shape[axis] - 1

        # Since numpy.take() only works in one dimension, we need to
        # map the three dimensional indices of the original array to
        # the one dimensional indices used by the raveled version of
        # that array.
        raveled_indices = np.zeros_like(hash_whole[0])
        for axis in axes:
            remaining_axes = range(num_dim)[axis + 1:]
            axis_incr = 1
            for r_axis in remaining_axes:
                axis_incr *= orig_shape[r_axis]
            raveled_indices += hash_whole[axis] * axis_incr

        # Get the value of the pixel in the original array.
        side = np.take(raveled, raveled_indices.astype(int))
        index = int(int(hash, 2) // 2)
        if hash.endswith('0'):
            a_sides[index] = side
        else:
            b_sides[index] = side

    return a_sides, b_sides


def _map_resized_array(a: np.ndarray,
                       size: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Map out how the values of a grid are spread when the grid
    is resized.
    """
    axes = range(len(size))
    indices = np.indices(size)
    new_ends = [s - 1 for s in size]
    old_ends = [s - 1 for s in a.shape]
    true_factors = [n / o for n, o in zip(new_ends, old_ends)]
    whole = indices.copy()
    parts = indices.copy().astype(float)
    for axis in axes:
        whole[axis] = (indices[axis] // true_factors[axis])
        parts[axis] = (indices[axis] / true_factors[axis] - whole[axis])
    return whole, parts


# Function aliases.
lerp = linear_interpolation
nderp = n_dimensional_interpolation

if __name__ == '__main__':
    a = np.array([
        [
            [0.0, 1.0, 2.0, ],
            [1.0, 2.0, 3.0, ],
            [2.0, 3.0, 4.0, ],
        ],
        [
            [1.0, 2.0, 3.0, ],
            [2.0, 3.0, 4.0, ],
            [3.0, 4.0, 5.0, ],
        ],
    ])
    b = np.array([
        [
            [1.0, 2.0, 3.0, ],
            [2.0, 3.0, 4.0, ],
            [3.0, 4.0, 5.0, ],
        ],
        [
            [2.0, 3.0, 4.0, ],
            [3.0, 4.0, 5.0, ],
            [4.0, 5.0, 6.0, ],
        ],
    ])
    x = np.array([
        [
            [0.5, 0.5, 0.5, ],
            [0.5, 0.5, 0.5, ],
            [0.5, 0.5, 0.5, ],
        ],
        [
            [0.5, 0.5, 0.5, ],
            [0.5, 0.5, 0.5, ],
            [0.5, 0.5, 0.5, ],
        ],
    ])
    size = (5, 5, 5)
    result = nderp(a, b, x)
    print_array(result, 2, dec_round=1)
