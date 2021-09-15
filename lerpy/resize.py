"""
resize
~~~~~~

Functions for resizing numpy arrays through interpolation.
"""
from math import prod
from typing import Callable, Optional

import numpy as np

from lerpy import lerpy as lp


# Public functions.
def build_resizing_matrices(src_shape: tuple[int, ...],
                            dst_shape: tuple[int, ...]
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the indexing and distance arrays needed to interpolate
    values when resizing an array.

    :param src_shape: The original shape of the array.
    :param dst_shape: The resized shape of the array.
    :return: A :class:tuple object.
    :rtype: tuple
    """
    # Interpolation guesses a value between known data values. To
    # do this you need to know those points. The number of points
    # surrounding the value being guessed is the square of the
    # dimensions in the array.
    num_dim = len(src_shape)
    axes = range(num_dim)
    points = range(2 ** num_dim)

    # The relative positions of the points compared to the interpolated
    # value is coded by a binary text string where 1 is after the value
    # on the axis and 0 is before the value.
    rel_positions = _build_relative_position_masks(num_dim)

    # Create the map for position 0, which is before the interpolated
    # value on every axis.
    factors = _get_resizing_factors(src_shape, dst_shape)
    src_indices, x = _map_indices_and_distances(dst_shape, factors)

    # Create the maps for the rest of the positions.
    matrix_shape = (len(points) // 2, num_dim, *dst_shape)
    a = np.zeros(matrix_shape, dtype=int)
    b = a.copy()
    for pos in rel_positions:
        matrix_index = int(pos, 2) // 2
        pos_indices = src_indices.copy()
        for axis in axes:
            if pos[axis] == '1':
                pos_indices[axis] += 1

                # Cap the values in the array to the highest index in
                # the original array.
                cap = src_shape[axis] - 1
                pos_indices[pos_indices > cap] = cap

        # Put the value in the correct side of the resizing matrices.
        if pos.endswith('0'):
            a[matrix_index] = pos_indices
        else:
            b[matrix_index] = pos_indices

    # Return the arrays for the resizing interpolation.
    return a, b, x


def magnify_size(size: tuple[int, ...], factor: int) -> tuple[int, ...]:
    """Magnify the shape of an array."""
    return tuple(int(n * factor) for n in size)


def resize_array(src: np.ndarray,
                 size: tuple[int, ...],
                 interpolator: Optional[Callable] = None) -> np.ndarray:
    """Resize an two dimensional array using linear interpolation.

    :param a: The array to resize. The array is expected to have at
        least two dimensions.
    :param size: The shape for the resized array.
    :param interpolator: The interpolation algorithm for the resizing.
    :return: A :class:ndarray object.
    :rtype: numpy.ndarray
    """
    # Perform defensive actions to prevent unneeded processing if
    # the array won't actually change and to make sure any changes
    # to the array won't have unexpected side effects.
    if size == src.shape:
        return src
    src = src.copy()

    # Map out the relationship between the old space and the
    # new space.
    a, b, x = build_resizing_matrices(src.shape, size)
    a = _replace_indices_with_values(src, a)
    b = _replace_indices_with_values(src, b)

    # Perform the interpolation using the mapped space and return.
    if interpolator is None:
        interpolator = lp.ndlerp
    return interpolator(a, b, x)


# Private functions.
def _build_relative_position_masks(dimensions: int) -> list[str]:
    """Create the masks for identifying the different points used in
    an n-dimensional interpolation.
    """
    points = range(2 ** dimensions)
    mask_template = '{:>0' + str(dimensions) + 'b}'
    mask = [mask_template.format(p)[::-1] for p in points]
    return sorted(mask)


def _get_resizing_factors(src_shape: tuple[int, ...],
                          dst_shape: tuple[int, ...]) -> tuple[float, ...]:
    """Determine how much each axis is resized by."""
    # The ternary is a quick fix for cases where there are dimensions
    # of length one. It may cause weird effects, so a more thoughtful
    # fix would be good in the future.
    src_ends = [n - 1 if n != 1 else 1 for n in src_shape]
    dst_ends = [n - 1 if n != 1 else 1 for n in dst_shape]
    factors = tuple(d / s for s, d in zip(src_ends, dst_ends))
    return factors


def _map_indices_and_distances(shape: tuple[int, ...],
                               factors: tuple[float, ...]
                               ) -> tuple[np.ndarray, ...]:
    """Map the indices for the zero position array and the distances
    for the distance array for an array resizing interpolation.
    """
    axes = range(len(shape))
    indices = np.indices(shape, dtype=float)
    for axis in axes:
        indices[axis] /= factors[axis]
    src_indices = np.trunc(indices)
    distances = indices - src_indices
    return src_indices, distances


def _replace_indices_with_values(src: np.ndarray,
                                 indices: np.ndarray) -> np.ndarray:
    """Replace the indices in an array with values from another array."""
    # numpy.take only works in one dimension. We'll need to
    # ravel the original array to be able to get the values, but
    # we still need the original shape to calculate the new indices.
    src_shape = src.shape
    raveled = np.ravel(src)

    # Calculate the raveled indices for each dimension.
    num_dim = len(src_shape)
    axes = range(num_dim)
    raveled_shape = (indices.shape[0], *indices.shape[2:])
    raveled_indices = np.zeros((raveled_shape), dtype=int)
    for axis in axes:
        remaining_dims = src_shape[axis + 1:]
        axis_mod = prod(remaining_dims)
        raveled_indices += indices[:, axis] * axis_mod

    # Return the values from the original array.
    result = np.take(raveled, raveled_indices.astype(int))
    return result
