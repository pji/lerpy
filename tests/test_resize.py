"""
test_resize
~~~~~~~~~~~

Unit tests for the lerpy.resize module.
"""
import numpy as np

from lerpy import lerpy as lp
from lerpy import resize as rs
from tests.common import ArrayTestCase


# Unit tests.
class BuildResizingMatricesTestCase(ArrayTestCase):
    def test_build_matrix_increase_size(self):
        """Given an original size and a final size, return two arrays
        of indexes and one of distances that can be used to interpolate
        the values of the resized array.
        """
        # Expected value.
        exp_a = np.array([
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2],
                ],
                [
                    [0, 0, 1, 1, 2],
                    [0, 0, 1, 1, 2],
                    [0, 0, 1, 1, 2],
                    [0, 0, 1, 1, 2],
                    [0, 0, 1, 1, 2],
                ],
            ],
            [
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2],
                ],
                [
                    [0, 0, 1, 1, 2],
                    [0, 0, 1, 1, 2],
                    [0, 0, 1, 1, 2],
                    [0, 0, 1, 1, 2],
                    [0, 0, 1, 1, 2],
                ],
            ],
        ])
        exp_b = np.array([
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2],
                ],
                [
                    [1, 1, 2, 2, 2],
                    [1, 1, 2, 2, 2],
                    [1, 1, 2, 2, 2],
                    [1, 1, 2, 2, 2],
                    [1, 1, 2, 2, 2],
                ],
            ],
            [
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2],
                ],
                [
                    [1, 1, 2, 2, 2],
                    [1, 1, 2, 2, 2],
                    [1, 1, 2, 2, 2],
                    [1, 1, 2, 2, 2],
                    [1, 1, 2, 2, 2],
                ],
            ],
        ])
        exp_x = np.array([
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ],
            [
                [0.0000, 0.5000, 0.0000, 0.5000, 0.0000],
                [0.0000, 0.5000, 0.0000, 0.5000, 0.0000],
                [0.0000, 0.5000, 0.0000, 0.5000, 0.0000],
                [0.0000, 0.5000, 0.0000, 0.5000, 0.0000],
                [0.0000, 0.5000, 0.0000, 0.5000, 0.0000],
            ],
        ])

        # Test data and state.
        src_shape = (3, 3)
        dst_shape = (5, 5)

        # Run test.
        act_a, act_b, act_x = rs.build_resizing_matrices(src_shape, dst_shape)

        # Determine test result.
        self.assertArrayEqual(exp_a, act_a)
        self.assertArrayEqual(exp_b, act_b)
        self.assertArrayEqual(exp_x, act_x)


class MagnifySize(ArrayTestCase):
    def test_magnify_size(self):
        """Given the shape of an array and a magnification factor,
        return the magnified shape of the array.
        """
        # Expected values.
        exp = (10, 10, 10)

        # Test data and state.
        size = (5, 5, 5)
        factor = 2.0

        # Run test.
        act = rs.magnify_size(size, factor)

        # Determine test result.
        self.assertTupleEqual(exp, act)


class ResizeArrayTestCase(ArrayTestCase):
    def test_three_dimensions(self):
        """Given a three-dimensional array and a new size, return an
        array of the new size with the data resized through trilinear
        interpolation.
        """
        # Expected values.
        exp = np.array([
            [
                [0.0, 0.5, 1.0, 1.5, 2.0],
                [0.5, 1.0, 1.5, 2.0, 2.5],
                [1.0, 1.5, 2.0, 2.5, 3.0],
                [1.5, 2.0, 2.5, 3.0, 3.5],
                [2.0, 2.5, 3.0, 3.5, 4.0],
            ],
            [
                [0.5, 1.0, 1.5, 2.0, 2.5],
                [1.0, 1.5, 2.0, 2.5, 3.0],
                [1.5, 2.0, 2.5, 3.0, 3.5],
                [2.0, 2.5, 3.0, 3.5, 4.0],
                [2.5, 3.0, 3.5, 4.0, 4.5],
            ],
            [
                [1.0, 1.5, 2.0, 2.5, 3.0],
                [1.5, 2.0, 2.5, 3.0, 3.5],
                [2.0, 2.5, 3.0, 3.5, 4.0],
                [2.5, 3.0, 3.5, 4.0, 4.5],
                [3.0, 3.5, 4.0, 4.5, 5.0],
            ],
            [
                [1.5, 2.0, 2.5, 3.0, 3.5],
                [2.0, 2.5, 3.0, 3.5, 4.0],
                [2.5, 3.0, 3.5, 4.0, 4.5],
                [3.0, 3.5, 4.0, 4.5, 5.0],
                [3.5, 4.0, 4.5, 5.0, 5.5],
            ],
            [
                [2.0, 2.5, 3.0, 3.5, 4.0],
                [2.5, 3.0, 3.5, 4.0, 4.5],
                [3.0, 3.5, 4.0, 4.5, 5.0],
                [3.5, 4.0, 4.5, 5.0, 5.5],
                [4.0, 4.5, 5.0, 5.5, 6.0],
            ],
        ])

        # Test data and state.
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
            [
                [2.0, 3.0, 4.0, ],
                [3.0, 4.0, 5.0, ],
                [4.0, 5.0, 6.0, ],
            ],
        ])
        size = (5, 5, 5)

        # Run test.
        act = rs.resize_array(a, size)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_resize_two_dimensions(self):
        """Given a two-dimensional array and a new size, return an
        array of the new size with the data resized through bilinear
        interpolation.
        """
        # Expected values.
        exp = np.array([
            [0.0, 0.5, 1.0, 1.5, 2.0, ],
            [0.5, 1.0, 1.5, 2.0, 2.5, ],
            [1.0, 1.5, 2.0, 2.5, 3.0, ],
            [1.5, 2.0, 2.5, 3.0, 3.5, ],
            [2.0, 2.5, 3.0, 3.5, 4.0, ],
        ])

        # Test data and state.
        a = np.array([
            [0.0, 1.0, 2.0, ],
            [1.0, 2.0, 3.0, ],
            [2.0, 3.0, 4.0, ],
        ])
        size = (5, 5)

        # Run test.
        act = rs.resize_array(a, size)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_shrink_array(self):
        """If the new size is smaller than the original size, the
        returned array should be smaller.
        """
        # Expected value.
        exp = np.array([
            [1, 3, 5],
            [3, 3, 3],
            [5, 3, 1],
        ])

        # Test data and state.
        a = np.array([
            [1, 2, 3, 4, 5],
            [2, 3, 3, 3, 4],
            [3, 3, 3, 3, 3],
            [4, 3, 3, 3, 2],
            [5, 4, 3, 2, 1],
        ])
        size = (3, 3)

        # Run test.
        act = rs.resize_array(a, size)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_use_ndcerp(self):
        """Given the n-dimensional cubic interpolation function,
        the resizing should use it for interpolation instead of the
        n-dimensional linear interpolation function.
        """
        # Expected values.
        exp = np.array([
            [0.0000, 0.3125, 1.0000, 2.5000, 4.0000],
            [4.3164, 5.8906, 8.2617, 11.3125, 14.5938],
            [9.0000, 11.9375, 16.0000, 20.5000, 25.0000],
            [22.1523, 26.4688, 32.2852, 38.3125, 44.7812],
            [36.0000, 41.5625, 49.0000, 56.5000, 64.0000],
        ])

        # Test data and state.
        a = np.arange(9, dtype=float).reshape((3, 3))
        a = a ** 2
        size = (5, 5)
        erp = lp.ndcerp

        # Run test.
        act = rs.resize_array(a, size, erp)

        # Determine test result.
        self.assertArrayEqual(exp, act, round_=True)
