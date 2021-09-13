"""
test_lerpy
~~~~~~~~~~
"""
from math import prod
import unittest as ut

import numpy as np

from lerpy import lerpy as lp
from tests.common import ArrayTestCase


# Test cases.
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
        act_a, act_b, act_x = lp.build_resizing_matrices(src_shape, dst_shape)

        # Determine test result.
        self.assertArrayEqual(exp_a, act_a)
        self.assertArrayEqual(exp_b, act_b)
        self.assertArrayEqual(exp_x, act_x)


class CubicInterpolcationTestCase(ArrayTestCase):
    def test_cubic_interpolation(self):
        """Given four arrays of values named a, b, c, and d and an
        array of distances between the points in the b and c arrays,
        return an array of the interpolated values.
        """
        # Expected values.
        exp = np.array([0.3750, 2.2500, 6.2500, 12.2500])

        # Test data and state.
        a_ = np.array([-1.0, 0.0, 1.0, 4.0])
        a = np.array([0.0, 1.0, 4.0, 9.0])
        b = np.array([1.0, 4.0, 9.0, 16.0])
        b_ = np.array([4.0, 9.0, 16.0, 25.0])
        x = np.array([0.5, 0.5, 0.5, 0.5])

        # Run test.
        act = lp.cubic_interpolation(a, b, x, a_, b_)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_cubic_interpolation_without_primes(self):
        """Given two arrays of values named a, b, and an
        array of distances between the points in two arrays,
        return an array of the interpolated values.
        """
        # Expected values.
        exp = np.array([0.3125, 2.2500, 6.2500, 12.8125])

        # Test data and state.
        a = np.array([0.0, 1.0, 4.0, 9.0])
        b = np.array([1.0, 4.0, 9.0, 16.0])
        x = np.array([0.5, 0.5, 0.5, 0.5])

        # Run test.
        act = lp.cubic_interpolation(a, b, x)

        # Determine test result.
        self.assertArrayEqual(exp, act)


class LinearInterpolationTestCase(ArrayTestCase):
    def test_linear_interpolation(self):
        """Given two linear arrays of values and one linear array of
        distances, return an array that is the linear interpolation of
        the values at the distances.
        """
        # Expected value.
        exp = np.array([0.5, 1.5, 2.5, 3.5, 4.5, ])

        # Test data and state.
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, ])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, ])
        x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, ])

        # Run test.
        act = lp.linear_interpolation(a, b, x)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_perserving_array_data_type(self):
        """The returned array should have the same datatype as the
        first of the given value arrays.
        """
        # Expected value.
        exp = np.uint8

        # Test data and state.
        a = np.array([0, 1, 2, 3], dtype=exp)
        b = np.array([1, 2, 3, 4], dtype=exp)
        x = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)

        # Run test.
        result = lp.linear_interpolation(a, b, x)
        act = result.dtype

        # Determine test result.
        self.assertEqual(exp, act)


class NDimensionalInterpolationTestCase(ArrayTestCase):
    def test_wrong_number_of_points(self):
        """If the passed values do not contain enough points to
        perform the interpolation over multiple dimensions, raise
        a ValueError exception.
        """
        # Expected values.
        exp_ex = ValueError
        exp_msg = 'Not the correct number of points for the dimensions.'

        # Test values and state.
        size = (3, 3, 3)
        dims = len(size)
        ab_shape = tuple([2 * dims // 2, *size])      # Should be 2 ** dims.
        length = prod(ab_shape)
        a = np.arange(length).reshape(ab_shape)
        b = a ** 2
        x = np.full((dims, *size), .5)

        # Run test and determine result.
        with self.assertRaisesRegex(exp_ex, exp_msg):
            act = lp.n_dimensional_interpolation(a, b, x, lp.lerp)


class NDimensionalCubicInterpolationTestCase(ArrayTestCase):
    def test_two_dimensions(self):
        '''Given two arrays of values and an array of distances, return
        an array with the bilinear interpolation of the value arrays.
        '''
        # Expected values.
        exp = np.array([
            [20, 25, 34],
            [39, 48, 60],
            [68, 79, 95],
        ])

        # Test values and state.
        size = (3, 3)
        dims = len(size)
        ab_shape = tuple([2 ** dims // 2, *size])
        length = prod(ab_shape)
        a = np.arange(length).reshape(ab_shape)
        b = a ** 2
        x = np.full((dims, *size), .5)

        # Run test.
        act = lp.n_dimensional_cubic_interpolation(a, b, x)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_three_dimensions(self):
        '''The interpolation should still work with three dimensional
        arrays.
        '''
        # Expected values.
        exp = np.array([
            [
                [1282, 1324, 1382],
                [1408, 1455, 1516],
                [1544, 1594, 1659],
            ],
            [
                [1688, 1742, 1811],
                [1843, 1898, 1972],
                [2005, 2064, 2142],
            ],
            [
                [2177, 2239, 2321],
                [2358, 2423, 2509],
                [2548, 2616, 2706],
            ],
        ])

        # Test values and state.
        size = (3, 3, 3)
        dims = len(size)
        ab_shape = tuple([2 ** dims // 2, *size])
        length = prod(ab_shape)
        a = np.arange(length).reshape(ab_shape)
        b = a ** 2
        x = np.full((dims, *size), .5)

        # Run test.
        act = lp.n_dimensional_cubic_interpolation(a, b, x)

        # Determine test result.
        self.assertArrayEqual(exp, act)


class NDimensionalLinearInterpolationTestCase(ArrayTestCase):
    def test_two_dimensions(self):
        '''Given two arrays of values and an array of distances, return
        an array with the bilinear interpolation of the value arrays.
        '''
        # Expected values.
        exp = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ])

        # Test values and state.
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

        # Run test.
        act = lp.n_dimensional_linear_interpolation(a, b, x)

        # Determine test result.
        self.assertArrayEqual(exp, act)


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
        act = lp.resize_array(a, size)

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
        act = lp.resize_array(a, size)

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
        act = lp.resize_array(a, size)

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
        act = lp.resize_array(a, size, erp)

        # Determine test result.
        self.assertArrayEqual(exp, act, round_=True)
