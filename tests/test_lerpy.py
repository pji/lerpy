"""
test_lerpy
~~~~~~~~~~
"""
from math import prod

import numpy as np

from lerpy import lerpy as lp
from tests.common import ArrayTestCase


# Test cases.
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
