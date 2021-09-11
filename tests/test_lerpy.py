"""
test_lerpy
~~~~~~~~~~
"""
import unittest as ut

import numpy as np

from lerpy import lerpy as lp
from tests.common import ArrayTestCase


# Test cases.
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


class NDimensionalInterpolationTestCase(ArrayTestCase):
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
        act = lp.n_dimensional_interpolation(a, b, x)

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
