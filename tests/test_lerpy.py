"""
test_lerpy
~~~~~~~~~~
"""
import numpy as np

from lerpy import lerpy as lp
from tests.common import ArrayTestCase


# Test cases.
class BilinearInterpolationTestCase(ArrayTestCase):
    def test_bilinear_interpolation(self):
        """Given two two-dimensional arrays of values, and an array
        of distances, return an array that is the bidimensional
        interpolation of the values at the distances.
        """
        # Expected value.
        exp = np.array([
            [0.00, 0.25, 0.50, 0.75, 1.00, ],
            [0.50, 0.75, 1.00, 1.25, 1.50, ],
            [1.00, 1.25, 1.50, 1.75, 2.00, ],
            [1.50, 1.75, 2.00, 2.25, 2.50, ],
            [2.00, 2.25, 2.50, 2.75, 3.00, ],
        ])
        
        # Test data and setup.
        a = np.array([
            [0.00, 0.00, 0.00, 0.00, 1.00, ],
            [0.00, 0.00, 0.00, 0.00, 1.00, ],
            [0.00, 0.00, 0.00, 0.00, 1.00, ],
            [0.00, 0.00, 0.00, 0.00, 1.00, ],
            [2.00, 2.00, 2.00, 2.00, 3.00, ],
        ])
        b = np.array([
            [0.00, 1.00, 1.00, 1.00, 1.00, ],
            [2.00, 3.00, 3.00, 3.00, 3.00, ],
            [2.00, 3.00, 3.00, 3.00, 3.00, ],
            [2.00, 3.00, 3.00, 3.00, 3.00, ],
            [2.00, 3.00, 3.00, 3.00, 3.00, ],
        ])
        x = np.array([
            [
                [0.00, 0.25, 0.50, 0.75, 1.00, ],
                [0.00, 0.25, 0.50, 0.75, 1.00, ],
                [0.00, 0.25, 0.50, 0.75, 1.00, ],
                [0.00, 0.25, 0.50, 0.75, 1.00, ],
                [0.00, 0.25, 0.50, 0.75, 1.00, ],
            ],
            [
                [0.00, 0.00, 0.00, 0.00, 0.00, ],
                [0.25, 0.25, 0.25, 0.25, 0.25, ],
                [0.50, 0.50, 0.50, 0.50, 0.50, ],
                [0.75, 0.75, 0.75, 0.75, 0.75, ],
                [1.00, 1.00, 1.00, 1.00, 1.00, ],
            ],
        ])
        
        # Run test.
        act = lp.bilinear_interpolation(a, b, x)
        
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

        # Test data and setup.
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, ])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, ])
        x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, ])

        # Run test.
        act = lp.linear_interpolation(a, b, x)

        # Determine test result.
        self.assertArrayEqual(exp, act)
