"""
test_lerpy
~~~~~~~~~~
"""
import numpy as np

from lerpy import lerpy as lp
from tests.common import ArrayTestCase


# Test cases.
class LinearInterpolationTestCase(ArrayTestCase):
    def test_linear_interpolation(self):
        """Given two arrays of values and one array of distances, return
        an array that is the linear interpolation of the values at the
        distances.
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
