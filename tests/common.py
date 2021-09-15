"""
common
~~~~~~

Common code used in multiple test modules
"""
from math import prod
import unittest as ut

import numpy as np


# Base test cases.
class ArrayTestCase(ut.TestCase):
    def assertArrayEqual(self, a, b, round_=False):
        """Assert that two numpy.ndarrays are equal."""
        if round_:
            a = np.around(a, 4)
            b = np.around(b, 4)

        if prod(a.shape) < 100:
            a_list = a.tolist()
            b_list = b.tolist()
            self.assertListEqual(a_list, b_list)

        else:
            self.assertTrue(np.array_equal(a, b))

    def assertArrayNotEqual(self, a, b, round_=False):
        """Assert that two numpy.ndarrays are not equal."""
        if round_:
            a = np.around(a, 4)
            b = np.around(b, 4)
        a_list = a.tolist()
        b_list = b.tolist()
        self.assertFalse(a_list == b_list)
