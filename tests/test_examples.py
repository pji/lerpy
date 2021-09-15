"""
test_examples
~~~~~~~~~~~~~

Unit tests to ensure the provided example code still works.
"""
import os
import shutil
import unittest as ut

from imgwriter import read_image

from tests.common import ArrayTestCase


# Test cases.
class ResizeImageTestCase(ArrayTestCase):
    example: str = 'resize_image.py'

    def setUp(self):
        """Set up the test environment."""
        src_path = f'examples/{self.example}'
        shutil.copy2(src_path, self.example)

    def tearDown(self):
        """Clean up after the tests are complete."""
        os.remove(self.example)

    def test_magnify(self):
        """Given an image file, the save location, and a magnification
        factor, resize_image should save the resized image in the save
        location.
        """
        # Expected value.
        exp_file = 'tests/data/__test_resize_image_after_mag.jpg'
        exp = read_image(exp_file)

        # Test data and state.
        src_file = 'tests/data/__test_resize_image_before.jpg'
        dst_file = '__ResizeImageTestCase__test_magnify.jpg'
        options = '-m 10'
        cmd = f'python {self.example} {src_file} {dst_file} {options}'

        # Run test.
        try:
            os.system(cmd)

            # Determine test result.
            act = read_image(dst_file)
            self.assertArrayEqual(exp, act, round_=True)

        # Clean up after test.
        finally:
            if os.path.exists(dst_file):
                os.remove(dst_file)

    def test_resize(self):
        """Given an image file, the save location, and a new size,
        resize_image should save the resized image in the save
        location.
        """
        # Expected value.
        exp_file = 'tests/data/__test_resize_image_after.jpg'
        exp = read_image(exp_file)

        # Test data and state.
        src_file = 'tests/data/__test_resize_image_before.jpg'
        dst_file = '__ResizeImageTestCase__test_resize.jpg'
        options = '-s 10 10'
        cmd = f'python {self.example} {src_file} {dst_file} {options}'

        # Run test.
        try:
            os.system(cmd)

            # Determine test result.
            act = read_image(dst_file)
            self.assertArrayEqual(exp, act)

        # Clean up after test.
        finally:
            if os.path.exists(dst_file):
                os.remove(dst_file)
