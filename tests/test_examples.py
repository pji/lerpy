"""
test_examples
~~~~~~~~~~~~~

Unit tests to ensure the provided example code still works.
"""
import shutil
from pathlib import Path
from subprocess import run

from imgwriter import read_image
import pytest as pt


# Test fixtures.
@pt.fixture
def cleanup(request):
    """Clean up output files after the test."""
    files = request.node.get_closest_marker('outfile')
    yield files.args[0]
    for file in files.args:
        path = Path(file)
        if path.exists():
            path.unlink()


@pt.fixture
def script():
    """Move the tested script so it can be tested."""
    script = 'resize_image.py'
    src_path = Path('examples') / script
    shutil.copy2(src_path, script)
    yield script
    Path(script).unlink()


# Tests for resize_image.py.
@pt.mark.outfile('__ResizeImageTestCase__test_magnify.jpg')
def test_resize_image_magnify(script, cleanup):
    """Given an image file, the save location, and a magnification
    factor, `resize_image.py` should save the resized image in the
    save location.
    """
    exp_file = 'tests/data/__test_resize_image_after_mag.jpg'
    src_file = 'tests/data/__test_resize_image_before.jpg'
    dst_file = cleanup
    cmd = ['python', script, src_file, dst_file, '-m', '10']
    run(cmd)
    assert (read_image(exp_file) == read_image(dst_file)).all()


@pt.mark.outfile('__ResizeImageTestCase__test_resize.jpg')
def test_resize_image_resize(script, cleanup):
    """Given an image file, the save location, and a new size,
    `resize_image.py` should save the resized image in the save
    location.
    """
    exp_file = 'tests/data/__test_resize_image_after.jpg'
    src_file = 'tests/data/__test_resize_image_before.jpg'
    dst_file = cleanup
    cmd = ['python', script, src_file, dst_file, '-s', '10', '10']
    run(cmd)
    assert (read_image(exp_file) == read_image(dst_file)).all()
