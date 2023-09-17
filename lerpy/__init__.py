"""
__init__
~~~~~~~~

Common interpolation algorithms for numpy arrays.
"""
__all__ = ['lerpy', 'resize']
from lerpy.lerpy import (
    cerp,
    cubic_interpolation,
    lerp,
    linear_interpolation,
    n_dimensional_cubic_interpolation,
    n_dimensional_interpolation,
    n_dimensional_linear_interpolation,
    ndcerp,
    nderp,
    ndlerp
)
from lerpy.resize import magnify_size, resize_array
