"""
__init__
~~~~~~~~

Common interpolation algorithms for numpy arrays.
"""
__all__ = ['lerpy', 'resize']
from lerpy.lerpy import (
    cubic_interpolation,
    linear_interpolation,
    n_dimensional_interpolation,
    n_dimensional_cubic_interpolation,
    n_dimensional_linear_interpolation,
    cerp,
    lerp,
    nderp,
    ndcerp,
    ndlerp
)
from lerpy.resize import (
    magnify_size,
    resize_array
)
