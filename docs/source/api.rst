.. _api:

##########
Public API
##########

The following are the functions that make up the public API of
:mod:`lerpy`.


Interpolators
=============
The following are the main interpolation functions in :mod:`lerpy`:

.. autofunction:: lerpy.cubic_interpolation
.. autofunction:: lerpy.linear_interpolation
.. autofunction:: lerpy.n_dimensional_interpolation
.. autofunction:: lerpy.n_dimensional_cubic_interpolation
.. autofunction:: lerpy.n_dimensional_linear_interpolation


Interpolator Aliases
====================
For ease of use, the interpolators have shorter aliases you can use:

*   `cerp`: :func:`lerpy.cubic_interpolation`
*   `lerp`: :func:`lerpy.linear_interpolation`
*   `nderp`: :func:`lerpy.n_dimensional_interpolation`
*   `ndcerp`: :func:`lerpy.n_dimensional_cubic_interpolation`
*   `ndlerp`: :func:`lerpy.n_dimensional_linear_interpolation`


Resizing
========
The following are utility functions that use the interpolators to resize
arrays:

.. autofunction:: lerpy.magnify_size
.. autofunction:: lerpy.resize_array


Debugging
=========
The following function is useful for debugging code that works with
:class:`numpy.ndarray` objects:

.. autofunction:: lerpy.print_array
