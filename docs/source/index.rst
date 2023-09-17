.. lerpy documentation master file, created by
   sphinx-quickstart on Sun Sep 17 11:52:32 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. welcome:

Welcome to lerpy's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   self
   /api.rst


Why did you write this?
=======================
I've been working on some code to procedurally generate images and
video. It is getting pretty bloated, and the interpolations seemed
like something reasonable to peel off. I have no idea if its useful
to anyone else. :mod:`numpy` has its own interpolation function:
:func:`numpy.interp`. There is also :func:`scipy.interpolate`.
So, maybe this is redundant. Either way, it's here.


How do I run lerpy?
===================
You can clone the repository to your local system and play with it.
It's intended to be used as a module by other python applications. You
can find examples of how to use it in the examples directory.


Is it portable?
===============
It should be. It's pure Python, and the main library it uses is
:mod:`numpy`.


Can I install this package with pipenv?
=======================================
Yes, but :mod:`imgwriter` is not currently available through PyPI.
You will need to clone the repository to the system you want to install
:mod:`imgwriter` on and run the following::

    pipenv install path/to/local/copy

Replace `path/to/local/copy` with the path for your local clone of
this repository.

That said :mod:`imgwriter` is only need for the unit tests and example
code, so it's not necessary for just using :mod:`lerpy` as a package.


How do I run the tests?
=======================
The `Makefile` at the root of the directory is set up to quickly run
the tests. You will need to have installed the development requirements
in order to run them. To install the development requirements::

    python3 -m pip install pipenv
    pipenv install --dev

Then to run the unit tests::

    make test

To run all the precommit tests::

    make pre

The package is also set up to allow testing against supported versions
of Python using :mod:`tox`. To use those tests::

    tox 


How do I contribute?
====================
At this time, this is code is really just me exploring and learning.
I've made it available in case it helps anyone else, but I'm not really
intending to turn this into anything other than a personal project.

That said, if other people do find it useful and start using it, I'll
reconsider. If you do use it and see something you want changed or
added, go ahead and open an issue. If anyone ever does that, I'll
figure out how to handle it.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
