"""Subclasses dualpy for pint variables"""
from .duals import dlarray
# import pint


class dlarray_pint(dlarray):
    """A subclass of dlarray that wraps a pint variable.

    To wrap a regular numpy array, see dlarray (see there also for most of the
    documentation).  To wrap an astropy.Quantity see dlarray_pint. (More may follow).
    """

    pass
