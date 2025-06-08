"""Subclasses dualpy for pint variables"""

import copy
import warnings
from typing import Union, Optional

import numpy as np
from numpy.typing import DTypeLike
import pint

from .duals import dlarray
from .unitless import Unitless


class dlarray_pint(dlarray):
    """A subclass of dlarray that wraps a pint variable.

    To wrap a regular numpy array, see dlarray (see there also for most of the
    documentation).  To wrap an astropy.Quantity see dlarray_astropy. (More may follow).
    """

    # --------------------------------------------- Overload attributes
    @property
    def _rad(self):
        return self.variable.units._REGISTRY.rad

    @property
    def _per_rad(self):
        return self.variable.units._REGISTRY.rad ** (-1)

    @property
    def _dimensionless(self):
        return self.variable.units._REGISTRY.dimensionless

    # --------------------------------------------- Some customization
    # pylint: disable-next=redefined-outer-name
    def __array__(self, dtype: Optional[DTypeLike] = None, copy: Optional[bool] = None):
        # Not 100% sure that this filterwarning is warranted here, but it raises
        # warnings (e.g., in matplotlib) when regular pint does not, so deciding to
        # mimic that approach.
        if np.__version__ != "1.26.4":
            warnings.warn(
                "The behavior of the copy argument in numpy.array may have improved, check this"
            )
        if copy is None:
            copy = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pint.UnitStrippedWarning)
            if dtype is None and hasattr(self, "dtype"):
                dtype = self.dtype
            return np.array(self.variable, dtype=dtype, copy=copy)

    # --------------------------------------------- Extra properties
    @property
    def units(self):
        return self.variable.units

    @property
    def magnitude(self):
        out = dlarray(self.variable.magnitude)
        for name, jacobian in self.jacobians.items():
            out.jacobians[name] = copy.copy(jacobian)
            out.jacobians[name].dependent_unit = Unitless()
        return out

    # --------------------------------------------- This replaces an attribute
    @property
    def _dependent_unit(self):
        return self.variable.units

    # --------------------------------------------- Some specific methods
    def to(self, units, **kwargs):
        # If it's a no-op, then take advantage of that
        if self.units is units:
            return self
        out_ = self.variable.to(units, **kwargs)
        out = dlarray(out_)
        if self.hasJ:
            # This will probably (hopefully!) fail if we're using one of
            # those function units things (such as dB).
            scale = (1.0 * self.units).to(units, **kwargs) / self.units
            out._chain_rule(self, scale)
        return out

    def to_base_units(self):
        result_ = self.variable.to_base_units()
        result = dlarray(result_)
        if self.hasJ:
            for key, jacobian in self.jacobians.items():
                result.jacobians[key] = self.jacobians[key].to(result.units)
        return result

    def _to_radians(self):
        return self.to(self._rad)

    @staticmethod
    def _force_unit(quantity, units):
        """Apply a unit to a quantity"""
        if units is not None:
            return quantity.magnitude * units
        else:
            return quantity

    @staticmethod
    def _force_unit_from(quantity, source):
        """Apply a unit to a quantity"""
        try:
            units = source.units
            return np.array(quantity) * units
        except AttributeError:
            return quantity

    @staticmethod
    def _get_magnitude_if_dimensionless(units: pint.Unit):
        # Turn it from a unit to a quantity
        value = 1.0 * units
        # Try to force it into dimensionless
        value = value.to(value.units._REGISTRY.dimensionless)
        # Return its magnitude
        return value.magnitude

    # ------------------------------------------ Some dunders
    # def __mul__(self, other):
    #     # Needed for dual * unit case
    #     if isinstance(other, (ureg.Unit, str)):
    #         return self * other
    #     return super().__mul__(other)

    # def __truediv__(self, other):
    #     # Needed for dual * unit case
    #     if isinstance(other, (ureg.Unit, str)):
    #         return self / other
    #     return super().__truediv__(other)


# ------------------------------------------- Upcasting

# Make sure pint defers to us when appropriate.  Do this by adding dlarray and
# dlarray_pint to the pint.compat.upcast ecosystem.  This varies by pint version


def add_upcast_types(types_to_add: Union[list[Union[type, str]], str]):
    """Add a type to the upcast mechanism in a pint-version agnostic way

    Parameters:
    -----------
    name : list[str] or str
        Names of types to be added
    """
    # Force names into a list
    if isinstance(types_to_add, str) or isinstance(types_to_add, type):
        types_to_add = [types_to_add]
    # Work out how we do this (pint-version-dependent)
    upcast_list = None
    upcast_dict = None
    # Work out where these things are going go
    if hasattr(pint.compat, "upcast_type_map"):
        upcast_dict = pint.compat.upcast_type_map
    elif hasattr(pint.compat, "upcast_types"):
        upcast_list = pint.compat.upcast_types
    else:
        upcast_list = pint.compat.upcast_type_names
    # Work out actually what we're inserting
    if upcast_dict:
        # If we have a dict to fill, then we fill it with the names of types
        items_to_add = [
            type_to_add.__module__ + "." + type_to_add.__qualname__
            for type_to_add in types_to_add
        ]
    else:
        items_to_add = types_to_add

    # Insert them
    for item in items_to_add:
        if upcast_list:
            if item not in upcast_list:
                upcast_list.append(item)
        elif upcast_dict:
            if item not in upcast_dict:
                upcast_dict[item] = None
        else:
            raise NotImplementedError("Do not understand current version of pint")


# Now actually do it for our types
add_upcast_types([dlarray, dlarray_pint])
