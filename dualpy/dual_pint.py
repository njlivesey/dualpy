"""Subclasses dualpy for pint variables"""

import numpy as np
import pint
import copy
import warnings

from .duals import dlarray
from .unitless import Unitless


class dlarray_pint(dlarray):
    """A subclass of dlarray that wraps a pint variable.

    To wrap a regular numpy array, see dlarray (see there also for most of the
    documentation).  To wrap an astropy.Quantity see dlarray_pint. (More may follow).
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
    def __array__(self, dtype=None):
        # Not 100% sure that this filterwarning is warranted here, but
        # it raises warnings (e.g., in matplotlib) when regular pint
        # does not, so deciding to mimic that approach.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pint.UnitStrippedWarning)
            if dtype is None and hasattr(self, "dtype"):
                dtype = self.dtype
            return np.array(self.variable, dtype=dtype)

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
# Make sure pint defers to us when appropriate
if dlarray not in pint.compat.upcast_types:
    pint.compat.upcast_types.append(dlarray)
if dlarray_pint not in pint.compat.upcast_types:
    pint.compat.upcast_types.append(dlarray_pint)
