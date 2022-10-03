"""Subclasses dualpy for astropy.Quantity variables"""
import numpy as np
import astropy.units as units

from .duals import dlarray

class dlarray_astropy(dlarray):
    """A subclass of dlarray that wraps an astropy.Quantity variable.

    To wrap a regular numpy array, see dlarray (see there also for most of the
    documentation).  To wrap a pint quantity see dlarray_pint. (More may follow).
    """

    # --------------------------------------------- Overload attributes
    _rad = units.rad
    _per_rad = units.rad ** (-1)
    _dimensionless = units.dimensionless_unscaled 

    # --------------------------------------------- Extra properties
    @property
    def unit(self):
        return self.variable.unit

    # @property
    # def value(self):
    #     return self.variable.value

    # --------------------------------------------- This replaces an attribute
    @property
    def _dependent_unit(self):
        return self.variable.unit

    # --------------------------------------------- Some specific methods
    def to(self, unit, **kwargs):
        # If it's a no-op, then take advantage of that
        if self.unit is unit:
            return self
        out_ = self.variable.to(unit, **kwargs)
        out = dlarray(out_)
        # This will probably (hopefully!) fail if we're using one of
        # those function unit things (such as dB).
        if self.hasJ:
            scale = self.unit.to(unit, **kwargs) << unit / self.unit
            out._chain_rule(self, scale)
        return out

    def decompose(self):
        return self.to(self.unit.decompose())

    def __lshift__(self, other):
        out = dlarray(np.array(self) << other)
        for name, jacobian in self.jacobians.items():
            out.jacobians[name] = jacobian << other
        return out

    def _to_radians(self):
        return self.to(units.rad)

    @staticmethod
    def _force_unit(quantity, unit):
        """Apply a unit to a quantity"""
        if unit is not None:
            return np.array(quantity) << unit
        else:
            return quantity

    @staticmethod
    def _force_unit_from(quantity, source):
        """Apply a unit to a quantity"""
        try:
            unit = source.units
            return np.array(quantity) << unit
        except AttributeError:
            return quantity
    # ------------------------------------------ Some dunders
    # def __mul__(self, other):
    #     # Needed for dual * unit case
    #     if isinstance(other, (units.UnitBase, str)):
    #         return self * units.Quantity(1.0, other)
    #     return super().__mul__(other)

    # def __truediv__(self, other):
    #     # Needed for dual * unit case
    #     if isinstance(other, (units.UnitBase, str)):
    #         return self / units.Quantity(1.0, other)
    #     return super().__truediv__(other)

