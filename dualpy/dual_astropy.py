"""Subclasses dualpy for astropy.Quantity variables"""
from .duals import dlarray
import astropy.units as units

class dlarray_astropy(dlarray):
    """A subclass of dlarray that wraps an astropy.Quantity variable.

    To wrap a regular numpy array, see dlarray (see there also for most of the
    documentation).  To wrap a pint quantity see dlarray_pint. (More may follow).
    """

    # --------------------------------------------- Overload attributes
    _rad = units.rad
    _per_rad = units.rad ** (-1)

    # --------------------------------------------- Extra properties
    @property
    def unit(self):
        return self.variable.unit

    @property
    def value(self):
        return self.variable.value

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

    def _to_radians(self):
        return self.to(units.rad)
