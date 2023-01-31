"""Some low level routines to help with handling duals"""

import astropy.units as units
import pint
import numpy as np
import operator
from numpy.core import umath
import scipy.sparse as sparse

from mls_scf_tools.mls_pint import ureg
from .unitless import Unitless

__all__ = [
    "dedual",
    "isunit",
    "broadcast_jacobians",
    "setup_dual_operation",
    "force_unit",
    "DualOperatorsMixin",
    "get_unit",
    "get_magnitude",
    "get_magnitude_and_unit",
    "get_unit_conversion_scale",
    "has_jacobians",
    "apply_units",
]

from mls_scf_tools.mls_pint import mp_broadcast_arrays


def dedual(x, units_ok=False):
    """Return duck-array form of x that's not a dual

    Thus, it will return x.variable if x a dlarray or just x (at least for now,
    perhaps we should check x over in the latter case, but we'll only do so if it
    becomes clear we need to.

    """
    from .duals import dlarray

    if hasattr(x, "_dedual"):
        return x._dedual()
    if isinstance(x, dlarray):
        return x.variable
    elif isinstance(x, np.ndarray) or hasattr(x, "__array__"):
        return x
    elif (
        isinstance(x, int)
        or isinstance(x, float)
        or isinstance(x, complex)
        or isinstance(x, sparse.base.spmatrix)
    ):
        return x
    elif units_ok and (isinstance(x, units.core.UnitBase) or isinstance(x, pint.Unit)):
        return x
    else:
        raise ValueError(f"Do not know how to dedual object of type {type(x)}")


def isunit(x):
    return isinstance(x, units.core.UnitBase) or isinstance(x, pint.Unit)


def broadcast_jacobians(js, new):
    # Loop over jacobians and broadcast each of them to new shape
    out = {}
    for name, jacobian in js.items():
        out[name] = jacobian.broadcast_to(new)
    return out


def force_unit(quantity, *, unit=None, source=None):
    """Apply a unit to a quantity"""
    return quantity


def to_dimensionless(x):
    from .dual_astropy import dlarray_astropy
    from .dual_pint import dlarray_pint

    """Convert astropy/pint quantity to dimensionless and return as ndarray/dlarray"""
    if isinstance(x, dlarray_pint) or isinstance(x, pint.Quantity):
        return x.to(ureg.dimensionless)
    elif isinstance(x, dlarray_astropy) or isinstance(x, units.Quantity):
        return x.to(units.dimensionless_unscaled)
    else:
        return x


def setup_dual_operation(*args, out=None, broadcast=True):
    # Get the variables for all the arguments, be they duals or a non-dual
    # duck-array, strip the units off for now.
    arrays_ = [dedual(x, units_ok=True) for x in args]
    any_are_units = any(isunit(x) for x in arrays_)
    if broadcast and not any_are_units:
        # Down the road, pint will give us problems here.
        arrays_ = mp_broadcast_arrays(*arrays_, subok=True)
    # Now go through the jacobians
    jacobians = []
    for x, orig in zip(arrays_, args):
        if hasattr(orig, "jacobians"):
            try:
                x_shape = x.shape
            except AttributeError:
                x_shape = tuple()
            try:
                orig_shape = orig.shape
            except AttributeError:
                orig_shape = tuple()
            if orig_shape != x_shape and broadcast:
                j = broadcast_jacobians(orig.jacobians, x_shape)
            else:
                j = orig.jacobians
        else:
            j = {}
        jacobians.append(j)

    # Handle the case where an "out" is provided. I originally had some intelligence
    # about tracking whether out shares memory with any of the arguments but that
    # got complicated. In the end I've recognized that we can code up the binary
    # operations such that we don't need to worry about such cases.
    if out is not None:
        if isinstance(out, tuple):
            if len(out) != 1:
                raise NotImplementedError("Cannot support multiple outs")
            out = out[0]
        # As inefficient as this might appear, I'm pretty sure I need to blow away
        # the Jacobians in out and let the calling code recreate them from scratch.
        # Unpleasent things happen if not, as things done to out.jacobians leak back
        # on a and b's jacobians.
        if hasattr(out, "jacobians"):
            out.jacobians = {}

    return tuple(arrays_) + tuple(jacobians) + (out,)


def apply_units(values, unit):
    """Apply units to Jacobian values extracted to duckarrays"""
    if isinstance(unit, Unitless):
        return values
    if isinstance(unit, pint.Unit):
        return values * unit
    if isinstance(unit, units.UnitBase):
        return values << unit
    raise ValueError(f"Unsupported unit type {type(unit)}")


def get_magnitude_and_unit(x):
    """Split a pint/astropy into value and units"""
    from .dual_pint import dlarray_pint
    from .dual_astropy import dlarray_astropy

    if isinstance(x, pint.Quantity) or isinstance(x, dlarray_pint):
        return x.magnitude, x.units
    if isinstance(x, units.Quantity) or isinstance(x, dlarray_astropy):
        return x.value, x.unit
    return x, Unitless()


def get_magnitude(x):
    """Split a pint/astropy into value and units"""
    from .dual_pint import dlarray_pint
    from .dual_astropy import dlarray_astropy

    if isinstance(x, pint.Quantity) or isinstance(x, dlarray_pint):
        return x.magnitude
    if isinstance(x, units.Quantity) or isinstance(x, dlarray_astropy):
        return x.value
    return x


def get_unit(x):
    """Return the units for x"""
    from .dual_pint import dlarray_pint
    from .dual_astropy import dlarray_astropy

    if isinstance(x, pint.Quantity) or isinstance(x, dlarray_pint):
        return x.units
    if isinstance(x, units.Quantity) or isinstance(x, dlarray_astropy):
        return x.unit
    return Unitless()


def get_unit_conversion_scale(old_unit, new_unit):
    """Get a scale factor for converting from old_unit to new_unit"""
    if isinstance(old_unit, units.UnitBase):
        return old_unit._to(new_unit) * (new_unit / old_unit)
    elif isinstance(old_unit, pint.Unit) or isinstance(old_unit, pint.Quantity):
        return new_unit.from_(old_unit) / old_unit
    elif isinstance(old_unit, Unitless):
        return 1
    else:
        raise TypeError(f"Unable to handle units of type {type(old_unit)}")


def has_jacobians(a):
    """Return true if a is a dual with Jacobians"""
    # If it has a _has_jacobians method, invoke that.
    if hasattr(a, "_has_jacobians"):
        return a._has_jacobians()
    if not hasattr(a, "jacobians"):
        return False
    return bool(a.jacobians)


class DualOperatorsMixin:
    __slots__ = ()

    def _binary_op(self, other, f, reflexive=False):
        raise NotImplementedError

    def __add__(self, other):
        return self._binary_op(other, self.add)

    def __sub__(self, other):
        return self._binary_op(other, self.subtract)

    def __mul__(self, other):
        return self._binary_op(other, self.multiply)

    def __pow__(self, other):
        return self._binary_op(other, self.power)

    def __truediv__(self, other):
        return self._binary_op(other, self.true_divide)

    # def __floordiv__(self, other):
    #     return self._binary_op(other, self.floordiv)

    # def __mod__(self, other):
    #     return self._binary_op(other, self.mod)

    # def __and__(self, other):
    #     return self._binary_op(other, self.and_)

    # def __xor__(self, other):
    #     return self._binary_op(other, self.xor)

    # def __or__(self, other):
    #     return self._binary_op(other, self.or_)

    def __lt__(self, other):
        return self._binary_op(other, umath.less)

    def __le__(self, other):
        return self._binary_op(other, umath.less_equal)

    def __gt__(self, other):
        return self._binary_op(other, umath.greater)

    def __ge__(self, other):
        return self._binary_op(other, umath.greater_equal)

    def __eq__(self, other):
        return self._binary_op(other, umath.equal)

    def __ne__(self, other):
        return self._binary_op(other, umath.not_equal)

    def __radd__(self, other):
        return self._binary_op(other, self.add, reflexive=True)

    def __rsub__(self, other):
        return self._binary_op(other, self.subtract, reflexive=True)

    def __rmul__(self, other):
        return self._binary_op(other, self.multiply, reflexive=True)

    def __rpow__(self, other):
        return self._binary_op(other, self.power, reflexive=True)

    def __rtruediv__(self, other):
        return self._binary_op(other, self.true_divide, reflexive=True)

    # def __rfloordiv__(self, other):
    #     return self._binary_op(other, self.floordiv, reflexive=True)

    # def __rmod__(self, other):
    #     return self._binary_op(other, self.mod, reflexive=True)

    # def __rand__(self, other):
    #     return self._binary_op(other, self.and_, reflexive=True)

    # def __rxor__(self, other):
    #     return self._binary_op(other, self.xor, reflexive=True)

    # def __ror__(self, other):
    #     return self._binary_op(other, self.or_, reflexive=True)

    def _inplace_binary_op(self, other, f):
        raise NotImplementedError

    # def __iadd__(self, other):
    #     return self._inplace_binary_op(other, self.iadd)

    # def __isub__(self, other):
    #     return self._inplace_binary_op(other, self.isub)

    # def __imul__(self, other):
    #     return self._inplace_binary_op(other, self.imul)

    # def __ipow__(self, other):
    #     return self._inplace_binary_op(other, self.ipow)

    # def __itruediv__(self, other):
    #     return self._inplace_binary_op(other, self.itruediv)

    # def __ifloordiv__(self, other):
    #     return self._inplace_binary_op(other, self.ifloordiv)

    # def __imod__(self, other):
    #     return self._inplace_binary_op(other, self.imod)

    # def __iand__(self, other):
    #     return self._inplace_binary_op(other, self.iand)

    # def __ixor__(self, other):
    #     return self._inplace_binary_op(other, self.ixor)

    # def __ior__(self, other):
    #     return self._inplace_binary_op(other, self.ior)

    def _unary_op(self, f, *args, **kwargs):
        raise NotImplementedError

    def __neg__(self):
        return self._unary_op(self.negative)

    def __pos__(self):
        return self._unary_op(self.positive)

    def __abs__(self):
        return self._unary_op(self.abs)

    # def __invert__(self):
    #     return self._unary_op(self.invert)

    # def round(self, *args, **kwargs):
    #     return self._unary_op(ops.round_, *args, **kwargs)

    # def argsort(self, *args, **kwargs):
    #     return self._unary_op(ops.argsort, *args, **kwargs)

    # def conj(self, *args, **kwargs):
    #     return self._unary_op(ops.conj, *args, **kwargs)

    # def conjugate(self, *args, **kwargs):
    #     return self._unary_op(ops.conjugate, *args, **kwargs)

    __add__.__doc__ = operator.add.__doc__
    __sub__.__doc__ = operator.sub.__doc__
    __mul__.__doc__ = operator.mul.__doc__
    __pow__.__doc__ = operator.pow.__doc__
    __truediv__.__doc__ = operator.truediv.__doc__
    # __floordiv__.__doc__ = operator.floordiv.__doc__
    # __mod__.__doc__ = operator.mod.__doc__
    # __and__.__doc__ = operator.and_.__doc__
    # __xor__.__doc__ = operator.xor.__doc__
    # __or__.__doc__ = operator.or_.__doc__
    __lt__.__doc__ = operator.lt.__doc__
    __le__.__doc__ = operator.le.__doc__
    __gt__.__doc__ = operator.gt.__doc__
    __ge__.__doc__ = operator.ge.__doc__
    __eq__.__doc__ = operator.eq.__doc__
    __ne__.__doc__ = operator.ne.__doc__
    __radd__.__doc__ = operator.add.__doc__
    __rsub__.__doc__ = operator.sub.__doc__
    __rmul__.__doc__ = operator.mul.__doc__
    __rpow__.__doc__ = operator.pow.__doc__
    __rtruediv__.__doc__ = operator.truediv.__doc__
    # __rfloordiv__.__doc__ = operator.floordiv.__doc__
    # __rmod__.__doc__ = operator.mod.__doc__
    # __rand__.__doc__ = operator.and_.__doc__
    # __rxor__.__doc__ = operator.xor.__doc__
    # __ror__.__doc__ = operator.or_.__doc__
    # __iadd__.__doc__ = operator.iadd.__doc__
    # __isub__.__doc__ = operator.isub.__doc__
    # __imul__.__doc__ = operator.imul.__doc__
    # __ipow__.__doc__ = operator.ipow.__doc__
    # __itruediv__.__doc__ = operator.itruediv.__doc__
    # __ifloordiv__.__doc__ = operator.ifloordiv.__doc__
    # __imod__.__doc__ = operator.imod.__doc__
    # __iand__.__doc__ = operator.iand.__doc__
    # __ixor__.__doc__ = operator.ixor.__doc__
    # __ior__.__doc__ = operator.ior.__doc__
    __neg__.__doc__ = operator.neg.__doc__
    __pos__.__doc__ = operator.pos.__doc__
    __abs__.__doc__ = operator.abs.__doc__
    # __invert__.__doc__ = operator.invert.__doc__
    # round.__doc__ = ops.round_.__doc__
    # argsort.__doc__ = ops.argsort.__doc__
    # conj.__doc__ = ops.conj.__doc__
    # conjugate.__doc__ = ops.conjugate.__doc__
