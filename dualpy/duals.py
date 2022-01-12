"""The dual type for dualpy"""

import numpy as np
import astropy.units as units
import fnmatch

from .dual_helpers import _setup_dual_operation, _per_rad, _broadcast_jacobians
from .jacobians import (
    _setitem_jacobians,
    _join_jacobians,
    _concatenate_jacobians,
    _stack_jacobians,
)


__all__ = ["dlarray", "nan_to_num_jacobians", "_setup_dual_operation", "tensordot"]


class dlarray(units.Quantity):
    """A combination of an astropy Quantity (which is in turn numpy array
    and a collection of jacobian information stored as a optionally
    sparse 2D matrix.

    This follows the recommendations on subclassing ndarrays in the
    numpy docum2entation.

    """

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance.
        # We first cast to be our class type
        obj = units.Quantity(input_array).view(cls)
        if hasattr(input_array, "unit"):
            obj._set_unit(input_array.unit)
        if hasattr(input_array, "jacobians"):
            obj.jacobians = input_array.jacobians
        # Finally, we return the newly created object
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object outing from
        # ndarray.__new__(dual, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. dlarray():
        #    jacobians is None
        #    (we're in the middle of the dual.__new__
        #    constructor, and dlarray.jacobians will be set when we return to
        #    dlarray.__new__)
        if obj is None:
            return
        # From view casting - e.g arr.view(dual):
        #    obj is arr
        #    (type(obj) can be dual)
        # From new-from-template - e.g a_dual[:3]
        #    type(obj) is dual
        #
        # Note that it is here, rather than in the __new__ method, that we set the
        # default value for 'jacobians', because this method sees all creation of
        # default objects - with the dual.__new__ constructor, but also with
        # arr.view(dual).
        self.jacobians = getattr(obj, "jacobians", {})
        # We might need to put more here once we're doing reshapes and
        # stuff, I'm not sure.
        # We do not need to return anything

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        # The comparators can just call their astropy.Quantity equivalents, I'm going to
        # blythely assert that we don't care to compare jacobians.  Later, we may decide
        # that other operators fall into this category.
        if ufunc in (
            np.equal,
            np.not_equal,
            np.greater,
            np.less,
            np.greater_equal,
            np.less_equal,
        ):
            return ufunc(units.Quantity(self), units.Quantity(args[1]))
        # Also, do the same for some unary operators
        if ufunc in (np.isfinite,):
            return ufunc(units.Quantity(self))
        # Otherwise, we look for this same ufunc in our own type and
        # try to invoke that.
        # However, first some intervention
        dlufunc = getattr(dlarray, ufunc.__name__, None)
        if dlufunc is None:
            raise NotImplementedError(
                f"No implementation for ufunc {ufunc}, method {method}"
            )
            return NotImplemented
        result = dlufunc(*args, **kwargs)
        # result._check()
        return result

    def __array_function__(self, func, types, args, kwargs):
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)
        elif func in FALLTHROUGH_FUNCTIONS:
            return super().__array_function__(func, types, args, kwargs)
        elif func in RECAST_FUNCTIONS:
            # This doesn't work and generates a weird c-runtime error. I've obviated the
            # need for it in any case (thus far).
            if types != (dlarray,):
                return NotImplemented
            qargs = (units.Quantity(arg) for arg in args)
            return super().__array_function__(func, (units.Quantity,), qargs, kwargs)
        else:
            return NotImplemented

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        out = dlarray(units.Quantity(self))
        for name, jacobian in self.jacobians.items():
            out.jacobians[name] = jacobian
        return out

    def __deepcopy__(self, memo):
        # If we don't define this, ``copy.deepcopy(quantity)`` will
        # return a bare Numpy array.
        result = self.copy()
        return result

    def __getitem__(self, key):
        out = dlarray(units.Quantity(self).__getitem__(key))
        for name, jacobian in self.jacobians.items():
            out.jacobians[name] = jacobian._getjitem(out.shape, key)
        return out

    def __setitem__(self, key, value):
        s, v, sj, vj, out_ = _setup_dual_operation(self, value, broadcast=False)
        s.__setitem__(key, value)
        # Doing a setitem on the Jacobians requires some more intimate knowledge so let
        # the jacobians module handle it.
        _setitem_jacobians(key, s, sj, vj)

    def __eq__(a, b):
        a_, b_, aj, bj, out = _setup_dual_operation(a, b)
        return a_ == b_

    def __ne__(a, b):
        a_, b_, aj, bj, out = _setup_dual_operation(a, b)
        return a_ != b_

    def __gt__(a, b):
        a_, b_, aj, bj, out = _setup_dual_operation(a, b)
        return a_ > b_

    def __lt__(a, b):
        a_, b_, aj, bj, out = _setup_dual_operation(a, b)
        return a_ < b_

    def __le__(a, b):
        a_, b_, aj, bj, out = _setup_dual_operation(a, b)
        return a_ <= b_

    def _check(self, name="<unknown>"):
        """Check consistency of a dual"""
        for jname, jacobian in self.jacobians.items():
            assert self.unit == jacobian.dependent_unit, (
                f"The {jname} Jacobian for {name} has the wrong dependent "
                f"units ({jacobian.dependent_unit} rather "
                f"than {self.unit})"
            )
            assert self.shape == jacobian.dependent_shape, (
                f"The {jname} Jacobian for {name} has the wrong dependent "
                f"size ({jacobian.dependent_shape} rather "
                f"than {self.shape})"
            )

    def to(self, unit, **kwargs):
        # If it's a no-op, then take advantage of that
        if self.unit is unit:
            return self
        out_ = units.Quantity(self).to(unit, **kwargs)
        out = dlarray(out_)
        # This will probably (hopefully!) fail if we're using one of
        # those function unit things (such as dB).
        if self.hasJ:
            scale = self.unit.to(unit, **kwargs) << unit / self.unit
            out._chain_rule(self, scale)
        return out

    def decompose(self):
        return self.to(self.unit.decompose())

    def _ones_like(a):
        result_ = np.ones_like(units.Quantity(a))
        result = dlarray(result_)
        return result

    def hasJ(self):
        return len(self.jacobians) != 0

    # Update all the Jacobians in a dlarray by premultiplying them by
    # a diagonal.  Most of the work is done by the premul_diag method
    # for the Jacobian itself.
    def _chain_rule(self, a, d, unit=None):
        """Finish up the computation of Jacobians
        a: dlarray - original input to ufunc
        d: ndarray - d(self)/da"""
        if unit is not None:
            for name, jacobian in a.jacobians.items():
                self.jacobians[name] = jacobian.to(unit).premul_diag(d)
        else:
            for name, jacobian in a.jacobians.items():
                self.jacobians[name] = jacobian.premul_diag(d)

    def ravel(self, *args):
        return np.reshape(self, (self.size,))

    # Now a whole bunch of binary operators
    def add(a, b, out=None):
        a_, b_, aj, bj, out = _setup_dual_operation(a, b, out=out)
        if out is None:
            out = dlarray(a_ + b_)
        else:
            if not isinstance(out, dlarray):
                out = dlarray(out)
            out[:] = a_ + b_
        for name, jacobian in aj.items():
            out.jacobians[name] = jacobian
        for name, jacobian in bj.items():
            if name in out.jacobians:
                out.jacobians[name] += jacobian
            else:
                out.jacobians[name] = jacobian.to(out.unit)
        return out

    def subtract(a, b, out=None):
        a_, b_, aj, bj, out = _setup_dual_operation(a, b, out=out)
        if out is None:
            out = dlarray(a_ - b_)
        else:
            if not isinstance(out, dlarray):
                out = dlarray(out)
            out[:] = a_ - b_
        for name, jacobian in aj.items():
            out.jacobians[name] = jacobian
        for name, jacobian in bj.items():
            if name in out.jacobians:
                out.jacobians[name] += -jacobian
            else:
                out.jacobians[name] = -jacobian.to(out.unit)
        return out

    def multiply(a, b, out=None):
        a_, b_, aj, bj, out = _setup_dual_operation(a, b, out=out)
        # Because out may share memory with a or b, we need to do the
        # Jacobians first as they need access to a and b
        # unadulterated.
        out_jacobians = {}
        for name, jacobian in aj.items():
            out_jacobians[name] = jacobian.premul_diag(b_)
        for name, jacobian in bj.items():
            if name in out_jacobians:
                out_jacobians[name] += jacobian.premul_diag(a_)
            else:
                out_jacobians[name] = jacobian.premul_diag(a_)
        if out is None:
            out = dlarray(a_ * b_)
        else:
            if not isinstance(out, dlarray):
                out = dlarray(out)
            new_unit = a_.unit * b_.unit
            out = out << new_unit
            out[:] = a_ * b_
        out.jacobians = out_jacobians
        return out

    def rmultiply(a, b, out=None):
        a_, b_, aj, bj, out = _setup_dual_operation(a, b, out=out)
        # Because out may share memory with a or b, we need to do the
        # Jacobians first as they need access to a and b
        # unadulterated.
        out_jacobians = {}
        for name, jacobian in aj.items():
            out_jacobians[name] = jacobian.premul_diag(b_)
        for name, jacobian in bj.items():
            if name in out_jacobians:
                out_jacobians[name] += jacobian.premul_diag(a_)
            else:
                out_jacobians[name] = jacobian.premul_diag(a_).to(out.unit)
        if out is None:
            out = dlarray(a_ * b_)
        else:
            if not isinstance(out, dlarray):
                out = dlarray(out)
            out = out << a_.unit * b_.unit
            out[:] = a_ * b_
        out.jacobians = out_jacobians
        return out

    def true_divide(a, b, out=None):
        a_, b_, aj, bj, out = _setup_dual_operation(a, b, out=out)
        # Note that, the way this is constructed, it's OK if out
        # shares memory with a and/or b, neither a or b are used after
        # out is filled. We do need to keep 1/b though.
        if b_.dtype.char in np.typecodes["AllInteger"]:
            b_ = 1.0 * b_
        r_ = np.reciprocal(b_)
        if out is None:
            out = dlarray(a_ * r_)
        else:
            if not isinstance(out, dlarray):
                out = dlarray(out)
            out = out << a_.unit / b_.unit
            out[:] = a_ * r_
        out_ = units.Quantity(out)
        # We're going to do the quotient rule as (1/b)a' - (a/(b^2))b'
        # The premultiplier for a' is the reciprocal _r, computed above
        # The premultiplier for b' is that times the result
        c_ = out_ * r_
        for name, jacobian in aj.items():
            out.jacobians[name] = jacobian.premul_diag(r_)
        for name, jacobian in bj.items():
            if name in out.jacobians:
                out.jacobians[name] += -jacobian.premul_diag(c_)
            else:
                out.jacobians[name] = -jacobian.premul_diag(c_).to(out.unit)
        return out

    def remainder(a, b, out=None):
        raise NotImplementedError("Pretty sure this is wrong")
        a_, b_, aj, bj, out = _setup_dual_operation(a, b, out=out)
        out = dlarray(a_ % b_)
        # The resulting Jacobian is simply a copy of the jacobian for a, b has no impact
        for name, jacobian in aj.items():
            out.jacobians[name] = jacobian
        return out

    def power(a, b):
        # In case it's more efficient this divides up according to
        # whether either a or b or both are duals.  Note that the
        # "both" case has not been tested, so is currently disabled.
        a_, b_, aj, bj, out = _setup_dual_operation(a, b)
        if isinstance(a, dlarray) and isinstance(b, dlarray):
            # This has never been tested so, for now, I'm goint to
            # flag it as not implemented.  However, there is code below,
            # as you can see.
            return NotImplemented
            # a**(b-1)*(b*da/dx+a*log(a)*db/dx)
            # Multiply it out and we get:
            out_ = a ** b_
            dA_ = b_ * a_ ** (b_ - 1)
            dB_ = out_ * np.log(a_)
            out = dlarray(out_)
            for name, jacobian in aj.items():
                out.jacobians[name] = jacobian.premul_diag(dA_)
            for name, jacobian in bj.items():
                if name in out.jacobians:
                    out.jacobians[name] += jacobian.premul_diag(dB_)
                else:
                    out.jacobians[name] = jacobian.premul_diag(dB_).to(out.unit)
        elif isinstance(a, dlarray):
            out = dlarray(a_ ** b)
            d_ = b * a_ ** (b - 1)
            for name, jacobian in aj.items():
                out.jacobians[name] = jacobian.premul_diag(d_)
        elif isinstance(b, dlarray):
            a_, b_, aj, bj, out = _setup_dual_operation(a, b)
            out_ = a ** b_
            out = dlarray(out_)
            for name, jacobian in bj.items():
                out.jacobians[name] = jacobian.premul_diag(out_ * np.log(a_)).to(
                    out.unit
                )
        return out

    def arctan2(a, b):
        # See Wikpedia page on atan2 which conveniently lists the derivatives
        a_, b_, aj, bj, out = _setup_dual_operation(a, b)
        out = dlarray(np.arctan2(a_, b_))
        rr2 = units.rad * np.reciprocal(a_ ** 2 + b_ ** 2)
        for name, jacobian in aj.items():
            out.jacobians[name] = jacobian.premul_diag(b_ * rr2).to(out.unit)
        for name, jacobian in bj.items():
            if name in out.jacobians:
                out.jacobians[name] += jacobian.premul_diag(-a_ * rr2).to(out.unit)
            else:
                out.jacobians[name] = jacobian.premul_diag(-a_ * rr2).to(out.unit)
        return out

    def __mul__(self, other):
        # Needed for dual * unit case
        if isinstance(other, (units.UnitBase, str)):
            return self * units.Quantity(1.0, other)
        return super().__mul__(other)

    def __truediv__(self, other):
        # Needed for dual * unit case
        if isinstance(other, (units.UnitBase, str)):
            return self / units.Quantity(1.0, other)
        return super().__truediv__(other)

    def __lshift__(self, other):
        out = dlarray(np.array(self) << other)
        for name, jacobian in self.jacobians.items():
            out.jacobians[name] = jacobian << other
        return out

    # Now some unary operators

    def negative(a):
        a_ = units.Quantity(a)
        out = dlarray(-a_)
        for name, jacobian in a.jacobians.items():
            out.jacobians[name] = -jacobian
        return out

    def reciprocal(a):
        a_ = units.Quantity(a)
        out_ = 1.0 / a_
        out = dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, -(out_ ** 2))
        return out

    def square(a):
        a_ = units.Quantity(a)
        out_ = np.square(a_)
        out = dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, 2 * a_)
        return out

    def sqrt(a):
        a_ = units.Quantity(a)
        out_ = np.sqrt(a_)
        out = dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, 1.0 / (2 * out_))
        return out

    def exp(a):
        a_ = units.Quantity(a)
        out_ = np.exp(a_)
        out = dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, out_, unit=units.dimensionless_unscaled)
        return out

    def log(a):
        a_ = units.Quantity(a)
        out_ = np.log(a_)
        out = dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, 1.0 / a_, unit=units.dimensionless_unscaled)
        return out

    def log10(a):
        return (1.0 / np.log(10.0)) * np.log(a)

    def transpose(a, axes=None):
        out = dlarray(units.Quantity(a).transpose(axes))
        for name, jacobian in a.jacobians.items():
            out.jacobians[name] = jacobian.transpose(axes, out.shape)
        return out

    def matmul(a, b):
        raise NotImplementedError(
            "No implementation of matmul for duals, consider tensordot?"
        )

    def rmatmul(a, b):
        raise NotImplementedError(
            "No implementation of rmatmul for duals, consider tensordot?"
        )

    @property
    def T(self):
        return self.transpose()

    # A note on the trigonometric cases, we'll cut out the middle man
    # here and go straight to numpy, forcing the argument into
    # radians.
    def sin(a):
        a_rad = a.to(units.rad)
        out_ = np.sin(a_rad.value) << units.dimensionless_unscaled
        out = dlarray(out_)
        if a_rad.hasJ:
            out._chain_rule(a_rad, np.cos(a_rad.value) << _per_rad, unit=units.rad)
        return out

    def cos(a):
        a_rad = a.to(units.rad)
        out_ = np.cos(a_rad.value) << units.dimensionless_unscaled
        out = dlarray(out_)
        if a_rad.hasJ:
            out._chain_rule(a_rad, -np.sin(a_rad.value) << _per_rad, unit=units.rad)
        return out

    def tan(a):
        a_rad = a.to(units.rad)
        out_ = np.tan(a_rad.value) << units.dimensionless_unscaled
        out = dlarray(out_)
        if a_rad.hasJ:
            out._chain_rule(
                a_rad, 1.0 / (np.cos(a_rad.value) ** 2) << _per_rad, unit=units.rad
            )
        return out

    def arcsin(a):
        a_ = units.Quantity(a)
        out_ = np.arcsin(a_)
        out = dlarray(out_)
        if a.hasJ:
            out._chain_rule(
                a, units.rad / np.sqrt(1 - a_ ** 2), unit=units.dimensionless_unscaled
            )
        return out

    def arccos(a):
        a_ = units.Quantity(a)
        out_ = np.arccos(a_)
        out = dlarray(out_)
        if a.hasJ:
            out._chain_rule(
                a, -units.rad / np.sqrt(1 - a_ ** 2), unit=units.dimensionless_unscaled
            )
        return out

    def arctan(a):
        a_ = units.Quantity(a)
        out_ = np.arctan(a_)
        out = dlarray(out_)
        if a.hasJ:
            out._chain_rule(
                a, units.rad / (1 + a_ ** 2), unit=units.dimensionless_unscaled
            )
        return out

    def sinh(a):
        a_rad = a.to(units.rad)
        out_ = np.sinh(a_rad.value) << units.dimensionless_unscaled
        out = dlarray(out_)
        if a_rad.hasJ:
            out._chain_rule(a_rad, np.cosh(a_rad.value) << _per_rad, unit=units.rad)
        return out

    def cosh(a):
        a_rad = a.to(units.rad)
        out_ = np.cosh(a_rad.value) << units.dimensionless_unscaled
        out = dlarray(out_)
        if a.hasJ:
            out._chain_rule(a_rad, np.sinh(a_rad.value) << _per_rad, unit=units.rad)
        return out

    def tanh(a):
        a_rad = a.to(units.rad)
        out_ = np.tanh(a_rad.value) << units.dimensionless_unscaled
        out = dlarray(out_)
        if a.hasJ:
            out._chain_rule(
                a_rad, 1.0 / np.cosh(a_rad.value) ** 2 << _per_rad, unit=units.rad
            )
        return out

    def arcsinh(a):
        a_ = units.Quantity(a)
        out_ = np.arcsinh(a_)
        out = dlarray(out_)
        if a.hasJ:
            out._chain_rule(
                a, units.rad / np.sqrt(a_ ** 2 + 1), unit=units.dimensionless_unscaled
            )
        return out

    def arccosh(a):
        a_ = units.Quantity(a)
        out_ = np.arccosh(a_)
        out = dlarray(out_)
        if a.hasJ:
            out._chain_rule(
                a, units.rad / np.sqrt(a_ ** 2 - 1), unit=units.dimensionless_unscaled
            )
        return out

    def arctanh(a):
        a_ = units.Quantity(a)
        out_ = np.arctanh(a_)
        out = dlarray(out_)
        if a.hasJ:
            out._chain_rule(
                a, units.rad / (1 - a_ ** 2), unit=units.dimensionless_unscaled
            )
        return out

    def absolute(a):
        a_ = units.Quantity(a)
        out_ = np.absolute(a_)
        out = dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, np.sign(a_))
        return out

    def abs(a):
        return np.absolute(a)

    def maximum(a, b, out=None, **kwargs):
        if out is not None:
            raise NotImplementedError("dlarray.maximum cannot support out")
        if len(kwargs) != 0:
            raise NotImplementedError("dlarray.maximum cannot support non-empty kwargs")
        a_, b_, aj, bj, out = _setup_dual_operation(a, b, out=out)
        out = dlarray(np.maximum(a_, b_))
        if a.hasJ or b.hasJ:
            factor = a_ >= b_
            if hasattr(a, "jacobians"):
                out._chain_rule(a, factor.astype(int))
            if hasattr(b, "jacobians"):
                out._chain_rule(b, np.logical_not(factor).astype(int))
        return out

    def minimum(a, b, out=None, **kwargs):
        if out is not None:
            raise NotImplementedError("dlarray.minimum cannot support out")
        if len(kwargs) != 0:
            raise NotImplementedError("dlarray.minimum cannot support non-empty kwargs")
        a_, b_, aj, bj, out = _setup_dual_operation(a, b, out=out)
        out = dlarray(np.minimum(a_, b_))
        if a.hasJ or b.hasJ:
            factor = a_ <= b_
            if hasattr(a, "jacobians"):
                out._chain_rule(a, factor.astype(int))
            if hasattr(b, "jacobians"):
                out._chain_rule(b, np.logical_not(factor).astype(int))
        return out

    def floor(a):
        # For now, when we take the floor, let's assume no Jacobians survive
        return np.floor(units.Quantity(a))

    def flatten(self, order="C"):
        result = dlarray(units.Quantity(self).flatten(order))
        for name, jacobian in self.jacobians.items():
            result.jacobians[name] = jacobian.flatten(order)
        return result

    def squeeze(self, axis=None):
        """Remove axis of length 1 from self"""
        result = dlarray(units.Quantity(self).squeeze(axis))
        for name, jacobian in self.jacobians.items():
            result.jacobians[name] = jacobian.reshape(result.shape)
        return result

    def remove_jacobian(self, name=None, wildcard=None, remain_dual=False):
        # Remove a named jacobian from a dual.  If none are left, then
        # possibly demote back to an astropy quantity, unless
        # remain_dual is set
        if name is not None and wildcard is not None:
            raise ValueError("Cannote set both name and wildcard")
        if name is None and wildcard is None:
            raise ValueError("Must set either name or wildcard")
        if name is not None:
            self.jacobians.pop(name)
        if wildcard is not None:
            # Build a list of keys to remove
            keys = [
                key for key in self.jacobians.keys() if fnmatch.fnmatch(key, wildcard)
            ]
            # Remove them (done this way because Python complains if
            # iterating over a list of keys that keeps changing as
            # keys are removed)
            for key in keys:
                self.jacobians.pop(key)
        # Now, are any jacobians leff ("if" below tests True if so)
        if self.jacobians or remain_dual:
            return self
        else:
            return units.Quantity(self)

    def reshape(array, *args, **kwargs):
        return _reshape(array, *args, **kwargs)

    @property
    def uvalue(self):
        return units.Quantity(self)


# -------------------------------------- Now the array functions
HANDLED_FUNCTIONS = {}
FALLTHROUGH_FUNCTIONS = []
RECAST_FUNCTIONS = []  # [np.empty_like, np.zeros_like, np.ones_like]


def implements(numpy_function):
    """Register an __array_function__ implementation for dlarray objects."""

    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


# @implements(np.amin)
# def amin(a, axis=None, out=None, keepdims=False, initial=None, where=None):
#     if out is not None:
#         raise NotImplementedError("Cannot call np.amin on duals with out")
#     if initial is not None:
#         raise NotImplementedError("Cannot call np.amin on duals with initial")
#     if where is not None:
#         raise NotImplementedError("Cannot call np.amin on duals with where")
#     i = np.argmin(np.array(a), axis=axis)
#     if keepdims:
#         pass


@implements(np.sum)
def sum(a, axis=None, dtype=None, keepdims=False):
    a_, aj, out = _setup_dual_operation(a)
    out = dlarray(np.sum(a_, axis=axis, dtype=dtype, keepdims=keepdims))
    for name, jacobian in aj.items():
        out.jacobians[name] = jacobian.sum(
            out.shape, axis=axis, dtype=dtype, keepdims=keepdims
        )
    return out


@implements(np.cumsum)
def cumsum(a, axis=None, dtype=None, out=None):
    if out is not None:
        raise NotImplementedError("out not supported for dual cumsum (yet?)")
    a_, aj, out = _setup_dual_operation(a)
    out = dlarray(np.cumsum(a_, axis=axis, dtype=dtype))
    for name, jacobian in aj.items():
        out.jacobians[name] = jacobian.cumsum(axis)
    return out


# One of these days I should look into whether broadcast_arrays and
# broadcast_to really need the subok argument, given that it's ignored.
@implements(np.broadcast_arrays)
def broadcast_arrays(*args, subok=False):
    values = []
    for a in args:
        values.append(a.value)
    result_ = np.broadcast_arrays(*values, subok=subok)
    shape = result_[0].shape
    result = []
    for i, a in enumerate(args):
        thisResult = dlarray(result_[i] << a.unit)
        if hasattr(a, "jacobians"):
            thisResult.jacobians = _broadcast_jacobians(a.jacobians, shape)
        result.append(thisResult)
    return result


@implements(np.broadcast_to)
def broadcast_to(array, shape, subok=False):
    result_ = np.broadcast_to(array.value, shape, subok=subok) << array.unit
    result = dlarray(result_)
    result.jacobians = _broadcast_jacobians(array.jacobians, shape)
    return result


def _reshape(array, *args, **kwargs):
    result = dlarray(units.Quantity(array).reshape(*args, **kwargs))
    for name, jacobian in array.jacobians.items():
        result.jacobians[name] = jacobian.reshape(result.shape)
    return result


@implements(np.reshape)
def reshape(array, *args, **kwargs):
    return _reshape(array, *args, **kwargs)


@implements(np.atleast_1d)
def atleast_1d(*args):
    result = []
    for a in args:
        a1d = np.atleast_1d(units.Quantity(a))
        a1d.jacobians = a.jacobians.copy()
        result.append(a1d)
    return tuple(result)


@implements(np.diff)
def diff(array, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    result_ = np.diff(array.value, n, axis, prepend, append) << array.unit
    dependent_shape = result_.shape
    result = dlarray(result_)
    for name, jacobian in array.jacobians.items():
        result.jacobians[name] = jacobian.diff(
            dependent_shape, n, axis, prepend, append
        )
    return result


@implements(np.where)
def where(condition, a=None, b=None):
    if a is None or b is None:
        return NotImplemented
    cond_, a_, b_, condj, aj, bj, out = _setup_dual_operation(condition, a, b)
    if condj:
        raise ValueError("Jacobians not allowed on condition argument in 'where'")
    out = dlarray(np.where(cond_, a_, b_))
    # Now go through the jacobians and insert them where the condition
    # applies, otherwise they're zero.
    for name, jacobian in aj.items():
        out.jacobians[name] = jacobian.premul_diag(cond_)
    for name, jacobian in bj.items():
        if name in out.jacobians:
            out.jacobians[name] += jacobian.premul_diag(np.logical_not(cond_))
        else:
            out.jacobians[name] = jacobian.premul_diag(np.logical_not(cond_))
    return out


@implements(np.insert)
def insert(arr, obj, values, axis=None):
    # Note that this is supposed to flatten the array first if axis is None.  By doing
    # that here rather than relying on the original np.insert to do it, we can handle
    # the issue with the Jacobians.
    if axis is None:
        axis = 0
        arr = arr.flatten()
        try:
            # Try to flatten values also
            values = values.flatten()
        except AttributeError:
            pass
    result = dlarray(np.insert(units.Quantity(arr), obj, units.Quantity(values), axis))
    # Now deal with the Jacobians, first deal with anythat are in the values to add
    result.jacobians = _join_jacobians(arr, values, obj, axis, result.shape)
    return result


@implements(np.append)
def append(arr, values, axis=None):
    """Append values to the end of an array"""
    # Note that this is supposed to flatten the array first if axis is None.  By doing
    # that here rather than relying on the original np.insert to do it, we can handle
    # the issue with the Jacobians.
    if axis is None:
        axis = 0
        arr = arr.flatten()
        try:
            # Try to flatten values also
            values = values.flatten()
        except AttributeError:
            pass
    result = dlarray(np.append(units.Quantity(arr), units.Quantity(values), axis))
    result.jacobians = _join_jacobians(arr, values, arr.shape[axis], axis, result.shape)
    return result


@implements(np.searchsorted)
def searchsorted(a, v, side="left", sorter=None):
    return np.searchsorted(
        units.Quantity(a), units.Quantity(v), side=side, sorter=sorter
    )


@implements(np.clip)
def clip(a, a_min, a_max, out=None, **kwargs):
    if type(a_min) is dlarray:
        raise NotImplementedError(
            "dlarray.clip does not (currently) support dual "
            "for a_min, use dualarray.minimum"
        )
    if type(a_max) is dlarray:
        raise NotImplementedError(
            "dlarray.clip does not (currently) support dual "
            "for a_max, use dualarray.maximum"
        )
    if out is not None:
        raise NotImplementedError("dlarray.clip cannot support out")
    if len(kwargs) != 0:
        raise NotImplementedError("dlarray.clip cannot support non-empty kwargs")
    out = dlarray(np.clip(units.Quantity(a), a_min, a_max))
    if a.hasJ:
        factor = np.logical_and(a >= a_min, a <= a_max)
        out._chain_rule(a, factor.astype(int))
    return out


@implements(np.nan_to_num)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None, jacobians_only=False):
    x_, j, out = _setup_dual_operation(x)
    if jacobians_only:
        result = dlarray(x_)
    else:
        result = dlarray(
            np.nan_to_num(x_, copy=copy, nan=nan, posinf=posinf, neginf=neginf)
        )
    for name, jacobian in j.items():
        result.jacobians[name] = jacobian.nan_to_num(
            copy=copy, nan=nan, posinf=posinf, neginf=neginf
        )
    return result


@implements(np.real)
def real(a):
    out = dlarray(np.real(units.Quantity(a)))
    for name, jacobian in a.jacobians.items():
        out.jacobians[name] = jacobian.real()
    return out


@implements(np.empty_like)
def empty_like(prototype, dtype=None, order="K", subok=True, shape=None):
    return dlarray(
        np.empty_like(units.Quantity(prototype), dtype, order, subok, shape)
        << prototype.unit
    )


@implements(np.zeros_like)
def zeros_like(prototype, dtype=None, order="K", subok=True, shape=None):
    return dlarray(
        np.zeros_like(units.Quantity(prototype), dtype, order, subok, shape)
        << prototype.unit
    )


@implements(np.ones_like)
def ones_like(prototype, dtype=None, order="K", subok=True, shape=None):
    return dlarray(
        np.ones_like(units.Quantity(prototype), dtype, order, subok, shape)
        << prototype.unit
    )


@implements(np.expand_dims)
def expand_dims(a, axis):
    result = dlarray(np.expand_dims(units.Quantity(a), axis))
    for name, jacobian in a.jacobians.items():
        result.jacobians[name] = jacobian.reshape(result.shape)
    return result


@implements(np.concatenate)
def concatenate(values, axis=0, out=None):
    if out is not None:
        raise ValueError("Cannot concatenate duals into an out")
    # If axis is zero, flatten the inputs
    if axis is None:
        values = [value.flatten() for value in values]
        axis = 0
    # Populate the result
    values_ = [units.Quantity(value) for value in values]
    result_ = np.concatenate(values_, axis, out)
    result = dlarray(result_)
    # Get the Jacobians concatenated
    result.jacobians = _concatenate_jacobians(values, axis, result.shape)
    return result


@implements(np.stack)
def stack(arrays, axis=0, out=None):
    if out is not None:
        raise ValueError("Cannot stack duals into an out")
    # Populate the result
    arrays_ = [units.Quantity(array) for array in arrays]
    result_ = np.stack(arrays_, axis, out)
    result = dlarray(result_)
    # Get the Jacobians stacked
    result.jacobians = _stack_jacobians(arrays, axis, result.shape)
    return result


@implements(np.ndim)
def ndim(array):
    return array.ndim


@implements(np.transpose)
def transpose(array, axes=None):
    return array.transpose(axes)


@implements(np.tensordot)
def tensordot(a, b, axes):
    import sparse as st

    a_, b_, aj, bj, out = _setup_dual_operation(a, b, out=None, broadcast=False)
    result_unit = getattr(a_, "unit", units.dimensionless_unscaled) * getattr(
        b_, "unit", units.dimensionless_unscaled
    )
    result = dlarray(st.tensordot(a_, b_, axes) * result_unit)
    # Now deal with the Jacobians.  For this, we need to ensure that axes are in the
    # (2,) array-like form that is the second version np.tensordot can accept them.
    if isinstance(axes, int):
        axes = [list(range(a.ndim - axes, a.ndim)), list(range(axes))]
    # Remove units from a_ and b_ for doing the tensor dot product
    a_no_unit = getattr(a_, "value", a_)
    b_no_unit = getattr(b_, "value", b_)
    for name, jacobian in aj.items():
        result.jacobians[name] = jacobian.tensordot(
            b_no_unit, axes, dependent_unit=result.unit
        )
    for name, jacobian in bj.items():
        if name in result.jacobians:
            result.jacobains[name] += jacobian.rtensordot(
                a_no_unit,
                axes,
                dependent_unit=result.unit,
            )
        else:
            result.jacobians[name] = jacobian.rtensordot(
                a_no_unit,
                axes,
                dependent_unit=result.unit,
            )
    return result


def nan_to_num_jacobians(x, copy=True, nan=0.0, posinf=None, neginf=None):
    return nan_to_num(
        x, copy=copy, nan=nan, posinf=posinf, neginf=neginf, jacobians_only=True
    )
