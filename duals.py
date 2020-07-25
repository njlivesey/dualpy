"""The dual type for dualpy"""

import numpy as np
import scipy.sparse as sparse
import scipy.special as special
import scipy.constants as constants
import astropy.units as units
import fnmatch
import itertools
import copy

from .jacobians import *
from .dual_helpers import *

__all__ = ["dlarray"]

class dlarray(units.Quantity):
    """A combination of an astropy Quantity (which is in turn numpy array and a collection of jacobian
    information stored as a optionally sparse 2D matrix.

    This follows the recommendations on subclassing ndarrays in the
    numpy docum2entation."""

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance.
        # We first cast to be our class type
        obj=units.Quantity(input_array).view(cls)
        if hasattr(input_array, "unit"):
            obj._set_unit(input_array.unit)
        if hasattr(input_array, "jacobians"):
            obj.jacobians=input_array.jacobians
        # Finally, we return the newly created object
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object outing from
        # ndarray.__new__(dual, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. dual():
        #    jacobians is None
        #    (we're in the middle of the dual.__new__
        #    constructor, and dual.jacobians will be set when we return to
        #    dual.__new__)
        if obj is None:
            return
        # From view casting - e.g arr.view(dual):
        #    obj is arr
        #    (type(obj) can be dual)
        # From new-from-template - e.g a_dual[:3]
        #    type(obj) is dual
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'jacobians', because this
        # method sees all creation of default objects - with the
        # dual.__new__ constructor, but also with
        # arr.view(dual).
        self.jacobians=getattr(obj, 'jacobians', {})
        # We might need to put more here once we're doing reshapes and
        # stuff, I'm not sure.
        # We do not need to return anything

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        # print(f"----------------- In __array_ufunc__")
        # print(f"type(self): {type(self)}")
        # print(f"self: {self}")
        # print(f"type(ufunc): {type(ufunc)}")
        # print(f"ufunc: {ufunc}")
        # print(f"type(method): {type(method)}")
        # print(f"method: {method}")
        # print(f"self: {self}")
        # for a,i in zip(args,np.arange(len(args))):
        #     print (f"args[{i}]: {a}")

        # The comparators can just call their numpy equivalents, I'm
        # going to blythely assert that we don't care to compare
        # jacobians.  Later, we may decide that other operators fall
        # into this category.
        if ufunc in (np.equal, np.not_equal,
                     np.greater, np.less,
                     np.greater_equal, np.less_equal):
            return ufunc(np.asarray(self), np.asarray(args[1]))
        # Also, do the same for some unary operators
        if ufunc in (np.isfinite,):
            return ufunc(np.asarray(self))
        # Otherwise, we look for this same ufunc in our own type and
        # try to invoke that.
        # if method == "reduce":
        #     print (args)
        #     print (kwargs)
        #     return 0
        # elif method == "__call__":
        # However, first some intervention
        dlufunc=getattr(dlarray, ufunc.__name__, None)
        if dlufunc is None:
            return NotImplemented
        return dlufunc(*args, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle dlarray objects
        #
        # Note: I've commented out the part below (which I thnk the
        # comments above are describing) to allow for cases where some
        # but not all arguments are duals.
        # if not all(issubclass(t, dlarray) for t in types):
        #    return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __getitem__(self, *args):
        print(f"---- Asked for get item on shape {self.shape}")
        print(f"Args is {args}")
        out = dlarray(units.Quantity(self).__getitem__(*args))
        for name, jacobian in self.jacobians.items():
            out.jacobians[name] = jacobian._getjitem(out.shape, *args)
        return out

    def _check(self, name="<unknown>"):
        # A routine to check that a dual is OK
        for jname, jacobian in self.jacobians.items():
            if self.unit != jacobian.dependent_unit:
                raise units.UnitsError(
                    f"The {jname} Jacobian for {name} has the wrong dependent units " +
                    f"({jacobian.dependent_unit} rather than {self.unit})")

    def to(self, unit, **kwargs):
        # If it's a no-op, then take advantage of that
        if self.unit == unit:
            return self
        out_ = units.Quantity(self).to(unit, **kwargs)
        out = dlarray(out_)
        # This will probably (hopefully!) fail if we're using one of
        # those function unit things (such as dB).
        if self.hasJ:
            scale = self.unit.to(unit) << unit/self.unit
            out._chain_rule(self, scale)
        return out

    # def decompose(self):
    #     return self.to(self.unit.decompose())

    def _ones_like(a):
        out=dlarray(a)
        out[:]=1.0
        return out

    def hasJ(self):
        return self.jacobians

    # Update all the Jacobians in a dlarray by premultiplying them by
    # a diagonal.  Most of the work is done by the premul_diag method
    # for the Jacobian itself.
    def _chain_rule(self, a, d):
        """Finish up the computation of Jacobians
           a: dlarray - original input to ufunc
           d: ndarray - d(self)/da"""
        for name, jacobian in a.jacobians.items():
            self.jacobians[name] = jacobian.premul_diag(d)

    def ravel(self, *args):
        out=dlarray(units.Quantity(self).ravel(*args))
        out.jacobians=self.jacobians
        return out

    # Now a whole bunch of binary operators
    def add(a, b, out=None):
        a_, b_, aj, bj, out=_setup_dual_operation(a, b, out=out)
        if out is None:
            out=dlarray(a_ + b_)
        else:
            if not isinstance(out, dlarray):
                out=dlarray(out)
            out[:]=a_ + b_
        for name, jacobian in aj.items():
            out.jacobians[name]=jacobian
        for name, jacobian in bj.items():
            if name in out.jacobians:
                out.jacobians[name] += jacobian
            else:
                out.jacobians[name]=jacobian
        return out

    def subtract(a, b, out=None):
        a_, b_, aj, bj, out=_setup_dual_operation(a, b, out=out)
        if out is None:
            out=dlarray(a_ - b_)
        else:
            if not isinstance(out, dlarray):
                out=dlarray(out)
            out[:]=a_ - b_
        for name, jacobian in aj.items():
            out.jacobians[name]=jacobian
        for name, jacobian in bj.items():
            if name in out.jacobians:
                out.jacobians[name] += -jacobian
            else:
                out.jacobians[name]=-jacobian
        return out

    def multiply(a, b, out=None):
        a_, b_, aj, bj, out=_setup_dual_operation(a, b, out=out)
        # Because out may share memory with a or b, we need to do the
        # Jacobians first as they need access to a and b
        # unadulterated.
        out_jacobians={}
        for name, jacobian in aj.items():
            out_jacobians[name] = jacobian.premul_diag(b_)
        for name, jacobian in bj.items():
            if name in out_jacobians:
                out_jacobians[name] += jacobian.premul_diag(a_)
            else:
                out_jacobians[name] = jacobian.premul_diag(a_)
        if out is None:
            out=dlarray(a_ * b_)
        else:
            if not isinstance(out, dlarray):
                out=dlarray(out)
            out[:]=a_ * b_
        out.jacobians=out_jacobians
        return out

    def true_divide(a, b, out=None):
        a_, b_, aj, bj, out=_setup_dual_operation(a, b, out=out)
        # Note that, the way this is constructed, it's OK if out
        # shares memory with a and/or b, neither a or b are used after
        # out is filled. We do need to keep 1/b though.
        r_=np.reciprocal(b_)
        if out is None:
            out=dlarray(a_ * r_)
        else:
            out[:]=a_ * r_
        out_=units.Quantity(out)
        # We're going to do the quotient rule as (1/b)a' - (a/(b^2))b'
        # The premultiplier for a' is the reciprocal _r, computed above
        # The premultiplier for b' is second is that times the result
        c_=out_*r_
        for name, jacobian in aj.items():
            out.jacobians[name]=jacobian.premul_diag(r_)
        for name, jacobian in bj.items():
            if name in out.jacobians:
                out.jacobians[name] += -jacobian.premul_diag(c_)
            else:
                out.jacobians[name]=-jacobian.premul_diag(c_)
        return out

    def power(a, b):
        # In case it's more efficient this divides up according to
        # whether either a or b or both are duals.  Note that the
        # "both" case has not been tested, so is currently disabled.
        a_, b_, aj, bj, out=_setup_dual_operation(a, b)
        if isinstance(a, dlarray) and isinstance(b, dlarray):
            # This has never been tested so, for now, I'm goint to
            # flag it as not implemented.  However, there is code below,
            # as you can see.
            return NotImplemented
            # a**(b-1)*(b*da/dx+a*log(a)*db/dx)
            # Multiply it out and we get:
            out_=a**b_
            dA_=b_*a_**(b_-1)
            dB_=out_*np.log(a_)
            out=dlarray(out_)
            for name, jacobian in aj.items():
                out.jacobians[name]=jacobian.premul_diag(dA_)
            for name, jacobian in bj.items():
                if name in out.jacobians:
                    out.jacobians[name] += jacobian.premul_diag(dB_)
                else:
                    out.jacobians[name]=jacobian.premul_diag(dB_)
        elif isinstance(a, dlarray):
            out=dlarray(a_**b)
            d_=b*a_**(b-1)
            for name, jacobian in aj.items():
                out.jacobians[name]=jacobian.premul_diag(d_)
        elif isinstance(b, dlarray):
            a_, b_, aj, bj, out=_setup_dual_operation(a, b)
            out_=a**b_
            out=dlarray(out_)
            for name, jacobian in bj.items():
                out.jacobians[name]=jacobian.premul_diag(out_*np.log(a_))
        return out

    def arctan2(a, b):
        # See Wikpedia page on atan2 which conveniently lists the derivatives
        a_, b_, aj, bj, out=_setup_dual_operation(a, b)
        out=dlarray(np.arctan2(a_, b_))
        rr2=units.rad*np.reciprocal(a_**2 + b_**2)
        for name, jacobian in aj.items():
            out.jacobians[name]=jacobian.premul_diag(b_*rr2)
        for name, jacobian in bj.items():
            if name in out.jacobians:
                out.jacobians[name] += jacobian.premul_diag(-a_*rr2)
            else:
                out.jacobians[name]=jacobian.premul_diag(-a_*rr2)
        return out

    # Now some unary operators

    def negative(a):
        a_=units.Quantity(a)
        out=dlarray(-a_)
        for name, jacobian in a.jacobians.items():
            out.jacobians[name]=-jacobian
        return out

    def reciprocal(a):
        a_=units.Quantity(a)
        out_=1.0/a_
        out=dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, -out_**2)
        return out

    def square(a):
        a_=units.Quantity(a)
        out_=np.square(a_)
        out=dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, 2*a_)
        return out

    def sqrt(a):
        a_=units.Quantity(a)
        out_=np.sqrt(a_)
        out=dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, 1.0/(2*out_))
        return out

    def exp(a):
        a_=units.Quantity(a)
        out_=np.exp(a_)
        out=dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, out_)
        return out

    def log(a):
        a_=units.Quantity(a)
        out_=np.log(a_)
        out=dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, 1.0/a_)
        return out

    def log10(a):
        return (1.0/np.log(10.0))*np.log(a)

    # A note on the trigonometric cases, we'll cut out the middle man
    # here and go straight to numpy, forcing the argument into
    # radians.
    def sin(a):
        a_rad=a.to(units.rad)
        out_=np.sin(a_rad.value) << units.dimensionless_unscaled
        out=dlarray(out_)
        if a_rad.hasJ:
            out._chain_rule(a_rad, np.cos(a_rad.value) << _perRad)
        return out

    def cos(a):
        a_rad=a.to(units.rad)
        out_=np.cos(a_rad.value) << units.dimensionless_unscaled
        out=dlarray(out_)
        if a_rad.hasJ:
            out._chain_rule(a_rad, -np.sin(a_rad.value) << _perRad)
        return out

    def tan(a):
        a_rad=a.to(units.rad)
        out_=np.tan(a_rad.value) << units.dimensionless_unscaled
        out=dlarray(out_)
        if a_rad.hasJ:
            out._chain_rule(a_rad, 1.0/(np.cos(a_rad.value)**2) << _perRad)
        return out

    def arcsin(a):
        a_=units.Quantity(a)
        out_=np.arcsin(a_)
        out=dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, units.rad/np.sqrt(1-a_**2))
        return out

    def arccos(a):
        a_=units.Quantity(a)
        out_=np.arccos(a_)
        out=dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, -units.rad/np.sqrt(1-a_**2))
        return out

    def arctan(a):
        a_=units.Quantity(a)
        out_=np.arctan(a_)
        out=dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, units.rad/(1+a_**2))
        return out

    def sinh(a):
        a_rad=a.to(units.rad)
        out_=np.sinh(a_rad.value) << units.dimensionless_unscaled
        out=dlarray(out_)
        if a_rad.hasJ:
            out._chain_rule(a_rad, np.cosh(a_rad.value) << _perRad)
        return out

    def cosh(a):
        a_rad=a.to(units.rad)
        out_=np.cosh(a_rad.value) << units.dimensionless_unscaled
        out=dlarray(out_)
        if a.hasJ:
            out._chain_rule(a_rad, np.sinh(a_rad.value) << _perRad)
        return out

    def tanh(a):
        a_rad=a.to(units.rad)
        out_=np.tanh(a_rad.value) << units.dimensionless_unscaled
        out=dlarray(out_)
        if a.hasJ:
            out._chain_rule(a_rad, 1.0/np.cosh(a_rad.value)**2 << _perRad)
        return out

    def arcsinh(a):
        a_=units.Quantity(a)
        out_=np.arcsinh(a_)
        out=dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, units.rad/np.sqrt(a_**2+1))
        return out

    def arccosh(a):
        a_=units.Quantity(a)
        out_=np.arccosh(a_)
        out=dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, units.rad/np.sqrt(a_**2-1))
        return out

    def arctanh(a):
        a_=units.Quantity(a)
        out_=np.arctanh(a_)
        out=dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, units.rad/(1-a_**2))
        return out

    def absolute(a):
        a_=units.Quantity(a)
        out_=np.absolute(a_)
        out=dlarray(out_)
        if a.hasJ:
            out._chain_rule(a, np.sign(a_))
        return out

    # Note that astropy doesn't supply this routine so, unlike sin,
    # cos, etc. this leapfrongs straight to scipy.special, for now.
    # That may give us problems down the road.
    def wofz(z):
        z_=units.Quantity(z)
        if z_.unit != units.dimensionless_unscaled:
            raise units.UnitsError(
                "Can only apply wofz function to dimensionless quantities")
        out_=special.wofz(z_.value) << units.dimensionless_unscaled
        out=dlarray(out_)
        # The derivative actually comes out of the definition of the
        # Fadeeva function pretty easily
        if z.hasJ():
            c=2*complex(0, 1)/np.sqrt(constants.pi)
            out._chain_rule(z, c-2*z_*out_)
        return out

    def voigt_profile(x, sigma, gamma):
        i=complex(0, 1)
        z=((x+gamma*i)/(sigma*np.sqrt(2))).to(units.dimensionless_unscaled)
        outZ=special.wofz(z)/(sigma*np.sqrt(2*constants.pi))
        # Taking the real part of a quantity appears to be more
        # complex than it should be, in large part because np.real is
        # not a ufunc.
        out=dlarray(np.real(outZ))
        out._set_unit(outZ.unit)
        if outZ.hasJ():
            for name, jacobian in outZ.jacobians.items():
                out.jacobians[name]=dljacobian(np.real(jacobian.data),
                                                 jacobian.dependent_unit,
                                                 jacobian.independent_unit)
        return out

    def maximum(a, b, out=None, **kwargs):
        if out is not None:
            raise NotImplementedError("dlarray.maximum cannot support out")
        if len(kwargs) != 0:
            raise NotImplementedError(
                "dlarray.maximum cannot support non-empty kwargs")
        a_, b_, aj, bj, out=_setup_dual_operation(a, b, out=out)
        out=dlarray(np.maximum(a_, b_))
        if a.hasJ or b.hasJ:
            factor=a_ >= b_
            if hasattr(a, "jacobians"):
                out._chain_rule(a, factor.astype(int))
            if hasattr(b, "jacobians"):
                out._chain_rule(b, np.logical_not(factor).astype(int))
        return out

    def minimum(a, b, out=None, **kwargs):
        if out is not None:
            raise NotImplementedError("dlarray.minimum cannot support out")
        if len(kwargs) != 0:
            raise NotImplementedError(
                "dlarray.minimum cannot support non-empty kwargs")
        a_, b_, aj, bj, out=_setup_dual_operation(a, b, out=out)
        out=dlarray(np.minimum(a_, b_))
        if a.hasJ or b.hasJ:
            factor=a_ <= b_
            out._chain_rule(a, factor.astype(int))
            out._chain_rule(b, np.logical_not(factor).astype(int))
        return out

    def flatten(self, order='C'):
        result = dlarray(units.Quantity(self).flatten(order))
        for name, jacobian in self.jacobians.items():
            result.jacobians[name] = jacobian.flatten(order)
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
            keys=[key
                    for key in self.jacobians.keys()
                    if fnmatch.fnmatch(key, wildcard)]
            # Remove them (done this way because Python complains if
            # iterating over a list of keys that keeps changing as
            # keys are rmoved)
            for key in keys:
                self.jacobians.pop(key)
        # Now, are any jacobians leff ("if" below tests True if so)
        if self.jacobians or remain_dual:
            return self
        else:
            return units.Quantity(self)

    @property
    def uvalue(self):
        return units.Quantity(self)

# -------------------------------------- Now the array functions
HANDLED_FUNCTIONS={}
def implements(numpy_function):
    """Register an __array_function__ implementation for dlarray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function]=func
        return func
    return decorator

@implements(np.sum)
def sum(a, axis=None, dtype=None, keepdims=False):
    a_, aj, out=_setup_dual_operation(a)
    out = dlarray(np.sum(a_, axis=axis, dtype=dtype, keepdims=keepdims))
    for name, jacobian in aj.items():
        out.jacobians[name] = jacobian.sum(
            out.shape, axis=axis, dtype=dtype, keepdims=keepdims)
    return out

@implements(np.cumsum)
def cumsum(a, axis=None, dtype=None, out=None):
    if out is not None:
        raise NotImplementedError("out not supported for dual cumsum (yet?)")
    a_, aj, out=_setup_dual_operation(a)
    out = dlarray(np.cumsum(a_, axis=axis, dtype=dtype))
    for name, jacobian in aj.items():
        out.jacobians[name] = jacobian.cumsum(axis)
    return out

@implements(np.broadcast_arrays)
# One of these days I should look into whether broadcast_arrays and
# broadcast_to really need the subok argument, given that it's ignored.
def broadcast_arrays(*args, subok=False):
    values=[]
    for a in args:
        values.append(a.value)
    result_=np.broadcast_arrays(*values, subok=subok)
    shape=result_[0].shape
    result=[]
    for i, a in enumerate(args):
        thisResult=dlarray(result_[i] << a.unit)
        if hasattr(a, "jacobians"):
            thisResult.jacobians=_broadcast_jacobians(a.jacobians, shape)
        result.append(thisResult)
    return result


@implements(np.broadcast_to)
def broadcast_to(array, shape, subok=False):
    result_=np.broadcast_to(array.value, shape, subok=subok) << array.unit
    result=dlarray(result_)
    result.jacobians=_broadcast_jacobians(array.jacobians, shape)
    return result


@implements(np.reshape)
def reshape(array, *args, **kwargs):
    result=dlarray(units.Quantity(array).reshape(*args, **kwargs))
    for name, jacobian in array.jacobians.items():
        result.jacobians[name] = jacobian.reshape(result.shape)
    return result

@implements(np.atleast_1d)
def atleast_1d(*args):
    result=[]
    for a in args:
        a1d=np.atleast_1d(units.Quantity(a))
        a1d.jacobians=a.jacobians.copy()
        result.append(a1d)
    return tuple(result)

@implements(np.where)
def where(condition, a=None, b=None):

    if a is None or b is None:
        return NotImplemented
    cond_, a_, b_, condj, aj, bj, out=_setup_dual_operation(condition, a, b)
    if condj:
        raise ValueError(
            "Jacobians not allowed on condition argument in 'where'")
    out=dlarray(np.where(condition, a_, b_))
    # Now go through the jacobians and insert them where the condition
    # applies, otherwise they're zero.
    for name, jacobian in aj.items():
        out.jacobians[name]=jacobian.premul_diag(cond_)
    for name, jacobian in bj.items():
        if name in out.jacobians:
            out.jacobians[name] += jacobian.premul_diag(np.logical_not(cond_))
        else:
            out.jacobians[name]=jacobian.premul_diag(np.logical_not(cond_))
    return out

@implements(np.insert)
def insert(arr, obj, values, axis=None):
    # Note that this is supposed to flatten the array first if axis is None
    if axis is None:
        arr = arr.flatten()
    # print (type(arr), arr.jacobians)
    result=dlarray(np.insert(units.Quantity(arr), obj, values, axis))
    # In principal the user could want to insert a dual, that is that
    # the values inserted have derivatives.  However, managing that
    # will be a bit of a pain, so I'll skip it for now, and thus
    # assert that we will not insert any values in the jacobians,
    # because the inserted Jacobian values should be zero.
    if isinstance(values, dlarray):
        raise NotImplementedError("The values inserted cannot be a dual")
    for name, jacobian in arr.jacobians.items():
        result.jacobians[name] = jacobian.insert(obj, axis, result.shape)
    return result

@implements(np.searchsorted)
def searchsorted(a, v, side="left", sorter=None):
    return np.searchsorted(units.Quantity(a), units.Quantity(v), side=side, sorter=sorter)


@implements(np.clip)
def clip(a, a_min, a_max, out=None, **kwargs):
    if type(a_min) is dlarray:
        raise NotImplementedError(
            "dlarray.clip does not (currently) support dual for a_min, use dualarray.minimum")
    if type(a_max) is dlarray:
        raise NotImplementedError(
            "dlarray.clip does not (currently) support dual for a_max, use dualarray.maximum")
    if out is not None:
        raise NotImplementedError("dlarray.clip cannot support out")
    if len(kwargs) != 0:
        raise NotImplementedError(
            "dlarray.clip cannot support non-empty kwargs")
    out=dlarray(np.clip(units.Quantity(a), a_min, a_max))
    if a.hasJ:
        factor=np.logical_and(a >= a_min, a <= a_max)
        out._chain_rule(a, factor.astype(int))
    return out

