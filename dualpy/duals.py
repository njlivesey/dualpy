"""The dual type for dualpy"""

import numpy as np
import fnmatch
import dask

from .unitless import Unitless
from .dual_helpers import (
    DualOperatorsMixin,
    broadcast_jacobians,
    dedual,
    has_jacobians,
    get_magnitude_and_unit,
    get_unit,
    setup_dual_operation,
)
from .jacobians import (
    setitem_jacobians as _setitem_jacobians,
    join_jacobians as _join_jacobians,
    concatenate_jacobians as _concatenate_jacobians,
    stack_jacobians as _stack_jacobians,
)


__all__ = [
    "dlarray",
    "dedual",
    "nan_to_num_jacobians",
    "tensordot",
    "setup_dual_operation",
]


class dlarray(DualOperatorsMixin):
    """A duck-array providing automatic differentiation using dual algebra

    In contrast with the previous version of dualpy this uses a wrapping approach,
    rather than inheritance.  The intent is that it can wrap any other type of
    duck-array, and perhaps that the Jacobians will be of the same type as the duck
    array they describe (at least the dense and diagonal ones?).  The duck array this
    wraps is held in the "variable" attribute, while the Jacobians are in the
    "jacobians" attribute.

    """

    # ---------------------------------------------------------- Initialization
    def __new__(cls, input_variable=None):
        """Instance ecreator for dlarray

        Uses type of variable argument to work out what flavor of dlarray to create.

        """
        # pylint: disable=import-outside-toplevel
        import astropy.units as units
        import pint
        from .dual_astropy import dlarray_astropy
        from .dual_pint import dlarray_pint

        # pylint: enable=import-outside-toplevel

        if isinstance(input_variable, units.Quantity):
            result_class = dlarray_astropy
        elif isinstance(input_variable, pint.Quantity):
            result_class = dlarray_pint
        elif isinstance(input_variable, (np.ndarray, float)):
            result_class = dlarray
        else:
            raise TypeError(
                f"Variables of type {type(input_variable)} cannot be wrapped in duals."
            )
        obj = object.__new__(result_class)
        return obj

    def __init__(self, input_variable):
        """Setup a new dual wrapping around a suitable variable"""
        if isinstance(input_variable, dlarray):
            self.variable = input_variable.variable
            self.jacobians = input_variable.jacobians
        else:
            self.variable = input_variable
            self.jacobians = {}

    def __getnewargs__(self):
        """Needed to correctly pickle/restore duals into right subclass"""
        return (self.variable,)

    # --------------------------------------------------- Fundamental properties
    # Avoid using attributes here, particularly as we might one day support in-place
    # reshaping (x.shape = new_shape).  For now that's disabled as it needs work
    @property
    def shape(self):
        """We want to track cases where shape is changed on the fly"""
        return self.variable.shape

    @shape.setter
    def shape(self, value):
        raise NotImplementedError("For now, in place reshaping is not supported")
        self.variable.shape = value
        for key, jacobian in self.jacobians.items():
            # This will break, should be something other then None in 3rd argument
            self.jacobians[key] = jacobian.reshape(value, "A", None)

    @property
    def ndim(self):
        return self.variable.ndim

    @property
    def size(self):
        return self.variable.size

    @property
    def dtype(self):
        return self.variable.dtype

    @property
    def flags(self):
        return self.variable.flags

    # This attribute is None for plain dlarrays but is a property aliasing to the
    # unit/units attribute of the pint/astropy variable if appropriate.
    _dependent_unit = Unitless()
    # These attributes are used to handle trigonometric cases
    _rad = None
    _per_rad = None
    _dimensionless = None

    # ----------------------------------------------- Helpers (staticmethods)

    @staticmethod
    def _force_unit(quantity, unit):
        """Apply a unit to a quantity"""
        return quantity

    @staticmethod
    def _force_unit_from(quantity, source):
        """Force a unit from source onto quantity"""
        return quantity

    # Find a better place for this one.
    def __lshift__(self, other):
        """Apply units to dual

        This is needed because astropy only recognizes lshift for the case where the LHS
        is an ndarray, or at last that's my conjuecture.

        """
        out = dlarray(self.variable << other)
        for name, jacobian in self.jacobians.items():
            out.jacobians[name] = jacobian._force_unit(other)
        return out

    # ----------------------------------------------------- ufuncs
    # This is based off the comparable one from xarray.DataArray
    def _binary_op(
        self,
        other,
        f,
        reflexive: bool = False,
    ):
        return f(self, other) if not reflexive else f(other, self)

    def _unary_op(self, f):
        return f()

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
            return ufunc(self.variable, args[1])
        # Also, do the same for some unary operators
        if ufunc in (np.isfinite, np.isnan):
            return ufunc(self.variable)
        # Otherwise, we look for this same ufunc in our own type and
        # try to invoke that.
        # However, first some intervention
        dlufunc = getattr(type(self), ufunc.__name__, None)
        if dlufunc is None:
            raise NotImplementedError(
                f"No implementation for ufunc {ufunc}, method {method}"
            )
            return NotImplemented
        result = dlufunc(*args, **kwargs)
        result._check()
        return result

    # __array_ufunc__ = None

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
            qargs = (type(self.variable)(arg) for arg in args)
            return super().__array_function__(
                func, (type(self.variable),), qargs, kwargs
            )
        else:
            return NotImplemented

    def __len__(self):
        return len(self.variable)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        out = dlarray(self.variable)
        for name, jacobian in self.jacobians.items():
            out.jacobians[name] = jacobian
        return out

    def __deepcopy__(self, memo):
        # If we don't define this, ``copy.deepcopy(quantity)`` will
        # return a bare Numpy array.
        result = self.copy()
        return result

    def __array__(self, dtype=None):
        return np.array(self.variable, dtype)

    def __getitem__(self, key):
        out = dlarray(self.variable[key])
        for name, jacobian in self.jacobians.items():
            out.jacobians[name] = jacobian._getjitem(out.shape, key)
        return out

    def __setitem__(self, key, value):
        s, v, sj, vj, out_ = setup_dual_operation(self, value, broadcast=False)
        try:
            self.variable[key] = v
        except TypeError:
            if not key:
                self.variable = v
            else:
                raise TypeError("Dual variable does not support non-empty indexing")
        # Doing a setitem on the Jacobians requires some more intimate knowledge so let
        # the jacobians module handle it.
        _setitem_jacobians(key, s, sj, vj)

    def __eq__(a, b):
        a_, b_, aj, bj, out = setup_dual_operation(a, b)
        return a_ == b_

    def __ne__(a, b):
        a_, b_, aj, bj, out = setup_dual_operation(a, b)
        return a_ != b_

    def __gt__(a, b):
        a_, b_, aj, bj, out = setup_dual_operation(a, b)
        return a_ > b_

    def __lt__(a, b):
        a_, b_, aj, bj, out = setup_dual_operation(a, b)
        return a_ < b_

    def __le__(a, b):
        a_, b_, aj, bj, out = setup_dual_operation(a, b)
        return a_ <= b_

    def _check(self, name="<unknown>"):
        """Check consistency of a dual"""
        for jname, jacobian in self.jacobians.items():
            jacobian._check(
                jname=jname,
                name=name,
                dependent_shape=self.shape,
                dependent_unit=get_unit(self),
            )

    def hasJ(self):
        return len(self.jacobians) != 0

    def _chain_rule(
        self,
        a,
        d,
        dependent_unit=None,
        add=False,
        forced_diagonal_unit=None,
    ):
        """Apply the chain rule to Jacobians to account for a given operation

        Modifies self.jacobians in place, computing:
           self.jacobian[thing] = a.jacobian[thing] * d
        (where * works in a diagonal matrix-multiply sense)

        Updates all the Jacobians in a dlarray by premultiplying them by a diagonal.
        Most of the work is done by the premultiply_diagonal method for the Jacobian itself.
        This method is invoked by almost every single dual method, so needs to aim for
        efficiency.

        Paramters:
        ----------
        a: array-like
           Original input to the operation invoked on the dual
        d: arrray-like
           d(self)/da (expressed as a vector corresponding to the digaonal of the
           (diagonal) matrix of that derivative.
        dependent_unit: astropy.unit / pint.unit optional
           If supplied convert Jacobian to this dependent unit before multiplying
        add: bool (default false)
           If set, do not overwrite existing jacobians, rather add these terms to them.
        forced_diagonal_unit: astropy.unit / pint unit, optional
           If set, force the diagonal to this unit before applying
        """
        d = self._force_unit(d, forced_diagonal_unit)
        if dependent_unit is not None:
            for name, jacobian in a.jacobians.items():
                if add and name in self.jacobians:
                    self.jacobians[name] += jacobian.to(
                        dependent_unit
                    ).premultiply_diagonal(d)
                else:
                    self.jacobians[name] = jacobian.to(
                        dependent_unit
                    ).premultiply_diagonal(d)
        else:
            for name, jacobian in a.jacobians.items():
                if add and name in self.jacobians:
                    self.jacobians[name] += jacobian.premultiply_diagonal(d)
                else:
                    self.jacobians[name] = jacobian.premultiply_diagonal(d)

    def ravel(self, order="C"):
        return _ravel(self, order=order)

    # Now a whole bunch of binary operators
    @staticmethod
    def add(a, b, out=None):
        a_, b_, aj, bj, out = setup_dual_operation(a, b, out=out)
        if out is None:
            out = dlarray(a_ + b_)
        else:
            if not isinstance(out, dlarray):
                out = dlarray(out)
            out.variable[...] = a_ + b_
        for name, jacobian in aj.items():
            out.jacobians[name] = jacobian
        for name, jacobian in bj.items():
            if name in out.jacobians:
                out.jacobians[name] += jacobian
            else:
                out.jacobians[name] = jacobian.to(out._dependent_unit)
        return out

    @staticmethod
    def subtract(a, b, out=None):
        a_, b_, aj, bj, out = setup_dual_operation(a, b, out=out)
        if out is None:
            out = dlarray(a_ - b_)
        else:
            if not isinstance(out, dlarray):
                out = dlarray(out)
            out.variable[...] = a_ - b_
        for name, jacobian in aj.items():
            out.jacobians[name] = jacobian
        for name, jacobian in bj.items():
            if name in out.jacobians:
                out.jacobians[name] += -jacobian
            else:
                out.jacobians[name] = -jacobian.to(out._dependent_unit)
        return out

    @staticmethod
    def multiply(a, b, out=None):
        if out is not None:
            raise NotImplementedError("Unable to handle cases with out supplied")
        a_, b_, aj, bj, out = setup_dual_operation(a, b, out=out)
        # Because out may share memory with a or b, we need to do the
        # Jacobians first as they need access to a and b
        # unadulterated.
        # out_jacobians = {}
        # for name, jacobian in aj.items():
        #     out_jacobians[name] = jacobian.premultiply_diagonal(b_)
        # for name, jacobian in bj.items():
        #     if name in out_jacobians:
        #         out_jacobians[name] += jacobian.premultiply_diagonal(a_)
        #     else:
        #         out_jacobians[name] = jacobian.premultiply_diagonal(a_)
        # if out is None:
        #     out = dlarray(a_ * b_)
        # else:
        #     if not isinstance(out, dlarray):
        #         out = dlarray(out)
        #     out.variable[...] = a_ * b_
        # out.jacobians = out_jacobians

        out = dlarray(a_ * b_)
        for name, jacobian in aj.items():
            out.jacobians[name] = jacobian.premultiply_diagonal(b_)
        for name, jacobian in bj.items():
            if name in out.jacobians:
                out.jacobians[name] += jacobian.premultiply_diagonal(a_)
            else:
                out.jacobians[name] = jacobian.premultiply_diagonal(a_)
        return out

    @staticmethod
    def true_divide(a, b, out=None):
        if out is not None:
            raise NotImplementedError("Unable to handle cases with out supplied")
        a_, b_, aj, bj, out = setup_dual_operation(a, b, out=out)
        r_ = 1.0 / b_
        out_ = a_ * r_
        out = dlarray(out_)
        # We're going to do the quotient rule as (1/b)a' - (a/(b^2))b'
        # The premultiplier for a' is the reciprocal r_, computed above
        # The premultiplier for b' is that times the result
        c_ = out_ * r_
        for name, jacobian in aj.items():
            out.jacobians[name] = jacobian.premultiply_diagonal(r_)
        for name, jacobian in bj.items():
            if name in out.jacobians:
                out.jacobians[name] += -jacobian.premultiply_diagonal(c_)
            else:
                out.jacobians[name] = -jacobian.premultiply_diagonal(c_).to(
                    out._dependent_unit
                )
        return out

    @staticmethod
    def remainder(a, b, out=None):
        raise NotImplementedError("Pretty sure this is wrong")
        a_, b_, aj, bj, out = setup_dual_operation(a, b, out=out)
        out = dlarray(a_ % b_)
        # The resulting Jacobian is simply a copy of the jacobian for a, b has no impact
        for name, jacobian in aj.items():
            out.jacobians[name] = jacobian
        return out

    @staticmethod
    def power(a, b):
        # In case it's more efficient this divides up according to
        # whether either a or b or both are duals.  Note that the
        # "both" case has not been tested, so is currently disabled.
        a_, b_, aj, bj, out = setup_dual_operation(a, b)
        if isinstance(a, dlarray) and isinstance(b, dlarray):
            # This has never been tested so, for now, I'm goint to flag it as not
            # implemented.  However, there is code below, as you can see.  In
            # particular, please pay special attention to the handling of units.
            return NotImplemented
            # a**(b-1)*(b*da/dx+a*log(a)*db/dx)
            # Multiply it out and we get:
            out_ = a**b_
            dA_ = b_ * a_ ** (b_ - 1)
            dB_ = out_ * np.log(a_)
            out = dlarray(out_)
            for name, jacobian in aj.items():
                out.jacobians[name] = jacobian.premultiply_diagonal(dA_)
            for name, jacobian in bj.items():
                if name in out.jacobians:
                    out.jacobians[name] += jacobian.premultiply_diagonal(dB_)
                else:
                    out.jacobians[name] = jacobian.premultiply_diagonal(dB_).to(
                        out._dependent_unit
                    )
        elif isinstance(a, dlarray):
            out = dlarray(a_**b)
            d_ = b * a_ ** (b - 1)
            for name, jacobian in aj.items():
                out.jacobians[name] = jacobian.premultiply_diagonal(d_)
        elif isinstance(b, dlarray):
            a_, b_, aj, bj, out = setup_dual_operation(a, b)
            out_ = a**b_
            out = dlarray(out_)
            for name, jacobian in bj.items():
                out.jacobians[name] = jacobian.premultiply_diagonal(out_ * np.log(a_))
            if hasattr(b, "units") or hasattr(b, "unit"):
                out = out * b._dimensionless
        return out

    @staticmethod
    def arctan2(a, b):
        # See Wikpedia page on atan2 which conveniently lists the derivatives
        a_, b_, aj, bj, out = setup_dual_operation(a, b)
        out = dlarray(np.arctan2(a_, b_))
        rr2 = a._rad * np.reciprocal(a_**2 + b_**2)
        for name, jacobian in aj.items():
            out.jacobians[name] = jacobian.premultiply_diagonal(b_ * rr2).to(
                out._dependent_unit
            )
        for name, jacobian in bj.items():
            if name in out.jacobians:
                out.jacobians[name] += jacobian.premultiply_diagonal(-a_ * rr2).to(
                    out._dependent_unit
                )
            else:
                out.jacobians[name] = jacobian.premultiply_diagonal(-a_ * rr2).to(
                    out._dependent_unit
                )
        return out

    # Now some unary operators
    def negative(self):
        out = dlarray(-self.variable)
        for name, jacobian in self.jacobians.items():
            out.jacobians[name] = -jacobian
        return out

    def positive(self):
        out = dlarray(+self.variable)
        for name, jacobian in self.jacobians.items():
            out.jacobians[name] = +jacobian
        return out

    def reciprocal(self):
        out = dlarray(1.0 / self.variable)
        if self.hasJ:
            out._chain_rule(self, -(out.variable**2))
        return out

    def square(self):
        out = dlarray(np.square(self.variable))
        if self.hasJ:
            out._chain_rule(self, 2 * self.variable)
        return out

    def sqrt(self):
        out = dlarray(np.sqrt(self.variable))
        if self.hasJ:
            out._chain_rule(self, 1.0 / (2 * out.variable))
        return out

    def exp(self):
        out_ = np.exp(self.variable)
        out = dlarray(out_)
        if self.hasJ:
            out._chain_rule(self, out_, dependent_unit=self._dimensionless)
        return out

    def log(self):
        out = dlarray(np.log(self.variable))
        if self.hasJ:
            out._chain_rule(
                self, 1.0 / self.variable, dependent_unit=self._dimensionless
            )
        return out

    def log10(self):
        return (1.0 / np.log(10.0)) * np.log(self)

    def transpose(self, axes=None):
        out = dlarray(self.variable.transpose(axes))
        for name, jacobian in self.jacobians.items():
            out.jacobians[name] = jacobian.transpose(axes, out.shape)
        return out

    def matmul(self, other):
        raise NotImplementedError(
            "No implementation of matmul for duals, consider tensordot?"
        )

    # def rmatmul(self):
    #     raise NotImplementedError(
    #         "No implementation of rmatmul for duals, consider tensordot?"
    #     )

    @property
    def T(self):
        return self.transpose()

    # A note on the trigonometric cases.  Here we need to force to radiances to
    # make sure our jacobians end up correct.  Use a built in method that is
    # simply a passthrough for the numpy array case, and more intelligent for
    # astropy.units and pint.
    def _to_radians(self):
        return self

    def sin(self):
        self_rad = self._to_radians()
        out_ = np.sin(self_rad.variable)
        out = dlarray(out_)
        if self_rad.hasJ:
            out._chain_rule(
                self_rad,
                np.cos(self_rad.variable),
                dependent_unit=self._rad,
                forced_diagonal_unit=self._per_rad,
            )
        return out

    def cos(self):
        self_rad = self._to_radians()
        out_ = np.cos(self_rad.variable)
        out = dlarray(out_)
        if self_rad.hasJ:
            out._chain_rule(
                self_rad,
                -np.sin(self_rad.variable),
                dependent_unit=self._rad,
                forced_diagonal_unit=self._per_rad,
            )
        return out

    def tan(self):
        self_rad = self._to_radians()
        out_ = np.tan(self_rad.variable)
        out = dlarray(out_)
        if self_rad.hasJ:
            out._chain_rule(
                self_rad,
                1.0 / (np.cos(self_rad.value) ** 2),
                dependent_unit=self._rad,
                forced_diagonal_unit=self._per_rad,
            )
        return out

    def arcsin(self):
        out = dlarray(np.arcsin(self.variable))
        if self.hasJ:
            out._chain_rule(
                self,
                1.0 / np.sqrt(1 - self.variable**2),
                dependent_unit=self._dimensionless,
                forced_diagonal_unit=self._rad,
            )
        return out

    def arccos(self):
        out = dlarray(np.arccos(self.variable))
        if self.hasJ:
            out._chain_rule(
                self,
                -1.0 / np.sqrt(1 - self.variable**2),
                dependent_unit=self._dimensionless,
                forced_diagonal_unit=self._rad,
            )
        return out

    def arctan(self):
        out = dlarray(np.arctan(self.variable))
        if self.hasJ:
            out._chain_rule(
                self,
                1.0 / (1 + self.variable**2),
                dependent_unit=self._dimensionless,
                forced_diagonal_unit=self._rad,
            )
        return out

    def sinh(self):
        out = dlarray(np.sinh(self.variable))
        if self.hasJ:
            out._chain_rule(
                self,
                np.cosh(self.variable),
                dependent_unit=self._rad,
                forced_diagonal_unit=self._per_rad,
            )
        return out

    def cosh(self):
        out = dlarray(np.cosh(self.variable))
        if self.hasJ:
            out._chain_rule(
                self,
                np.sinh(self.variable),
                dependent_unit=self._rad,
                forced_diagonal_unit=self._per_rad,
            )
        return out

    def tanh(self):
        out = dlarray(np.tanh(self.variable))
        if self.hasJ:
            out._chain_rule(
                self,
                1.0 / np.cosh(self.variable) ** 2,
                dependent_unit=self._rad,
                forced_diagonal_unit=self._per_rad,
            )
        return out

    def arcsinh(self):
        out = dlarray(np.arcsinh(self.variable))
        if self.hasJ:
            out._chain_rule(
                self,
                1.0 / np.sqrt(self.variable**2 + 1),
                dependent_unit=self._dimensionless,
                forced_diagonal_unit=self._rad,
            )
        return out

    def arccosh(self):
        out = dlarray(np.arcosh(self.variable))
        if self.hasJ:
            out._chain_rule(
                self,
                1.0 / np.sqrt(self.variable**2 - 1),
                dependent_unit=self._dimensionless,
                forced_diagonal_unit=self._rad,
            )
        return out

    def arctanh(self):
        out = dlarray(np.arctanh(self.variable))
        if self.hasJ:
            out._chain_rule(
                self,
                1.0 / (1 - self.variable**2),
                dependent_unit=self._dimensionless,
                forced_diagonal_unit=self._rad,
            )
        return out

    def absolute(self):
        out = dlarray(np.absolute(self.variable))
        if self.hasJ:
            out._chain_rule(self, np.sign(self.variable))
        return out

    def abs(self):
        return np.absolute(self)

    def maximum(a, b, out=None, **kwargs):
        if out is not None:
            raise NotImplementedError("dlarray.maximum cannot support out")
        if len(kwargs) != 0:
            raise NotImplementedError("dlarray.maximum cannot support non-empty kwargs")
        a_, b_, aj, bj, out = setup_dual_operation(a, b, out=out)
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
        a_, b_, aj, bj, out = setup_dual_operation(a, b, out=out)
        out = dlarray(np.minimum(a_, b_))
        if a.hasJ or b.hasJ:
            factor = a_ <= b_
            if hasattr(a, "jacobians"):
                out._chain_rule(a, factor.astype(int))
            if hasattr(b, "jacobians"):
                out._chain_rule(b, np.logical_not(factor).astype(int))
        return out

    def floor(self):
        # For now, when we take the floor, let's assume no Jacobians survive
        return np.floor(self.value)

    def flatten(self, order="C"):
        result = dlarray(self.variable.flatten(order))
        for name, jacobian in self.jacobians.items():
            result.jacobians[name] = jacobian.flatten(order, self.flags)
        return result

    def squeeze(self, axis=None):
        """Remove axis of length 1 from self"""
        result = dlarray(self.variable.squeeze(axis))
        for name, jacobian in self.jacobians.items():
            result.jacobians[name] = jacobian.reshape(
                result.shape, order="A", parent_flags=self.flags
            )
        return result

    def delete_jacobians(self, *names, wildcard=None):
        """Removes Jacobians from a dlarray

        Arguments
        ---------
        names : sequence[str]
            Sequence of named Jacobians to delete.  If absent (and no wildcard is
            suppled) then all the Jacobians are deleted.
        wildcard : str, optional
            A unix-style wildcard identifying Jacobians to deleted.
        """
        if names and (wildcard is not None):
            raise ValueError("Cannote supply both named Jacobians and a wildcard")
        if wildcard is not None:
            # If a wildcard is supplied, then identify the Jacobians to delete
            names = [
                key for key in self.jacobians.keys() if fnmatch.fnmatch(key, wildcard)
            ]
        elif not names:
            # If no names and no wildcard suppled, then we're deleting all the Jacobians
            names = list(self.jacobians.keys())
        # Now delete those Jaobians we want to delete
        for key in names:
            del self.jacobians[key]

    def reshape(array, *newshape, order="C"):
        try:
            if len(newshape) == 1:
                newshape = newshape[0]
        except TypeError:
            pass
        return _reshape(array, newshape, order)

    def __str__(self):
        return str(self.variable)

    def __repr__(self):
        return "dlarray-wrapped-" + repr(self.variable)


# ------------------------------------------------------
# Now some helper routines


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
    a_, aj, out = setup_dual_operation(a)
    out = dlarray(np.sum(a_, axis=axis, dtype=dtype, keepdims=keepdims))
    for name, jacobian in aj.items():
        out.jacobians[name] = jacobian.sum(
            out.shape, axis=axis, dtype=dtype, keepdims=keepdims
        )
    return out


@implements(np.mean)
def mean(a, axis=None, dtype=None, keepdims=False):
    a_, aj, out = setup_dual_operation(a)
    out = dlarray(np.mean(a_, axis=axis, dtype=dtype, keepdims=keepdims))
    for name, jacobian in aj.items():
        out.jacobians[name] = jacobian.mean(
            out.shape, axis=axis, dtype=dtype, keepdims=keepdims
        )
    return out


@implements(np.cumsum)
def cumsum(a, axis=None, dtype=None, out=None):
    if out is not None:
        raise NotImplementedError("out not supported for dual cumsum (yet?)")
    a_, aj, out = setup_dual_operation(a)
    out = dlarray(np.cumsum(a_, axis=axis, dtype=dtype))
    for name, jacobian in aj.items():
        out.jacobians[name] = jacobian.cumsum(axis)
    return out


# One of these days I should look into whether broadcast_arrays and
# broadcast_to really need the subok argument, given that it's ignored.
@implements(np.broadcast_arrays)
def broadcast_arrays(*args, subok=False):
    from mls_scf_tools.mls_pint import mp_broadcast_arrays

    variables = [dedual(a) for a in args]
    # This is going to give problems with pint, which does not implement it.
    result_ = mp_broadcast_arrays(*variables, subok=subok)
    shape = result_[0].shape
    result = []
    for i, a in enumerate(args):
        thisResult = dlarray(result_[i])
        if hasattr(a, "jacobians"):
            thisResult.jacobians = broadcast_jacobians(a.jacobians, shape)
        result.append(thisResult)
    return tuple(result)


@implements(np.broadcast_to)
def broadcast_to(array, shape, subok=False):
    result_ = np.broadcast_to(array.variable, shape, subok=subok)
    result = dlarray(result_)
    result.jacobians = broadcast_jacobians(array.jacobians, shape)
    return result


def _reshape(array, newshape, order="C"):
    array_, jacobians, out = setup_dual_operation(array)
    out = dlarray(np.reshape(array_, newshape, order=order))
    for name, jacobian in jacobians.items():
        out.jacobians[name] = jacobian.reshape(newshape, order, array_.flags)
    return out


@implements(np.reshape)
def reshape(array, newshape, order="C"):
    return _reshape(array, newshape, order=order)


def _ravel(array, order="C"):
    array_, jacobians, out = setup_dual_operation(array)
    out = dlarray(np.ravel(array_, order=order))
    for name, jacobian in jacobians.items():
        out.jacobians[name] = jacobian.ravel(order, array_.flags)
    return out


@implements(np.ravel)
def ravel(array, order="C"):
    return _ravel(array, order=order)


@implements(np.atleast_1d)
def atleast_1d(*args):
    result = []
    for a in args:
        a1d = np.atleast_1d(a.variable)
        a1d.jacobians = a.jacobians.copy()
        result.append(a1d)
    return tuple(result)


@implements(np.diff)
def diff(array, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    result_ = np.diff(array.variable, n, axis, prepend, append)
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
    cond_, a_, b_, condj, aj, bj, out = setup_dual_operation(condition, a, b)
    if condj:
        raise ValueError("Jacobians not allowed on condition argument in 'where'")
    out = dlarray(np.where(cond_, a_, b_))
    # Now go through the jacobians and insert them where the condition
    # applies, otherwise they're zero.
    for name, jacobian in aj.items():
        out.jacobians[name] = jacobian.premultiply_diagonal(cond_)
    for name, jacobian in bj.items():
        if name in out.jacobians:
            out.jacobians[name] += jacobian.premultiply_diagonal(np.logical_not(cond_))
        else:
            out.jacobians[name] = jacobian.premultiply_diagonal(np.logical_not(cond_))
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
    arr_, values_, aj, vj, out = setup_dual_operation(arr, values, broadcast=False)
    result = dlarray(np.insert(arr_, obj, values_, axis))
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
    arr_, values_, aj, vj, out = setup_dual_operation(arr, values, broadcast=False)
    result = dlarray(np.append(arr_, values_, axis))
    result.jacobians = _join_jacobians(arr, values, arr.shape[axis], axis, result.shape)
    return result


@implements(np.searchsorted)
def searchsorted(a, v, side="left", sorter=None):
    return np.searchsorted(dedual(a), dedual(v), side=side, sorter=sorter)


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
    out = dlarray(np.clip(a.variable, a_min, a_max))
    if a.hasJ:
        factor = np.logical_and(a >= a_min, a <= a_max)
        out._chain_rule(a, factor.astype(int))
    return out


@implements(np.argmin)
def argmin(a, axis=None, out=None, *, keepdims=np._NoValue):
    """Implements np.argmin"""
    return np.argmin(a.variable, axis=axis, out=out, keepdims=keepdims)


@implements(np.argmax)
def argmax(a, axis=None, out=None, *, keepdims=np._NoValue):
    """Implements np.argmax"""
    return np.argmax(a.variable, axis=axis, out=out, keepdims=keepdims)


@implements(np.amin)
def amin(a, axis=None, out=None, keepdims=None, initial=None):
    """Implements numpy amin (in limited form for now)"""
    if out is not None:
        raise NotImplementedError("dual amin does not support out (yet)")
    if keepdims is not None:
        raise NotImplementedError("dual amin does not support keepdims (yet)")
    if a.ndim == 0:
        return a
    else:
        i_min = np.argmin(a.variable, axis=axis)
        if axis is None:
            return a.ravel()[i_min]
        else:
            raise NotImplementedError("Not yet written this")


@implements(np.amax)
def amax(a, axis=None, out=None, keepdims=None, initial=None):
    """Implements numpy amax (in limited form for now)"""
    if out is not None:
        raise NotImplementedError("dual amax does not support out (yet)")
    if keepdims is not None:
        raise NotImplementedError("dual amax does not support keepdims (yet)")
    if a.ndim == 0:
        return a
    else:
        i_max = np.argmax(a.variable, axis=axis)
        if axis is None:
            return a.ravel()[i_max]
        else:
            raise NotImplementedError("Not yet written this")


@implements(np.nan_to_num)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None, jacobians_only=False):
    x_, j, out = setup_dual_operation(x)
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
    out = dlarray(np.real(a.variable))
    for name, jacobian in a.jacobians.items():
        out.jacobians[name] = jacobian.real()
    return out


@implements(np.empty_like)
def empty_like(prototype, dtype=None, order="K", subok=True, shape=None):
    return dlarray(np.empty_like(prototype.variable, dtype, order, subok, shape))


@implements(np.zeros_like)
def zeros_like(prototype, dtype=None, order="K", subok=True, shape=None):
    return dlarray(np.zeros_like(prototype.variable, dtype, order, subok, shape))


@implements(np.ones_like)
def ones_like(prototype, dtype=None, order="K", subok=True, shape=None):
    return dlarray(np.ones_like(prototype.variable, dtype, order, subok, shape))


@implements(np.expand_dims)
def expand_dims(a, axis):
    result = dlarray(np.expand_dims(a.variable, axis))
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
    values_ = [dedual(value) for value in values]
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
    arrays_ = [dedual(array) for array in arrays]
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

    a_, b_, aj, bj, out = setup_dual_operation(a, b, out=None, broadcast=False)
    a_magnitude, a_unit = get_magnitude_and_unit(a_)
    b_magnitude, b_unit = get_magnitude_and_unit(b_)
    result_ = st.tensordot(a_magnitude, b_magnitude, axes) * a_unit * b_unit
    result_ = result_.astype(np.result_type(a_magnitude.dtype, b_magnitude.dtype))
    result = dlarray(result_)

    result_unit = get_unit(result)
    # Now deal with the Jacobians.  For this, we need to ensure that axes are in the
    # (2,) array-like form that is the second version np.tensordot can accept them.
    if isinstance(axes, int):
        axes = [list(range(a.ndim - axes, a.ndim)), list(range(axes))]
    # Go in and set all the negative axes to the correct positive values (easiest to do
    # it here rathe than in each of the jacobian routines).
    cleaned_axes = []
    for original_axes, ndim in zip(axes, [a.ndim, b.ndim]):
        cleaned_axes.append(
            [axis if axis >= 0 else ndim - axis for axis in original_axes]
        )
    axes = cleaned_axes
    # Get the jacobian tensordot routines
    for name, jacobian in aj.items():
        result.jacobians[name] = jacobian.rtensordot(
            b_magnitude, axes, dependent_unit=result_unit
        )
    for name, jacobian in bj.items():
        if name in result.jacobians:
            result.jacobains[name] += jacobian.rtensordot(
                a_magnitude,
                axes,
                dependent_unit=result_unit,
            )
        else:
            result.jacobians[name] = jacobian.rtensordot(
                a_magnitude,
                axes,
                dependent_unit=result_unit,
            )
    # If there are no Jacobians in the result, then demote to a non dualed result.
    if not has_jacobians(result):
        result = dedual(result)
    return result


def nan_to_num_jacobians(x, copy=True, nan=0.0, posinf=None, neginf=None):
    return nan_to_num(
        x, copy=copy, nan=nan, posinf=posinf, neginf=neginf, jacobians_only=True
    )
