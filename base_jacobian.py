"""The base class for the jacobians"""
import copy
import numpy as np

from .jacobian_helpers import _broadcasted_shape, _prepare_jacobians_for_binary_op

__all__ = ["BaseJacobian"]


class BaseJacobian(object):

    """This is a container for a jacobian "matrix".  The various child
    classes store the information as either diagonal, a dense array,
    or a sparse array.

    """

    def __init__(
        self,
        template=None,
        dependent_unit=None,
        independent_unit=None,
        dependent_shape=None,
        independent_shape=None,
        dtype=None,
    ):
        """Define a new jacobian"""

        def pick(*args):
            return next((item for item in args if item is not None), None)

        # Set up the core metadata
        if isinstance(template, BaseJacobian):
            self.dependent_unit = pick(dependent_unit, template.dependent_unit)
            self.independent_unit = pick(independent_unit, template.independent_unit)
            self.dependent_shape = pick(dependent_shape, template.dependent_shape)
            self.independent_shape = pick(independent_shape, template.independent_shape)
        else:
            self.dependent_unit = dependent_unit
            self.independent_unit = independent_unit
            self.dependent_shape = tuple(dependent_shape)
            self.independent_shape = tuple(independent_shape)
        # Do a quick piece of housekeepting
        if type(self.dependent_shape) != tuple:
            self.dependent_shape = tuple(self.dependent_shape)
        if type(self.independent_shape) != tuple:
            self.independent_shape = tuple(self.independent_shape)
        # Now derive a bunch of metadata from that
        self.shape = tuple(self.dependent_shape + self.independent_shape)
        self.dependent_size = int(np.prod(self.dependent_shape))
        self.independent_size = int(np.prod(self.independent_shape))
        self.size = self.dependent_size * self.independent_size
        self.dependent_ndim = len(self.dependent_shape)
        self.independent_ndim = len(self.independent_shape)
        self.ndim = self.dependent_ndim + self.independent_ndim
        self.shape2d = (self.dependent_size, self.independent_size)
        self._dummy_dependent = (1,) * self.dependent_ndim
        self._dummy_independent = (1,) * self.independent_ndim
        self.dtype = dtype

    def __str__(self):
        return (
            f"Jacobian of type {type(self)}\n"
            + f"Dependent shape is {self.dependent_shape} <{self.dependent_size}>\n"
            + f"Independent shape is {self.independent_shape}"
            + f"<{self.independent_size}>\n"
            + f"Combined they are {self.shape} <{self.size}>\n"
            + f"Dummies are {self._dummy_dependent} and {self._dummy_independent}\n"
            + f"Units are d<{self.dependent_unit}>/d<{self.independent_unit}> = "
            + f"{(self.dependent_unit/self.independent_unit).decompose()}"
        )

    def __repr__(self):
        return self.__str__()

    def __neg__(self):
        return type(self)(-self.data, template=self)

    def __add__(self, other):
        from .dense_jacobians import DenseJacobian

        s_, o_, result_type = _prepare_jacobians_for_binary_op(self, other)
        result_ = s_ + o_
        if result_type is DenseJacobian:
            result_ = np.reshape(np.array(result_), self.shape)
        return result_type(data=result_, template=self)

    def __sub__(self, other):
        from .dense_jacobians import DenseJacobian

        s_, o_, result_type = _prepare_jacobians_for_binary_op(self, other)
        result_ = s_ - o_
        if result_type is DenseJacobian:
            result_ = np.reshape(np.array(result_), self.shape)
        return result_type(data=s_ - o_, template=self)

    def __lshift__(self, unit):
        result = copy.copy(self)  # or should this be deepcopy
        result.dependent_unit = unit

    def _get_jaxis(self, axis, none="none"):
        """Correct negative axis arguments so they're valid for jacobians"""
        # Negative axis requests count backwards from the last index,
        # but the Jacobians have the independent_shape appended to
        # their shape, so we need to correct for that (or not if its positive)
        if axis is None:
            if none == "none":
                return None
            if none == "flatten":
                return 0
            if none == "first":
                return 0
            if none == "last":
                return self.dependent_ndim - 1
            if none == "all":
                return tuple(range(self.dependent_ndim))
            if none == "transpose":
                return tuple(range(self.dependent_ndim)[::-1])
            else:
                ValueError(
                    '"none" argument must be one of '
                    + '"none", "first", "last", "all", or "transpose"'
                )
        else:
            try:
                return tuple(a if a >= 0 else a - self.independent_ndim for a in axis)
            except TypeError:
                return axis if axis >= 0 else axis - self.independent_ndim

    def _slice_axis(self, axis, s, none="none"):
        """Return a key that has full slices for all axes, but s for axis"""
        axis = self._get_jaxis(axis, none=none)
        if axis is None:
            raise ValueError("Axis cannot be None in this context")
        return [slice(None)] * axis + [s] + [slice(none)] * (self.ndim - axis - 1)

    def real(self):
        return type(self)(np.real(self.data), template=self)

    def _prepare_premul_diag(self, diag):
        """This routine is called by the child classes to set up for a
        premul_diag.  It works out the units issues and sets up for
        broadcasting.
        """
        if hasattr(diag, "unit"):
            dependent_unit = diag.unit * self.dependent_unit
            diag_ = diag.value
        else:
            dependent_unit = self.dependent_unit
            diag_ = diag
        dependent_shape = _broadcasted_shape(self.dependent_shape, diag_.shape)
        return diag_, dependent_unit, dependent_shape

    def flatten(self, order="C"):
        """flatten a jacobian"""
        return self.reshape((self.dependent_size,), order=order)

    def nan_to_num(self, copy=True, nan=0.0, posinf=None, neginf=None):
        return self.__class__(
            template=self,
            data=np.nan_to_num(
                self.data, copy=copy, nan=nan, posinf=posinf, neginf=neginf
            ),
        )

    def to(self, unit):
        """Change the dependent_unit for a Jacobian"""
        if unit == self.dependent_unit:
            return self
        scale = self.dependent_unit._to(unit) * (unit / self.dependent_unit)
        return self.scalar_multiply(scale)

    def make_dense(self):
        """Return a dense version of self"""
        from .dense_jacobians import DenseJacobian

        return DenseJacobian(self)

    def make_sparse(self):
        """Retrun a sparse version of self"""
        from .sparse_jacobians import SparseJacobian

        return SparseJacobian(self)

    def decompose(self):
        """Decompose the dependent_unit for a Jacobian"""
        raise NotImplementedError("Should not be needed")
        unit = self.dependent_unit.decompose()
        result = self.to(unit)
        return result

    def scalar_multiply(self, scale):
        """Multiply Jacobian by a scalar"""
        self.dependent_unit *= scale.unit
        self.data *= scale.value
        return self

    def independents_compatible(self, other):
        """Return true if the independent variables for two jacobians are compatible"""
        if self.independent_shape != other.independent_shape:
            return False
        if self.independent_unit != other.independent_unit:
            return False
        return True

    def _preprocess_getsetitem_key(self, key):
        """Get it into the right shape for the dependent variable"""
        # Most likely key is a tuple
        if isinstance(key, tuple):
            # If it's too short, then add a ..., unless we already have one, in which
            # case we're fine.
            if len(key) < self.dependent_ndim and Ellipsis not in key:
                key = key + (Ellipsis,)
            # If it's too long, then we have to hope that that is because the user has
            # added some np.newaxis terms.
            elif len(key) > self.dependent_ndim:
                n_extra = len(key) - self.dependent_ndim
                if sum(x is np.newaxis or x is Ellipsis for x in key) != n_extra:
                    raise ValueError(
                        "Dual key for getitem/setitem has extra entries "
                        "that are not np.newaxis (i.e., not None) or Ellipsis"
                    )
            # Otherwise, it's fine as is.
        else:
            if self.dependent_ndim > 1:
                key = (key, Ellipsis)
            else:
                # Don't say "tuple(key)" here, as it will convert a list to a tuple,
                # which is not what we want.
                key = (key,)
        return key

    def matrix_multiply(self, other):
        """Matrix multiply Jacobian with another (other on the right)"""
        from .dense_jacobians import DenseJacobian
        from .sparse_jacobians import SparseJacobian
        from .diagonal_jacobians import DiagonalJacobian
        # Check that the dimensions and units are agreeable
        if self.independent_shape != other.dependent_shape:
            raise ValueError("Shape mismatch for dense Jacobian matrix multiply")
        if self.independent_unit != other.dependent_unit:
            raise ValueError("Units mismatch for dense Jacobian matrix multiply")
        # Recast any diagonal Jacobians into sparse
        if isinstance(self, DiagonalJacobian):
            self = SparseJacobian(self)
        if isinstance(other, DiagonalJacobian):
            other = SparseJacobian(other)
        # Decide what our result type will be
        if isinstance(self, SparseJacobian) and isinstance(other, SparseJacobian):
            result_type = SparseJacobian
        else:
            result_type = DenseJacobian
        # OK, do the matrix multiplication
        result_data2d = self.data2d @ other.data2d
        # Work out its true shape
        result_shape = self.dependent_shape + other.independent_shape
        if result_type is DenseJacobian:
            result_data = result_data2d.reshape(result_shape)
            result_data2d = None
        else:
            result_data = None
        return result_type(
            data=result_data,
            data2d=result_data2d,
            dependent_shape=self.dependent_shape,
            independent_shape=other.independent_shape,
            dependent_unit=self.dependent_unit,
            independent_unit=other.independent_unit,
        )
