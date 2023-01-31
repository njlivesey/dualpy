"""The class for diagonal jacobians"""

import numpy as np
from typing import Union

from .dual_helpers import apply_units
from .base_jacobian import BaseJacobian
from .sparse_jacobians import SparseJacobian
from .dense_jacobians import DenseJacobian


__all__ = ["DiagonalJacobian", "SeedJacobian"]


# This function takes a DiagonalJacobian that is about to become non-diagonal and
# changes it to another type.  Typically this is a SparseJacobian, but if the
# independent size is small, then DenseJacobian is used instead.
def to_non_diagonal(
    jacobian: type["DiagonalJacobian"],
) -> Union[DenseJacobian, SparseJacobian]:
    if jacobian.independent_size <= 3:
        return DenseJacobian(jacobian)
    else:
        return SparseJacobian(jacobian)


class DiagonalJacobian(BaseJacobian):
    """dljacobian that's really a diagonal"""

    def __init__(self, data, template=None, **kwargs):
        if isinstance(data, BaseJacobian):
            if template is None:
                template = data
            else:
                raise ValueError(
                    "Cannot supply template with jacobian for data simultaneously"
                )
        super().__init__(template=template, **kwargs)
        if self.dependent_shape != self.independent_shape:
            raise ValueError("Attempt to create a diagonal Jacobian that is not square")
        if isinstance(data, BaseJacobian):
            if isinstance(data, DiagonalJacobian):
                data_ = data.data
            else:
                raise ValueError(
                    "Can only create diagonal Jacobians from other diagonals"
                )
        elif data is None:
            data_ = np.zeros(shape=self.shape, dtype=self.dtype)
        else:
            data_ = data
        if data_.shape != self.dependent_shape:
            raise ValueError(
                "Attempt to create a diagonal Jacobian using wrong-shaped input"
            )
        self.data = data_
        self.dtype = self.data.dtype
        self._check()

    def __str__(self):
        return super().__str__() + f"\ndata is {self.data.shape}"

    def _check(self, name=None):
        """Integrity checks on diagonal Jacobian"""
        if name is None:
            name = "<unknown diagonal Jacobian>"
        self._check_jacobian_fundamentals(name)
        assert self.dependent_shape == self.independent_shape, (
            f"Non-diagonal shape for diagonal Jacobian {name}"
            f"{self.dependent_shape} != {self.independent_shape}"
        )
        assert self.dependent_size == self.independent_size, (
            f"Non-diagonal size for diagonal Jacobian {name}"
            f"{self.dependent_size} != {self.independent_size}"
        )
        assert self.data.shape == self.dependent_shape, (
            f"Array shape mismatch for {name}, "
            f"{self.data.shape} != {self.dependent_shape}"
        )

    def _getjitem(self, new_shape, key):
        """A getitem type method for diagonal Jacobians"""
        # OK, once we extract items, this will no longer be diagonal, so we convert to
        # sparse/dense before doing the subset.
        self = to_non_diagonal(self)
        result = self._getjitem(new_shape, key)
        return result

    def _setjitem(self, key, value):
        """A setitem type method for diagonal Jacobians"""
        # OK, once we insert items, this will no longer be diagonal, so we convert to
        # sparse/dense before doing the subset
        self = to_non_diagonal(self)
        self._setjitem(key, value)

    def broadcast_to(self, shape):
        """Broadcast diagonal jacobian to new dependent shape"""
        # OK, once you broadcast a diagonal, it is not longer, strictly speaking, a
        # diagonal So, convert to sparse/dense and broadcast that.  However, don't
        # bother doing anything if there is no actual broadcast going on.
        if shape == self.dependent_shape:
            return self
        self = to_non_diagonal(self)
        return self.broadcast_to(shape)

    def reshape(self, shape, order, parent_flags):
        # OK, once you reshape a diagonal, it is not longer, strictly speaking, a
        # diagonal So, convert to sparse/dense and reshape that.  However, don't bother
        # doing anything if there is no actual reshape going on.
        if shape == self.dependent_shape:
            return self
        self = to_non_diagonal(self)
        return self.reshape(shape, order, parent_flags)

    def premul_diag(self, diag):
        """Diagonal premulitply for diagonal Jacobian"""
        diag_, dependent_unit, dependent_shape = self._prepare_premul_diag(diag)
        # Do something special if we're just multiplying by a unit
        if diag_ is None:
            return DiagonalJacobian(self, dependent_unit=dependent_unit)
        if dependent_shape == self.independent_shape:
            return DiagonalJacobian(
                diag_ * self.data, template=self, dependent_unit=dependent_unit
            )
        else:
            return to_non_diagonal(self).premul_diag(diag)

    def insert(self, obj, axis, dependent_shape):
        """insert method for diagonal Jacobian"""
        # By construction this is no longer diagonal once inserted to
        # change to sparse/dense and insert there.
        self = to_non_diagonal(self)
        return self.insert(obj, axis, dependent_shape)

    def diff(
        self, dependent_shape, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue
    ):
        """diff method for diagonal jacobian"""
        # Again, the result will not be diagonal, so change to sparse/dense and do diff
        # in that space.
        self = to_non_diagonal(self)
        return self.diff(dependent_shape, n, axis, prepend, append)

    def transpose(self, axes, result_dependent_shape):
        return DiagonalJacobian(
            data=self.data.transpose(axes),
            template=self,
            dependent_shape=result_dependent_shape,
        )

    def tensordot(self, other, axes, dependent_unit, reverse_order=False):
        """Compute self(.)other, or other(.)self if reverse_order is True"""
        # Once we do this, we will no longer be diagonal, so convert to sparse/dense
        self = to_non_diagonal(self)
        return self.tensordot(other, axes, dependent_unit, reverse_order)

    def sum(self, dependent_shape, axis=None, dtype=None, keepdims=False):
        """Performs sum for the diagonal Jacobians"""
        # Once we take the sum, along any or all axes, the jacobian is no longer
        # diagonal by construction, so it needs to be converted to sparse/dense.
        self = to_non_diagonal(self)
        return self.sum(dependent_shape, axis=axis, dtype=dtype, keepdims=keepdims)

    def mean(self, dependent_shape, axis=None, dtype=None, keepdims=False):
        """Performs mean for the diagonal Jacobians"""
        # Once we take the sum, along any or all axes, the jacobian is no longer
        # diagonal by construction, so it needs to be converted to sparse/dense.
        self = to_non_diagonal(self)
        return self.mean(dependent_shape, axis=axis, dtype=dtype, keepdims=keepdims)

    def cumsum(self, axis):
        """Perform cumsum for the diagonal Jacobians"""
        # Once it's been cumsummed then it's no longer diagonal.  For that matter it's
        # not going to be particularly sparse either, so we might as well convert it to
        # dense.
        self = to_non_diagonal(self)
        return self.cumsum(axis)

    def diagonal(self):
        """Get diagonal elements (shape=dependent_shape)"""
        return apply_units(self.data, (self.dependent_unit / self.independent_unit))

    # The reaons we have extract_diagonal and diagonal is that diagonal is only
    # populated for diagonal Jacobians.  extract_diagonal is populated for all.
    def extract_diagonal(self):
        """Extract the diagonal from a diagonal Jacobian"""
        return self.diagonal()

    def todensearray(self):
        from .dense_jacobians import DenseJacobian

        selfdense = DenseJacobian(self)
        return selfdense.todensearray()

    def to2darray(self):
        self = to_non_diagonal(self)
        return self.to2darray()

    def to2ddensearray(self):
        selfdense = DenseJacobian(self)
        return selfdense.to2ddensearray()

    def linear_interpolator(self, x_in, axis=-1, extrapolate=None):
        """Return an interpolator for a given Jacobian axis"""
        return to_non_diagonal(self).linear_interpolator(
            x_in=x_in,
            axis=axis,
            extrapolate=extrapolate,
        )

    def spline_interpolator(
        self, x_in, axis=-1, bc_type="not_a_knot", extrapolate=None
    ):
        """Return an interpolator for a given Jacobian axis"""
        return DenseJacobian(self).spline_interpolator(
            x_in=x_in,
            axis=axis,
            bc_type=bc_type,
            extrapolate=extrapolate,
        )


class SeedJacobian(DiagonalJacobian):
    """A specific subclass of DiagonalJacobian for seeds

    There is no substantive difference.  The only intent is to indicate that this is an
    actual seed (it's only use thus far is to help the numeric Jacobian computation code
    identify what are the seeded variables so it can perturb them.

    """

    pass
