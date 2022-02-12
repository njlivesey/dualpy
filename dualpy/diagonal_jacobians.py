"""The class for diagonal jacobians"""

import numpy as np

from .base_jacobian import BaseJacobian
from .sparse_jacobians import SparseJacobian
from .dense_jacobians import DenseJacobian


__all__ = ["DiagonalJacobian", "SeedJacobian"]


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

    def __str__(self):
        return super().__str__() + f"\ndata is {self.data.shape}"

    def _check(self, name):
        """Integrity checks on diagonal Jacobian"""
        self._check_jacobian_fundamentals(name)
        assert self.depdent_shape == self.independent_shape, (
            f"Non-diagonal shape for diagonal Jacobian {name}"
            f"{self.dependent_shape} != {self.independent_shape}"
        )
        assert self.depdent_size == self.independent_size, (
            f"Non-diagonal size for diagonal Jacobian {name}"
            f"{self.dependent_size} != {self.independent_size}"
        )
        assert self.data.shape == self.dependent_shape, (
            f"Array shape mismatch for {name}, "
            f"{self.data.shape} != {self.dependent_shape}"
        )

    def _getjitem(self, new_shape, key):
        """A getitem type method for diagonal Jacobians"""
        # OK, once we extract items, this will no longer be diagonal,
        # so we convert to sparse before doing the subset.
        self_sparse = SparseJacobian(self)
        result = self_sparse._getjitem(new_shape, key)
        return result

    def _setjitem(self, key, value):
        """A setitem type method for diagonal Jacobians"""
        # OK, once we insert items, this will no longer be diagonal,
        # so we convert to sparse before doing the subset
        self = SparseJacobian(self)
        self._setjitem(key, value)

    def broadcast_to(self, shape):
        """Broadcast diagonal jacobian to new dependent shape"""
        # OK, once you broadcast a diagonal, it is not longer,
        # strictly speaking, a diagonal So, convert to sparse and
        # broadcast that.  However, don't bother doing anything if
        # there is no actual broadcast going on.
        if shape == self.dependent_shape:
            return self
        self_sparse = SparseJacobian(self)
        return self_sparse.broadcast_to(shape)

    def reshape(self, shape, order, parent_flags):
        # OK, once you reshape a diagonal, it is not longer,
        # strictly speaking, a diagonal So, convert to sparse and
        # reshape that.  However, don't bother doing anything if
        # there is no actual reshape going on.
        if shape == self.dependent_shape:
            return self
        self_sparse = SparseJacobian(self)
        return self_sparse.reshape(shape, order, parent_flags)

    def premul_diag(self, diag):
        """Diagonal premulitply for diagonal Jacobian"""
        diag_, dependent_unit, dependent_shape = self._prepare_premul_diag(diag)
        if dependent_shape == self.independent_shape:
            return DiagonalJacobian(
                diag_ * self.data, template=self, dependent_unit=dependent_unit
            )
        else:
            return SparseJacobian(self).premul_diag(diag)

    def insert(self, obj, axis, dependent_shape):
        """insert method for diagonal Jacobian"""
        # By construction this is no longer diagonal once inserted to
        # change to sparse and insert there.
        self_sparse = SparseJacobian(self)
        return self_sparse.insert(obj, axis, dependent_shape)

    def diff(
        self, dependent_shape, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue
    ):
        """diff method for diagonal jacobian"""
        # Again, the result will not be diagonal, so change to sparse and do diff in
        # that space.
        self_sparse = SparseJacobian(self)
        return self_sparse.diff(dependent_shape, n, axis, prepend, append)

    def transpose(self, axes, result_dependent_shape):
        return DiagonalJacobian(
            data=self.data.transpose(axes),
            template=self,
            dependent_shape=result_dependent_shape,
        )

    def tensordot(self, other, axes, dependent_unit, reverse_order=False):
        """Compute self(.)other, or other(.)self if reverse_order is True"""
        # Once we do this, we will no longer be diagonal, so convert to sparse
        self_sparse = SparseJacobian(self)
        return self_sparse.tensordot(other, axes, dependent_unit, reverse_order)

    def sum(self, dependent_shape, axis=None, dtype=None, keepdims=False):
        """Performs sum for the diagonal Jacobians"""
        # Once we take the sum, along any or all axes, the jacobian is
        # no longer diagonal by construction, so it needs to be
        # converted to sparse.
        self_sparse = SparseJacobian(self)
        return self_sparse.sum(
            dependent_shape, axis=axis, dtype=dtype, keepdims=keepdims
        )

    def cumsum(self, axis):
        """Perform cumsum for the diagonal Jacobians"""
        # Once it's been cumsummed then it's no longer diagonal.  For
        # that matter it's not going to be particularly sparse either,
        # so we might as well convert it to dense.
        return DenseJacobian(self).cumsum(axis)

    def diagonal(self):
        """Get diagonal elements (shape=dependent_shape)"""
        return self.data << (self.dependent_unit / self.independent_unit)

    # The reaons we have extract_diagonal and diagonal is that diagonal is
    # only populated for diagonal Jacobians.  extract_diagonal is
    # populated for all.
    def extract_diagonal(self):
        """Extract the diagonal from a diagonal Jacobian"""
        return self.diagonal()

    def todensearray(self):
        from .dense_jacobians import DenseJacobian

        self_dense = DenseJacobian(self)
        return self_dense.todensearray()

    def to2darray(self):
        self_sparse = SparseJacobian(self)
        return self_sparse.to2darray()

    def to2ddensearray(self):
        self_dense = DenseJacobian(self)
        return self_dense.to2ddensearray()

    def linear_interpolator(self, x_in, axis=-1):
        """Return an interpolator for a given Jacobian axis"""
        return SparseJacobian(self).linear_interpolator(x_in, axis)


class SeedJacobian(DiagonalJacobian):
    """A specific subclass of DiagonalJacobian for seeds

    There is no substantive difference.  The only intent is to indicate that this is an
    actual seed (it's only use thus far is to help the numeric Jacobian computation code
    identify what are the seeded variables so it can perturb them.

    """
    pass
