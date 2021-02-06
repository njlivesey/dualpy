"""The class for diagonal jacobians"""

import numpy as np

from .base_jacobian import BaseJacobian
# Seeing as we cast to sparse so often, we'll import it globally
from .sparse_jacobians import SparseJacobian


__all__ = ["DiagonalJacobian"]


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
        else:
            data_ = data
        if data_.shape != self.dependent_shape:
            raise ValueError(
                "Attempt to create a diagonal Jacobian using wrong-shaped input"
            )
        self.data = data_

    def __str__(self):
        return super().__str__() + f"\ndata is {self.data.shape}"

    def _getjitem(self, new_shape, key):
        """A getitem type method for diagonal Jacobians"""
        # OK, once we extract items, this will no longer be diagonal,
        # so we convert to sparse before doing the subset.
        self_sparse = SparseJacobian(data=self)
        result = self_sparse._getjitem(new_shape, key)
        return result

    def _setjitem(self, key, value):
        """A setitem type method for diagonal Jacobians"""
        # OK, once we insert items, this will no longer be diagonal,
        # so we convert to sparse before doing the subset
        self = SparseJacobian(data=self)
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

    def reshape(self, shape, order="C"):
        # OK, once you reshape a diagonal, it is not longer,
        # strictly speaking, a diagonal So, convert to sparse and
        # reshape that.  However, don't bother doing anything if
        # there is no actual reshape going on.
        if shape == self.dependent_shape:
            return self
        self_sparse = SparseJacobian(self)
        return self_sparse.reshape(shape, order)

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

    def diff(self, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
        """diff method for diagonal jacobian"""
        # Again, the result will not be diagonal, so change to sparse and do diff in
        # that space.
        self_sparse = SparseJacobian(self)
        return self_sparse.diff(n, axis, prepend, append)

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
        self_dense = DenseJacobian(self)
        return self_dense.todensearray()

    def to2darray(self):
        self_sparse = SparseJacobian(self)
        return self_sparse.to2darray()

    def to2ddensearray(self):
        self_dense = DenseJacobian(self)
        return self_dense.to2ddensearray()
