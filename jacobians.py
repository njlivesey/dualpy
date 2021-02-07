"""The various Jacobians for duals"""

__all__ = [
    "BaseJacobian",
    "DiagonalJacobian",
    "DenseJacobian",
    "SparseJacobian",
]

from .base_jacobian import BaseJacobian
from .dense_jacobians import DenseJacobian
from .diagonal_jacobians import DiagonalJacobian
from .sparse_jacobians import SparseJacobian


def _setitem_jacobians(key, target, target_jacobians, source_jacobians):
    """Called by dual __setitem__ to set jacobian items"""
    # Loop over the jacobians in the value (which is the source)
    for name, source_j in source_jacobians.items():
        if name not in target_jacobians:
            if isinstance(source_j, DenseJacobian):
                newj = DenseJacobian
            elif isinstance(source_j, SparseJacobian):
                newj = SparseJacobian
            elif isinstance(source_j, DiagonalJacobian):
                newj = SparseJacobian
            else:
                raise TypeError(f"Unrecognized type for jacobian {type(source_j)}")
            target_jacobians[name] = newj(
                dependent_unit=target.unit,
                independent_unit=source_j.independent_unit,
                dependent_shape=target.shape,
                independent_shape=source_j.independent_shape,
                dtype=source_j.dtype,
            )
        # Now insert the values
        target_jacobians[name]._setjitem(key, source_j)
