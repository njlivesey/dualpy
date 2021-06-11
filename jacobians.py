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


def _join_jacobians(a, b, location, axis, result_dependent_shape):
    """Used by insert, append, others? to merge jacobians"""
    from .user import has_jacobians

    # Get a list of all the jacobans in a and b
    jnames = set()
    if has_jacobians(a):
        jnames = jnames.union(set(a.jacobians.keys()))
    if has_jacobians(b):
        jnames = jnames.union(set(b.jacobians.keys()))
    # Now loop over all these jacobians and try to formulate a result.
    result = {}
    for name in jnames:
        # Find the jacobians in a and b, make them None if not included
        js = []
        for x in [a, b]:
            try:
                j = x.jacobians[name]
                # Any diagonal Jacobians should be converted to sparse
                if isinstance(j, DiagonalJacobian):
                    j = SparseJacobian(j)
            except (AttributeError, KeyError):
                j = None
            js.append(j)
        aj, bj = js
        # Now, if we did get a jacobian for both, make sure they're compatible
        if isinstance(aj, BaseJacobian) and isinstance(bj, BaseJacobian):
            if not aj.independents_compatible(bj):
                raise ValueError("The independent variables are not compatible")
        # Now we try to work out whether we're going to end up with a dense or sparse
        # result
        result_type = None
        if isinstance(aj, DenseJacobian) or isinstance(bj, DenseJacobian):
            result_type = DenseJacobian
        else:
            result_type = SparseJacobian
        # Now make sure aj and bj are of type result_type (they still might be None)
        if aj is None:
            aj = result_type(
                dependent_shape=a.shape,
                dependent_unit=a.unit,
                independent_shape=bj.independent_shape,
                independent_unit=bj.independent_unit,
            )
        if bj is None:
            bj = result_type(
                dependent_shape=b.shape,
                dependent_unit=b.unit,
                independent_shape=aj.independent_shape,
                independent_unit=aj.independent_unit,
            )
        # Now, finally, we're able to call upon the join method
        result[name] = aj._join(bj, location, axis, result_dependent_shape)
    return result
