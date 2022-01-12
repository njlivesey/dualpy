"""Helper routines for the dual jacobians"""

import numpy as np
import scipy.sparse as sparse
import itertools

__all__ = [
    "_array_to_sparse_diagonal",
    "_broadcasted_shape",
    "_shapes_broadcastable",
    "_prepare_jacobians_for_binary_op",
    "jacobian_2d_matrix_multiply",
]


def _array_to_sparse_diagonal(x):
    """Turn an ndarray into a diagonal, stored as csc"""
    x_ = np.array(x).ravel()
    result = sparse.diags(x_, 0, format="csc")
    return result


def _broadcasted_shape(shp1, shp2):
    # Return the broadcasted shape of the two arguments
    # Also check's they're legal
    result = []
    for a, b in itertools.zip_longest(shp1[::-1], shp2[::-1], fillvalue=1):
        if a == 1 or b == 1 or a == b:
            result.append(max(a, b))
        else:
            raise ValueError(f"Arrays not broadcastable {shp1} and {shp2}")
    return tuple(result[::-1])


def _shapes_broadcastable(shp1, shp2):
    # Test if two shapes can be broadcast together
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def _prepare_jacobians_for_binary_op(a, b):
    """Take two Jacobians about to have something binary done to them and
    return their contents in mutually compatible form from a units
    perspective, and as efficiently as possible"""
    from .sparse_jacobians import SparseJacobian
    from .diagonal_jacobians import DiagonalJacobian

    # Note that this code does not need to worry about broadcasting,
    # as that is handled elsewhere in this class, and invoked by the
    # methods in dlarray.  Note that the only thing we need to prepare
    # for here is addition, so we don't need to predict the shape of
    # the result.
    scale = 1.0
    if b.dependent_unit != a.dependent_unit:
        scale *= b.dependent_unit._to(a.dependent_unit)
    if b.independent_unit != a.independent_unit:
        scale /= b.independent_unit._to(a.independent_unit)
    # Now go throught the various type combinations
    if type(a) is type(b):
        # If they are both the same type, then things are pretty
        # straight forward.  If they are the same type, then their
        # data attributes are in the same form.
        if type(a) is SparseJacobian:
            a_ = a.data2d
            b_ = b.data2d
        else:
            a_ = a.data
            b_ = b.data
        result_type = type(a)
    elif type(a) is DiagonalJacobian:
        # If a is diagonal (and by implication b is not, otherwise the
        # above code would have handled things), then promote a to
        # sparse and use the 2d view of b
        a_ = _array_to_sparse_diagonal(a.data)
        b_ = b.data2d
        result_type = type(b)
    elif type(b) is DiagonalJacobian:
        # This is the converse caseb
        a_ = a.data2d
        b_ = _array_to_sparse_diagonal(b.data)
        result_type = type(a)
    elif type(a) is SparseJacobian:
        # OK, so, here a must be sparse, b dense
        a_ = a.data2d
        b_ = b.data2d
        result_type = type(b)
    elif type(b) is SparseJacobian:
        # Finally, so it must be that a is dense, b sparse
        a_ = a.data2d
        b_ = b.data2d
        result_type = type(a)
    else:
        raise AssertionError("Failed to understand binary Jacobian operation")
    # If needed, put them in the same units by scaling b to be in a's
    # units
    if scale != 1.0:
        b_ = b_ * scale
    return a_, b_, result_type


def jacobian_2d_matrix_multiply(a, b):
    """Do a 2D matrix multiply on two Jacobians, returning a third."""
    from .dense_jacobians import DenseJacobian
    from .sparse_jacobians import SparseJacobian
    from .diagonal_jacobians import DiagonalJacobian

    # Check that the dimensions and units are agreeable
    if a.independent_shape != b.dependent_shape:
        raise ValueError("Shape mismatch for dense Jacobian matrix multiply")
    if a.independent_unit != b.dependent_unit:
        raise ValueError("Units mismatch for dense Jacobian matrix multiply")
    # Recast any diagonal Jacobians into sparse
    if isinstance(a, DiagonalJacobian):
        a = SparseJacobian(a)
    if isinstance(b, DiagonalJacobian):
        b = SparseJacobian(b)
    # Decide what our result type will be
    if isinstance(a, SparseJacobian) and isinstance(b, SparseJacobian):
        result_type = SparseJacobian
    else:
        result_type = DenseJacobian
    # OK, do the matrix multiplication
    result_data2d = a.data2d @ b.data2d
    # Work out its true shape
    result_shape = a.dependent_shape + b.independent_shape
    if result_type is DenseJacobian:
        result_data = result_data2d.reshape(result_shape)
    else:
        result_data = result_data2d
    return result_type(
        data=result_data,
        dependent_shape=a.dependent_shape,
        independent_shape=b.independent_shape,
        dependent_unit=a.dependent_unit,
        independent_unit=b.independent_unit,
    )
