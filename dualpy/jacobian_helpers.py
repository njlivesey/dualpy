"""Helper routines for the dual jacobians"""

import numpy as np
import scipy.sparse as sparse
import itertools

from .dual_helpers import get_unit_conversion_scale, get_magnitude


__all__ = [
    "array_to_sparse_diagonal",
    "broadcasted_shape",
    "prepare_jacobians_for_binary_op",
    "shapes_broadcastable",
    "jacobian_2d_matrix_multiply",
    "GenericUnit",
]


def array_to_sparse_diagonal(x):
    """Turn an ndarray into a diagonal, stored as csc"""
    result = sparse.diags(np.ravel(x), 0, format="csc")
    return result


def broadcasted_shape(shp1, shp2):
    # Return the broadcasted shape of the two arguments
    # Also check's they're legal
    result = []
    for a, b in itertools.zip_longest(shp1[::-1], shp2[::-1], fillvalue=1):
        if a == 1 or b == 1 or a == b:
            result.append(max(a, b))
        else:
            raise ValueError(f"Arrays not broadcastable {shp1} and {shp2}")
    return tuple(result[::-1])


def shapes_broadcastable(shp1, shp2):
    # Test if two shapes can be broadcast together
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def prepare_jacobians_for_binary_op(a, b):
    """Take two Jacobians about to have something binary done to them and
    return their contents in mutually compatible form from a units
    perspective, and as efficiently as possible"""
    from .sparse_jacobians import SparseJacobian
    from .diagonal_jacobians import DiagonalJacobian, SeedJacobian

    # Note that this code does not need to worry about broadcasting,
    # as that is handled elsewhere in this class, and invoked by the
    # methods in dlarray.  Note that the only thing we need to prepare
    # for here is addition, so we don't need to predict the shape of
    # the result.
    scale = 1.0
    if b.dependent_unit != a.dependent_unit:
        scale *= get_magnitude(
            get_unit_conversion_scale(b.dependent_unit, a.dependent_unit)
        )
    if b.independent_unit != a.independent_unit:
        scale /= get_magnitude(
            get_unit_conversion_scale(b.independent_unit, a.independent_unit)
        )
    # Now go throught the various type combinations
    type_a = type(a)
    type_b = type(b)
    if type_a is SeedJacobian:
        type_a = DiagonalJacobian
    if type_b is SeedJacobian:
        type_b = DiagonalJacobian
    if type_a is type_b:
        # If they are both the same type, then things are pretty
        # straight forward.  If they are the same type, then their
        # data attributes are in the same form.
        a_ = a.data
        b_ = b.data
        result_type = type_a
    elif type_a is DiagonalJacobian:
        # If a is diagonal (and by implication b is not, otherwise the
        # above code would have handled things), then promote a to
        # sparse and use the 2d view of b
        a_ = array_to_sparse_diagonal(a.data)
        b_ = b.data
        result_type = type_b
    elif type_b is DiagonalJacobian:
        # This is the converse case
        a_ = a.data
        b_ = array_to_sparse_diagonal(b.data)
        result_type = type_a
    elif type_a is SparseJacobian:
        # OK, so, here a must be sparse, b dense
        a_ = a.data
        b_ = b.data
        result_type = type_b
    elif type_b is SparseJacobian:
        # Finally, so it must be that a is dense, b sparse
        a_ = a.data
        b_ = b.data
        result_type = type_a
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

    raise NotImplementedError("Pretty sure this has bugs, as doesn't deal with units")
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


def linear_interpolation_indices_and_coordinates(c_in, c_out):
    """Return lower/upper indices and coordiante values for an interpolation panel"""
    n = len(c_in)
    i_upper = np.searchsorted(c_in, c_out)
    i_lower = i_upper - 1
    i_upper = np.minimum(i_upper, n - 1)
    i_lower = np.maximum(i_lower, 0)
    c_lower = c_in[i_lower]
    c_upper = c_in[i_upper]
    return i_lower, i_upper, c_lower, c_upper


def linear_interpolation_indices_and_weights(c_in, c_out, extrapolate=None):
    """Return lower/upper indicies and weights for an interpolation panel"""
    i_lower, i_upper, c_lower, c_upper = linear_interpolation_indices_and_coordinates(
        c_in, c_out
    )
    if hasattr(c_in, "unit"):
        one = 1.0 * c_in.unit
    else:
        one = 1.0
    # The case where c_upper=c_lower, which implies we're over one edge or the
    # other is annoying. While the clipping of the eventual weight to 0..1 means
    # the code works, the division by zero warnings are tiresome. To avoid such
    # cases, we have a where statement.
    not_edge = c_upper > c_lower
    delta = np.where(not_edge, c_upper - c_lower, one)
    try:
        tiny = np.finfo(delta.dtype).tiny
        small = np.sqrt(tiny)
        if hasattr(delta, "unit"):
            raise NotImplementedError("This still uses astropy!")
            small = small << delta.unit
    except ValueError:
        small = 1
    delta = np.maximum(delta, small)
    # And another where, to handle our rough fix to get round the warnings
    # assocaited with going over the edge
    w_upper = np.where(not_edge, (c_out - c_lower) / delta, 1.0)
    if not extrapolate:
        w_upper = np.where(
            (w_upper >= 0.0) & (w_upper <= 1.0),
            w_upper,
            np.nan,
        )
    else:
        if extrapolate is True:
            w_upper = np.clip(w_upper, 0.0, 1.0)
        else:
            raise ValueError(f"Unable to handle extrapolate={extrapolate}")
    w_lower = 1.0 - w_upper
    return i_lower, i_upper, w_lower, w_upper


# pylint: disable=import-outside-toplevel
def get_unit_class():
    """Generate a generic unit class for type checking"""
    from pint import Unit as PintUnit
    from astropy.units import Unit as AstropyUnit
    from .unitless import Unitless

    return Unitless | PintUnit | AstropyUnit


GenericUnit = get_unit_class()
