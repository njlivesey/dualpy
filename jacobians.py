"""The various Jacobians for duals"""

__all__ = [
    "BaseJacobian",
    "DiagonalJacobian",
    "DenseJacobian",
    "SparseJacobian",
    "matrix_multiply_jacobians",
]

import astropy.units as units
import numpy as np

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


def _prep_jacobians_for_join(*args, result_dependent_shape):
    """Used by insert, append, concatenate, others(?) to prepare Jacobians"""
    from .user import has_jacobians

    # Get a list of all the Jacobian names
    jnames = set()
    for arg in args:
        if has_jacobians(arg):
            jnames = jnames.union(set(arg.jacobians.keys()))
    # Now loop over all these Jacobians and deal with them
    prepped_jacobians = {}
    result_jacobians = {}
    result_types = {}
    for name in jnames:
        # Go through the Jacobians and work out what we're going to need to do.
        result_type = SparseJacobian
        template = None
        for arg in args:
            # Get this jacobian in arg if it exists
            try:
                j = arg.jacobians[name]
            except (AttributeError, KeyError):
                j = None
            if j is not None:
                # If this is the first Jacobian record it, otherwise check it
                if template is None:
                    template = j
                else:
                    if not template.independents_compatible(j):
                        raise ValueError("The independent variables are not compatible")
                # If this is dense, then we might as well make the result dense
                if isinstance(j, DenseJacobian):
                    result_type = DenseJacobian
        # Now prepare a BaseJacobian for the result
        result_jacobians[name] = BaseJacobian(
            template=template, dependent_shape=result_dependent_shape
        )
        result_types[name] = result_type
        # Now go through again and convert all the Jacobians to the target type
        these_prepped_jacobians = []
        for arg in args:
            try:
                j = arg.jacobans[name]
            except (AttributeError, KeyError):
                j = None
            if j is not None:
                # If we got a Jacobian, make sure it's of the right type
                j = result_type(j)
            else:
                # Otherwise, create an empty one
                j = result_type(
                    dependent_shape=getattr(arg, "shape", tuple()),
                    dependent_unit=getattr(arg, "unit", units.dimensionless_unscaled),
                    independent_shape=template.independent_shape,
                    independent_unit=template.independent_unit,
                )
            these_prepped_jacobians.append(j)
        prepped_jacobians[name] = these_prepped_jacobians
    # Return the information in a tuple
    return prepped_jacobians, result_types, result_jacobians


def _join_jacobians(a, b, location, axis, result_dependent_shape):
    """Used by insert and append to do the work for Jocobians"""
    prepped_jacobians, result_types, result_jacobians = _prep_jacobians_for_join(
        a, b, result_dependent_shape=result_dependent_shape
    )
    for name, item in result_jacobians.items():
        aj, bj = prepped_jacobians[name]
        result_jacobians[name] = aj._join(bj, location, axis, result_dependent_shape)
    return result_jacobians


def matrix_multiply_jacobians(a, b):
    """Perform a matrix multiply on two Jacobians"""
    # I need this rarely enough that I'm not going to do it as an operator, in order to
    # avoid doing it by mistake.
    # First check that things are going to make sense.
    scale = 1.0
    if a.independent_unit != b.dependent_unit:
        try:
            scale *= b.dependent_unit.to(a.dependent_unit)
        except units.UnitsError:
            raise ValueError("Units not compatible for Jacobian matrix multiply")
    if a.independent_shape != b.dependent_shape:
        raise ValueError(
            f"Shapes not compatible for Jacobian matrix multiply "
            f"{a.independent_shape} vs. {b.dependent_shape}"
        )
    # If any Jacobians are diagonal, make them sparse for an easier time.
    if isinstance(a, DiagonalJacobian):
        a = SparseJacobian(a)
    if isinstance(b, DiagonalJacobian):
        b = SparseJacobian(b)
    # Now work out what the reuslt will be
    if isinstance(a, DenseJacobian) or isinstance(b, DenseJacobian):
        result_type = DenseJacobian
    else:
        result_type = SparseJacobian
    result_values = a.data2d @ b.data2d
    result_template = BaseJacobian(
        dependent_shape=a.dependent_shape,
        dependent_unit=a.dependent_unit,
        independent_shape=b.independent_shape,
        independent_unit=b.independent_unit,
    )
    if result_type is DenseJacobian:
        result_values = np.reshape(result_values, result_template.shape)
    return result_type(result_values, template=result_template)
