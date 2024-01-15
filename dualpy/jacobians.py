"""The various Jacobians for duals"""

import numpy as np
import scipy.sparse as sparse

from .base_jacobian import BaseJacobian
from .dense_jacobians import DenseJacobian
from .diagonal_jacobians import DiagonalJacobian
from .sparse_jacobians import SparseJacobian
from .dual_helpers import get_unit


def setitem_jacobians(key, target, target_jacobians, source_jacobians):
    """Called by dual __setitem__ to set jacobian items"""
    # Loop over the jacobians in the value (which is the source)
    for name, source_jacobian in source_jacobians.items():
        if name not in target_jacobians:
            if isinstance(source_jacobian, DenseJacobian):
                result_type = DenseJacobian
            elif isinstance(source_jacobian, SparseJacobian):
                result_type = SparseJacobian
            elif isinstance(source_jacobian, DiagonalJacobian):
                source_jacobian = SparseJacobian(source_jacobian)
                result_type = SparseJacobian
            else:
                raise TypeError(
                    f"Unrecognized type for jacobian {type(source_jacobian)}"
                )
            target_jacobians[name] = result_type(
                dependent_unit=get_unit(target),
                independent_unit=source_jacobian.independent_unit,
                dependent_shape=target.shape,
                independent_shape=source_jacobian.independent_shape,
                dtype=source_jacobian.dtype,
            )
        # Now insert the values
        #
        # pylint: disable=protected-access
        target_jacobians[name]._setjitem(key, source_jacobian)


def prep_jacobians_for_join(*args, result_dependent_shape):
    """Used by insert, append, concatenate, others(?) to prepare Jacobians"""
    # pylint: disable=import-outside-toplevel
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
                j = arg.jacobians[name]
            except (AttributeError, KeyError):
                j = None
            if j is not None:
                # If we got a Jacobian, make sure it's of the right type
                j = result_type(j)
            else:
                # Otherwise, create an empty one
                j = result_type(
                    dependent_shape=getattr(arg, "shape", tuple()),
                    dependent_unit=get_unit(arg),
                    independent_shape=template.independent_shape,
                    independent_unit=template.independent_unit,
                )
            these_prepped_jacobians.append(j)
        prepped_jacobians[name] = these_prepped_jacobians
    # Return the information in a tuple
    return prepped_jacobians, result_types, result_jacobians


def join_jacobians(a, b, location, axis, result_dependent_shape):
    """Used by insert and append to do the work for Jocobians"""
    # pylint: disable=unused-variable
    prepped_jacobians, result_types, result_jacobians = prep_jacobians_for_join(
        a, b, result_dependent_shape=result_dependent_shape
    )
    for name, item in result_jacobians.items():
        aj, bj = prepped_jacobians[name]
        # pylint: disable=protected-access
        result_jacobians[name] = aj._join(bj, location, axis, result_dependent_shape)
    return result_jacobians


def concatenate_jacobians(values, axis, result_dependent_shape):
    """Concatenate the Jacobians, supports dual concatenation"""
    prepped_jacobians, result_types, result_jacobians = prep_jacobians_for_join(
        *values, result_dependent_shape=result_dependent_shape
    )
    # Loop over the Jacobians in the result.
    result = {}
    for name, result_type in result_types.items():
        # Identify the correspondign axis in the jacobian
        # pylint: disable-next=protected-access
        jaxis = result_jacobians[name]._get_jaxis(axis)
        # This is cumbersome, but the best way to do this is separately for dense and
        # sparse
        if result_type is DenseJacobian:
            j_ins_ = [j_in.data for j_in in prepped_jacobians[name]]
            j_out_ = np.concatenate(j_ins_, axis=jaxis)
            result[name] = result_type(template=result_jacobians[name], data=j_out_)
        elif result_type is SparseJacobian:
            i = 0
            j_out = result_type(template=result_jacobians[name])
            for j_in in prepped_jacobians[name]:
                # Use the coo form of sparse matrices to move the values up to the right
                # place in the stacked axis
                j_in_coo = sparse.coo_matrix(j_in.data2d)
                row_indices = list(
                    np.unravel_index(j_in_coo.row, shape=j_in.dependent_shape)
                )
                row_indices[jaxis] += i
                out_row = np.ravel_multi_index(row_indices, j_out.dependent_shape)
                j_out_contribution = sparse.coo_matrix(
                    (j_in_coo.data, (out_row, j_in_coo.col)), shape=j_out.shape2d
                )
                j_out.data2d += sparse.csc_matrix(j_out_contribution)
                # Increment the start index
                i += j_in.dependent_shape[jaxis]
            result[name] = j_out
        else:
            raise TypeError(f"Unexpcted Jacobian type in result {result_type}")
    return result


def stack_jacobians(arrays, axis, result_dependent_shape):
    """Support the numpy.stack operationf for Jacobians"""
    prepped_jacobians, result_types, result_jacobians = prep_jacobians_for_join(
        *arrays, result_dependent_shape=result_dependent_shape
    )
    # Loop over the jacobians in the result
    result = {}
    for name, result_type in result_types.items():
        # Identify teh corresponding axis in the jacobian
        #
        # pylint: disable=protected-access
        jaxis = result_jacobians[name]._get_jaxis(axis)
        # This is cumbersome, but the best way to do this is separately for dense and
        # sparse
        if result_type is DenseJacobian:
            j_ins_ = [j_in.data for j_in in prepped_jacobians[name]]
            j_out_ = np.stack(j_ins_, axis=jaxis)
            result[name] = result_type(template=result_jacobians[name], data=j_out_)
        elif result_type is SparseJacobian:
            j_out = result_type(template=result_jacobians[name])
            for i, j_in in enumerate(prepped_jacobians[name]):
                # Use the coo form of sparse matrices to move the values up to the right
                # place in the stacked axis
                j_in_coo = sparse.coo_matrix(j_in.data2d)
                row_indices = list(
                    np.unravel_index(j_in_coo.row, shape=j_in.dependent_shape)
                )
                row_indices.insert(jaxis, i)
                out_row = np.ravel_multi_index(row_indices, j_out.dependent_shape)
                j_out_contribution = sparse.coo_matrix(
                    (j_in_coo.data, (out_row, j_in_coo.col)), shape=j_out.shape2d
                )
                j_out.data2d += sparse.csc_matrix(j_out_contribution)
            result[name] = j_out
        else:
            raise TypeError(f"Unexpcted Jacobian type in result {result_type}")
    return result


def matrix_multiply_jacobians(a, b):
    """Perform a matrix multiply on two Jacobians"""
    # I need this rarely enough that I'm not going to do it as an operator, in order to
    # avoid doing it by mistake.
    # First check that things are going to make sense.
    scale = 1.0
    if a.independent_unit != b.dependent_unit:
        scale *= b.dependent_unit.to(a.dependent_unit)
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
