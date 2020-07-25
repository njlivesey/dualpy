"""The various array functions for duals"""

import numpy as np

from .dual_helpers import *

# Now some stuff getting ready to deal with __array_funciton__
HANDLED_FUNCTIONS={}
__all__ = ["HANDLED_FUNCTIONS"]

def implements(numpy_function):
    """Register an __array_function__ implementation for dlarray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function]=func
        return func
    return decorator

@implements(np.sum)
def sum(a, axis=None, dtype=None, keepdims=False):
    a_, aj, out=_setup_dual_operation(a)
    out = dlarray(np.sum(a_, axis=axis, dtype=dtype, keepdims=keepdims))
    for name, jacobian in aj.items():
        out.jacobians[name] = jacobian.sum(
            out.shape, axis=axis, dtype=dtype, keepdims=keepdims)
    return out

@implements(np.cumsum)
def cumsum(a, axis=None, dtype=None, out=None):
    if out is not None:
        raise NotImplementedError("out not supported for dual cumsum (yet?)")
    a_, aj, out=_setup_dual_operation(a)
    out = dlarray(np.cumsum(a_, axis=axis, dtype=dtype))
    for name, jacobian in aj.items():
        out.jacobians[name] = jacobian.cumsum(axis)
    return out

@implements(np.broadcast_arrays)
# One of these days I should look into whether broadcast_arrays and
# broadcast_to really need the subok argument, given that it's ignored.
def broadcast_arrays(*args, subok=False):
    values=[]
    for a in args:
        values.append(a.value)
    result_=np.broadcast_arrays(*values, subok=subok)
    shape=result_[0].shape
    result=[]
    for i, a in enumerate(args):
        thisResult=dlarray(result_[i] << a.unit)
        if hasattr(a, "jacobians"):
            thisResult.jacobians=_broadcast_jacobians(a.jacobians, shape)
        result.append(thisResult)
    return result


@implements(np.broadcast_to)
def broadcast_to(array, shape, subok=False):
    result_=np.broadcast_to(array.value, shape, subok=subok) << array.unit
    result=dlarray(result_)
    result.jacobians=_broadcast_jacobians(array.jacobians, shape)
    return result


@implements(np.reshape)
def reshape(array, *args, **kwargs):
    result=dlarray(units.Quantity(array).reshape(*args, **kwargs))
    for name, jacobian in array.jacobians.items():
        result.jacobians[name] = jacobian.reshape(result.shape)
    return result

@implements(np.atleast_1d)
def atleast_1d(*args):
    result=[]
    for a in args:
        a1d=np.atleast_1d(units.Quantity(a))
        a1d.jacobians=a.jacobians.copy()
        result.append(a1d)
    return tuple(result)

@implements(np.where)
def where(condition, a=None, b=None):

    if a is None or b is None:
        return NotImplemented
    cond_, a_, b_, condj, aj, bj, out=_setup_dual_operation(condition, a, b)
    if condj:
        raise ValueError(
            "Jacobians not allowed on condition argument in 'where'")
    out=dlarray(np.where(condition, a_, b_))
    # Now go through the jacobians and insert them where the condition
    # applies, otherwise they're zero.
    for name, jacobian in aj.items():
        out.jacobians[name]=jacobian.premul_diag(cond_)
    for name, jacobian in bj.items():
        if name in out.jacobians:
            out.jacobians[name] += jacobian.premul_diag(np.logical_not(cond_))
        else:
            out.jacobians[name]=jacobian.premul_diag(np.logical_not(cond_))
    return out

@implements(np.insert)
def insert(arr, obj, values, axis=None):
    # Note that this is supposed to flatten the array first if axis is None
    if axis is None:
        arr = arr.flatten()
    # print (type(arr), arr.jacobians)
    result=dlarray(np.insert(units.Quantity(arr), obj, values, axis))
    # In principal the user could want to insert a dual, that is that
    # the values inserted have derivatives.  However, managing that
    # will be a bit of a pain, so I'll skip it for now, and thus
    # assert that we will not insert any values in the jacobians,
    # because the inserted Jacobian values should be zero.
    if isinstance(values, dlarray):
        raise NotImplementedError("The values inserted cannot be a dual")
    for name, jacobian in arr.jacobians.items():
        result.jacobians[name] = jacobian.insert(obj, axis, result.shape)
    return result

@implements(np.searchsorted)
def searchsorted(a, v, side="left", sorter=None):
    return np.searchsorted(units.Quantity(a), units.Quantity(v), side=side, sorter=sorter)


@implements(np.clip)
def clip(a, a_min, a_max, out=None, **kwargs):
    if type(a_min) is dlarray:
        raise NotImplementedError(
            "dlarray.clip does not (currently) support dual for a_min, use dualarray.minimum")
    if type(a_max) is dlarray:
        raise NotImplementedError(
            "dlarray.clip does not (currently) support dual for a_max, use dualarray.maximum")
    if out is not None:
        raise NotImplementedError("dlarray.clip cannot support out")
    if len(kwargs) != 0:
        raise NotImplementedError(
            "dlarray.clip cannot support non-empty kwargs")
    out=dlarray(np.clip(units.Quantity(a), a_min, a_max))
    if a.hasJ:
        factor=np.logical_and(a >= a_min, a <= a_max)
        out._chain_rule(a, factor.astype(int))
    return out

