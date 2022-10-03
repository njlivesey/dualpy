"""Some low level routines to help with handling duals"""

import numpy as np
__all__ = ["dedual", "broadcast_jacobians", "setup_dual_operation", "force_unit"]


def dedual(x):
    """Return duck-array form of x that's not a dual

    Thus, it will return x.variable if x a dlarray or just x (at least for now,
    perhaps we should check x over in the latter case, but we'll only do so if it
    becomes clear we need to.

    """
    from .duals import dlarray
    if isinstance(x, dlarray):
        return x.variable
    elif isinstance(x, np.ndarray) or hasattr(x, "__array__"):
        return x
    elif isinstance(x, int) or isinstance(x, float) or isinstance(x, complex):
        return x
    else:
        raise ValueError("Do not know how to dedual object of type {type(x)}")


def broadcast_jacobians(js, new):
    # Loop over jacobians and broadcast each of them to new shape
    out = {}
    for name, jacobian in js.items():
        out[name] = jacobian.broadcast_to(new)
    return out


def force_unit(quantity, *, unit=None, source=None):
    """Apply a unit to a quantity"""
    return quantity


def setup_dual_operation(*args, out=None, broadcast=True):
    # Get the variables for all the arguments, be they duals or a non-dual
    # duck-array, strip the units off for now.
    arrays_ = [dedual(x) for x in args]
    if broadcast:
        # Down the road, pint will give us problems here.
        arrays_ = np.broadcast_arrays(*arrays_, subok=True)
    # Now go through the jacobians
    jacobians = []
    for x, orig in zip(arrays_, args):
        if hasattr(orig, "jacobians"):
            try:
                x_shape = x.shape
            except AttributeError:
                x_shape = []
            if orig.shape != x_shape and broadcast:
                j = broadcast_jacobians(orig.jacobians, x_shape)
            else:
                j = orig.jacobians
        else:
            j = {}
        jacobians.append(j)

    # Handle the case where an "out" is provided. I originally had some intelligence
    # about tracking whether out shares memory with any of the arguments but that
    # got complicated. In the end I've recognized that we can code up the binary
    # operations such that we don't need to worry about such cases.
    if out is not None:
        if isinstance(out, tuple):
            if len(out) != 1:
                raise NotImplementedError("Cannot support multiple outs")
            out = out[0]
        # As inefficient as this might appear, I'm pretty sure I need to blow away
        # the Jacobians in out and let the calling code recreate them from scratch.
        # Unpleasent things happen if not, as things done to out.jacobians leak back
        # on a and b's jacobians.
        if hasattr(out, "jacobians"):
            out.jacobians = {}

    return tuple(arrays_) + tuple(jacobians) + (out,)
