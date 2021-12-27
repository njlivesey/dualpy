"""Some helper routines for duals"""

import numpy as np
import astropy.units as units

__all__ = ["_per_rad", "_broadcast_jacobians", "_setup_dual_operation"]

_per_rad = units.rad ** (-1)


def _broadcast_jacobians(js, new):
    # Loop over jacobians and broadcast each of them to new shape
    out = {}
    for name, jacobian in js.items():
        out[name] = jacobian.broadcast_to(new)
    return out


def _setup_dual_operation(*args, out=None, broadcast=True):
    """Prepare one or more (typically two) duals for some operation

    Given a sequence of arguments return a sequence that strips duals off the arguments,
    and a separate sequence that is just the jacobians, finally append the "out"
    quantity (if supplied).

    """
    # Get the ndarrays for the units.Quantity's or dlarrays, otherwise just
    # pass-through.  Note, this used to use np.asarray, but that did unpleasent things
    # to sparse arguments.
    arrays_ = [getattr(x, "value", x) for x in args]
    if broadcast:
        arrays_ = np.broadcast_arrays(*arrays_)
    # Put units back on after all that
    arrays_ = [
        x << orig.unit if hasattr(orig, "unit") else x for x, orig in zip(arrays_, args)
    ]
    # Now go through the jacobians
    jacobians = []
    for x, orig in zip(arrays_, args):
        if hasattr(orig, "jacobians"):
            if orig.shape != x.shape and broadcast:
                j = _broadcast_jacobians(orig.jacobians, x.shape)
            else:
                j = orig.jacobians
        else:
            j = {}
        jacobians.append(j)

    # Handle the case where an "out" is provided. I originally had some intelligence
    # about tracking whether out shares memory with any of the arguments but that got
    # complicated. In the end I've recognized that we can code up the binary operations
    # such that we don't need to worry about such cases.
    if out is not None:
        if isinstance(out, tuple):
            if len(out) != 1:
                raise NotImplementedError("Cannot support multiple outs")
            out = out[0]
        # As inefficient as this might appear, I'm pretty sure I need to blow away the
        # Jacobians in out and let the calling code recreate them from scratch.
        # Unpleasent things happen if not, as things done to out.jacobians leak back on
        # a and b's jacobians.
        if hasattr(out, "jacobians"):
            out.jacobians = {}

    return tuple(arrays_) + tuple(jacobians) + (out,)
