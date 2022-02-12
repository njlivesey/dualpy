import numpy as np
import astropy.units as units
import scipy.special as special
import scipy.constants as constants
import scipy.interpolate as interpolate
import scipy.fft as fft
from typing import Union
import copy
import dask

from .jacobians import (
    BaseJacobian,
    DenseJacobian,
    DiagonalJacobian,
    SeedJacobian,
    SparseJacobian,
)
from .duals import dlarray
from .dual_helpers import _setup_dual_operation
from .config import get_config

__all__ = [
    "CubicSplineLinearJacobians",
    "PossibleDual",
    "delete_jacobians",
    "has_jacobians",
    "interp1d",
    "irfft",
    "multi_newton_raphson",
    "rfft",
    "seed",
    "solve_quadratic",
    "voigt_profile",
]

PossibleDual = Union[units.Quantity, dlarray]


def seed(
    value,
    name,
    force=False,
    overwrite=False,
    reset=False,
    initial_type="diagonal",
    **kwargs,
):
    # In some senses, this is the most important routine in the package,
    # as it's probably the only one most users will knowingly invoke on a
    # regular basis.  It takes an astropy.Quantity and adds a diagonal
    # unit jacobian for it.  From that point on, anything computed from
    # the resulting dual will track Jacobians appropriately.
    """Return a dual for a quantity populated, with a unitary Jacobian matrix

    Parameters
    ----------
    value : array-like (or class having _seed method)
        The quantity to have a dual seed added to it.  Note, if value has a _seed
        method, then that is invoked to do all the work instead.
    name : str
        The name for the seed
    force : bool
        If set, make it seed even if it already has other Jacobians
    overwrite : bool
        If set, overwrite an existing Jacobian already called by proposed name (required
        force to be set also).
    reset : bool
        If it's a dual already, then blow away any previous Jacobians
    initial_type : str, optional
        "diagonal" (default), "dense", or "sparse"

    Other options may be passed to the _seed method

    Result
    ------
    returns quantity as dual with named seed added

    """
    # First see if the quantity has a _seed method, and, if so, invoke that to do the
    # work.
    if hasattr(value, "_seed"):
        return value._seed(
            name,
            force=force,
            overwrite=overwrite,
            reset=reset,
            initial_type=initial_type,
        )
    # Otherwise, kwargs is illegal
    if kwargs:
        raise ValueError("No additional arguements to seed allowed for this quantity")
    if type(value) is dlarray:
        if not force and not reset:
            raise ValueError("Proposed seed is already a dual (set force or reset?)")
        if name in value.jacobians and not overwrite and not reset:
            raise ValueError(
                f"Proposed seed already has a jacobian named '{name}'"
                + " (set overwrite as well as force?)"
            )
    if not isinstance(value, dlarray):
        out = dlarray(value)
    else:
        out = value
        if reset:
            # Blow away any previous Jacobians
            out.jacobians = {}
    # Create the Jacobian as diaongal initially
    jacobian = SeedJacobian(
        np.ones(out.shape),
        dependent_unit=value.unit,
        independent_unit=value.unit,
        dependent_shape=value.shape,
        independent_shape=value.shape,
    )
    # Possibly cast it to other forms
    if initial_type == "diagonal":
        pass
    elif initial_type == "sparse":
        jacobian = SparseJacobian(jacobian)
    elif initial_type == "dense":
        jacobian = DenseJacobian(jacobian)
    else:
        raise ValueError(f"Illegal initial_type ({initial_type})")
    out.jacobians[name] = jacobian
    return out


def delete_jacobians(
    a, *names, wildcard=None, remain_dual=False, **kwargs
):
    """Remove all or selected Jacobians from a quantity, if non left de-dual

    Not that, unlike the method of the same name, this does not operate in place,
    instead it returns a copy of the input with the named jacobians removed.

    Arguments:
    ----------
    a : array_like (notably dual)
        Quantity from which Jacobians are to be removed.  If it has a _delete_jacobians
        method, then that is invoked to do this work instead.
    names : sequence[str]
        Sequence of named Jacobians to delete.  If absent (and no wildcard is
        suppled) then all the Jacobians are deleted.
    wildcard : str, optional
        A unix-style wildcard identifying Jacobians to deleted.
    remain_dual : bool=False
        If, after deleting Jacobians, none are left, then this method will demote
        the dual back to a non-dual array, unless this argument is set to True
    **kwargs : dict, optional
        Other arguments that may be passed to any _delete_jacobians method
    """
    # First see if this quantity has a _delete_jacobians method.  If so, use it.
    if hasattr(a, "_delete_jacobians"):
        return a._delete_jacobians(*names, wildcard=wildcard, **kwargs)
    # Otherwise, this is a dlarray (or quacks like one), delete the jacobians ourselves.
    if kwargs:
        raise ValueError(
            "No additional arguments allowed to delete_jacobians for this quantity"
        )
    # Check if this has a delete_jacobians method, and invoke that if so
    if hasattr(a, "delete_jacobians"):
        result = copy.copy(a)
        result.delete_jacobians(*names, wildcard=wildcard)
        # Now possibly demote back to non-dual array if merited and desired
        if not result.jacobians and not remain_dual:
            result = units.Quantity(result)
        return result
    else:
        # OK, this isn't a dual, return the input unaffected
        return a


def solve_quadratic(a, b, c, sign=1):
    """Solve quadratic equation ax^2+bx+c=0 returning jacobians"""
    # Use Muller's equation for stability when a=0
    a_, b_, c_, aj, bj, cj, out = _setup_dual_operation(a, b, c)
    d_ = np.sqrt(b_**2 - 4 * a_ * c_)
    x_ = -2 * c_ / (b_ + sign * d_)
    anyJ = aj or bj or cj
    if anyJ:
        x = dlarray(x_)
    else:
        x = x_
    if anyJ:
        # Precompute some terms
        scale = -1.0 / (2 * a_ * x_ + b_)
        if len(aj):
            x_2 = x_**2
        for name, jacobian in aj.items():
            x.jacobians[name] = jacobian.premul_diag(x_2 * scale)
        for name, jacobian in bj.items():
            if name in x.jacobians:
                x.jacobians[name] += jacobian.premul_diag(x * scale)
            else:
                x.jacobians[name] = jacobian.premul_diag(x * scale)
        for name, jacobian in cj.items():
            if name in x.jacobians:
                x.jacobians[name] += jacobian.premul_diag(scale)
            else:
                x.jacobians[name] = jacobian.premul_diag(scale)
    return x


# Note that astropy doesn't supply this routine so, unlike sin,
# cos, etc. this leapfrongs straight to scipy.special, for now.
# That may give us problems down the road.
def wofz(z):
    z_ = units.Quantity(z)
    if z_.unit != units.dimensionless_unscaled:
        raise units.UnitsError(
            "Can only apply wofz function to dimensionless quantities"
        )
    out_ = special.wofz(z_.value) << units.dimensionless_unscaled
    if isinstance(z, dlarray):
        out = dlarray(out_)
    else:
        out = out_
    # The derivative actually comes out of the definition of the
    # Fadeeva function pretty easily
    if hasattr(z, "jacobians"):
        c = 2 * complex(0, 1) / np.sqrt(constants.pi)
        out._chain_rule(z, c - 2 * z_ * out_)
    return out


def voigt_profile(x, sigma, gamma):
    i = complex(0, 1)
    z = ((x + gamma * i) / (sigma * np.sqrt(2))).to(units.dimensionless_unscaled)
    outZ = wofz(z) / (sigma * np.sqrt(2 * constants.pi))
    out = np.real(outZ)
    return out


def has_jacobians(a):
    """Return true if a is a dual with Jacobians"""
    if not hasattr(a, "jacobians"):
        return False
    return bool(a.jacobians)


def multi_newton_raphson(
    x0,
    func,
    args=[],
    kwargs={},
    y=None,
    dy_tolerance=None,
    dx_tolerance=None,
    max_iter=None,
):
    """Solve for func(x)=y using the Newton-Raphson method in a dual-aware manner

    x0: array-like
        Initial guess for x

    func: function to solve.
        First argument must be x, and must return y.  Other arguments can be suppled
        (see args and kwargs below). func(x) should be the same shape as x.

    args, kwargs:
        Optional additional arguments for func

    y: array_like, optional, default None
        Target result of func(x) for which value of x is sought.  If None, zero assumed

    dy_tolerance: float or array-like, optional
        If supplied solutions where func(x) is within dy_tolerance of y for all elements
        are acceptable.

    dx_tolerance: float or array_like, optional
        If supplied, if all values of x move by an ammount smaller than dx_tolerance,
        then the system is deemed to have converged.

    max_iter: int, optional
        Maximum number of iterations to be performed before giving up

    Returns:
    --------

    Value of x that satisfy func(x)=y within supplied tolerances.  If y has Jacobian
    information, then x is returned with corresponding Jacobian information.

    To-do:
    ------

    Make it so it returns more useful information (optionally?)

    Get the Jacobians working on the output.

    """
    # Set defaults
    if max_iter is None:
        max_iter = -1
    else:
        if max_iter < 0:
            raise ValueError("Bad value for max_iter: {max_iter}")
    # Define some prefixes we'll use to track Jacobians
    j_name_x = "_mnr_x"
    # Take off any input jacobians.
    x = units.Quantity(x0)
    # Note if there are jacobians on the target, and take them off in any case
    y_has_jacobians = has_jacobians(y)
    if y is not None:
        y_ = delete_jacobians(y)
    # Do the same for args and kwargs
    args_ = [delete_jacobians(arg) for arg in args]
    kwargs_ = {
        name: delete_jacobians(kwarg)
        for name, kwarg in kwargs.items()
    }
    # Other startup issues
    i = 0
    finish = False
    reason = ""
    while (i < max_iter or max_iter < 0) and not finish:
        # Seed our special Jacobian
        x = seed(x, j_name_x)
        if y is not None:
            delta_y = func(x, *args_, **kwargs_) - y_
        else:
            delta_y = func(x, *args_, **kwargs_)
        if dy_tolerance is not None:
            finish = np.all(abs(delta_y) < dy_tolerance)
            reason = "dy"
        # Compute the Newton-Raphson step and drop the Jacobians we don't want to carry
        # round from iteration to interation
        j = delta_y.jacobians[j_name_x].extract_diagonal()
        delta_y = delete_jacobians(delta_y)
        delta_x = delta_y / j
        if dx_tolerance is not None:
            finish = finish or np.all(abs(delta_x) < dx_tolerance)
            reason = "dx"
        x = units.Quantity(x) - delta_x
        i += 1
    if reason == "":
        reason = "max_iter"

    # OK, we're done iterating.  Now we have to think about any Jacobians on the output.
    # This is acutally both more and less complicated than it seems.  We're after dx/dt
    # where t is a quantity we're differentiating by, and all derivatives here are
    # partial.  The complexity arrives that there may be contributions to df/dt from the
    # other arguments to f, or indeed to something buried in f itself.  I spent a lot of
    # time thinking I'd have to compute these terms one by one (and unbury them, which
    # was annoying).  However, we can work out what the sum of these is by calling the
    # function one more time, at our solution x, but without any Jacobian tracking for t
    # in the solution x itself, but intrinsically keeping all the Jacobian tracking in
    # the other arguments and anything related to t buried in f itself.  This gives us
    # the sum of all those other terms.  We can then subtract that from any dy/dt from
    # the target, and use our knowledge of df/dx to get dx/dt. So...

    # If args and/or kwargs have Jacobians, we need to do one more run of the
    # function, this time with Jacobians re-enabled to get the final Jacobian terms.
    # As before, we'll keep our extra jacobian to keep track of df/dx directly.
    x_solution = seed(x, j_name_x)
    y_solution = func(x_solution, *args, **kwargs)
    # If the target has Jacobians, or this last call has added more Jacobians, then we
    # need to provide Jacobians on the output
    if has_jacobians(y) or len(y_solution.jacobians) > 1:
        # Now, as x_solution only has Jacobians with respect to x, this means that any
        # Jacobians that emerge with respect to anything else must have come from other
        # (possibly hidden) arguments to the function. That is they are the sum of
        # df/db*db/dy, where the derivatives here are partial, so we'll need to factor
        # them in.
        #
        # We're after dx/dt, get it as [dy/dt-(all df/db*db/dt)]/(dy/dx), start with
        # dy/dt terms.
        accumulator = {}
        if y_has_jacobians:
            for name, jacobian in y.jacobians.items():
                accumulator[name] = jacobian
        if has_jacobians(y_solution):
            for name, jacobian in y_solution.jacobians.items():
                # Skip over the Jacobian we put in by hand.
                if name != j_name_x:
                    if name in accumulator:
                        accumulator[name] -= jacobian
                    else:
                        accumulator[name] = -jacobian
        # OK, we're finnally ready to add Jacobians to the result
        x = dlarray(x)
        j_reciprocal = 1.0 / y_solution.jacobians[j_name_x].extract_diagonal()
        for j_name, jacobian in accumulator.items():
            x.jacobians[j_name] = jacobian.premul_diag(j_reciprocal)
    return x


def interp1d(
    x,
    y,
    kind="linear",
    axis=-1,
    copy=True,
    bounds_error=None,
    fill_value=np.nan,
    assume_sorted=False,
):
    """A dual/units.Quantity wrapper for scipy.interpoalte.interp1d"""
    if has_jacobians(x):
        raise ValueError("dualpy.interp1d cannot (yet) handle Jacobians for x-old")
    y_interpolator = interpolate.interp1d(
        x,
        y,
        kind=kind,
        axis=axis,
        copy=copy,
        bounds_error=bounds_error,
        fill_value=fill_value,
        assume_sorted=assume_sorted,
    )
    if has_jacobians(y):
        j_interpolators = dict()
        for name, jacobian in y.jacobians.items():
            jacobian = DenseJacobian(jacobian)
            j_interpolators[name] = interpolate.interp1d(
                x,
                jacobian.data,
                kind=kind,
                axis=jacobian._get_jaxis(axis),
                copy=copy,
                bounds_error=bounds_error,
                fill_value=(0, 0),  # fill_value,
                assume_sorted=assume_sorted,
            )

    # Result - interpolator function
    def result(x_new):
        """Interpolator from dualpy.interp1d"""
        x_new_np = x_new.to(x.unit).value
        if has_jacobians(x_new):
            raise ValueError("dualpy.interp1d cannot (yet?) handle Jacobians on x-new")
        y_new = y_interpolator(x_new_np) << y.unit
        if has_jacobians(y):
            y_new = dlarray(y_new)
            for name, j_interpolator in j_interpolators.items():
                j_original = y.jacobians[name]
                new_data = j_interpolator(x_new_np)
                y_new.jacobians[name] = DenseJacobian(
                    data=new_data,
                    template=j_original,
                    dependent_shape=y_new.shape,
                )
        return y_new

    # End of result interpolator function.
    return result


def rfft(x, axis=-1):
    """Compute the 1-D discrete Fourier Transform for real input (includes duals)"""
    result = fft.rfft(np.array(x), axis=axis) << x.unit
    if has_jacobians(x):
        # Preparet the result
        result = dlarray(result)
        # Dense Jacobian's are simply handled as their own fourier transform.  Dense
        # ones dictate a different (hopefully more efficient approach that recognizes
        # that a Fourier transform is simply a matrix multiply (granted a multiplication
        # by a matrix whose properties allow for the efficiencies implicit in the FFT
        # algorithm, but may not be usefully exploited when chain ruling with a sparse
        # Jacobians.  So first see if any of the subsequent matrices are sparse (or
        # diagonal).
        any_non_dense = any(
            [
                not isinstance(jacobian, DenseJacobian)
                for jacobian in x.jacobians.values()
            ]
        )
        if any_non_dense:
            # Compute the matrix that is d<rfft>/dx
            n_in = x.shape[axis]
            n_out = n_in / 2 + 1 if n_in % 2 == 0 else (n_in + 1) / 2
            p, q = np.mgrid[0:n_out, 0:n_in]
            c = -2j * np.pi / n_in
            D = np.exp(c * p * q)
        # Now loop over the Jacobians and deal with them.
        use_dask = "rfft" in get_config().dask
        if use_dask:
            rfft_routine = dask.delayed(fft.rfft)
            make_dense_jacobian = dask.delayed(DenseJacobian)
        else:
            rfft_routine = fft.rfft
            make_dense_jacobian = DenseJacobian
        for name, jacobian in x.jacobians.items():
            if isinstance(jacobian, DenseJacobian):
                jaxis = jacobian._get_jaxis(axis)
                jfft = rfft_routine(jacobian.data, axis=jaxis)
                result.jacobians[name] = make_dense_jacobian(
                    jfft,
                    template=jacobian,
                    dependent_shape=result.shape,
                )
            else:
                if isinstance(jacobian, DiagonalJacobian):
                    jacobian = SparseJacobian(jacobian)
                jaxis = jacobian._get_jaxis(axis)
                # Do it the matrix multiply way (note that the diagonal case invokes the
                # sparse case under the hood).
                if use_dask:
                    result.jacobians[name] = dask.delayed(jacobian.rtensordot)(
                        D,
                        axes=[[1], [jaxis]],
                        dependent_unit=result.unit,
                    )
                else:
                    result.jacobians[name] = jacobian.rtensordot(
                        D,
                        axes=[[1], [jaxis]],
                        dependent_unit=result.unit,
                    )
        if use_dask:
            result.jacobians = dask.compute(result.jacobians)[0]
    return result


def irfft(x, axis=-1):
    """Compute 1-D discrete inverse Fourier Transform giving real result (with duals)"""
    result = fft.irfft(np.array(x), axis=axis) << x.unit
    if has_jacobians(x):
        # Preparet the result
        result = dlarray(result)
        # Dense Jacobian's are simply handled as their own fourier transform.  Dense
        # ones dictate a different (hopefully more efficient approach that recognizes
        # that a Fourier transform is simply a matrix multiply (granted a multiplication
        # by a matrix whose properties allow for the efficiencies implicit in the FFT
        # algorithm, but may not be usefully exploited when chain ruling with a sparse
        # Jacobians.  So first see if any of the subsequent matrices are sparse (or
        # diagonal).
        any_non_dense = any(
            [
                not isinstance(jacobian, DenseJacobian)
                for jacobian in x.jacobians.values()
            ]
        )
        if any_non_dense:
            # Compute the matrix that is d<rfft>/dx
            n_in = x.shape[axis]
            n_out = 2 * (n_in - 1)
            p, q = np.mgrid[0:n_out, 0:n_in]
            c = 2 * np.pi / n_out
            D = 2 * np.cos(c * p * q) / n_out
            D[:, 0] *= 0.5
            D[:, -1] *= 0.5
        # Now loop over the Jacobians and deal with them.
        use_dask = "irfft" in get_config().dask
        if use_dask:
            irfft_routine = dask.delayed(fft.irfft)
            make_dense_jacobian = dask.delayed(DenseJacobian)
        else:
            irfft_routine = fft.irfft
            make_dense_jacobian = DenseJacobian
        for name, jacobian in x.jacobians.items():
            if isinstance(jacobian, DenseJacobian):
                jaxis = jacobian._get_jaxis(axis)
                jfft = irfft_routine(jacobian.data, axis=jaxis)
                result.jacobians[name] = make_dense_jacobian(
                    jfft,
                    template=jacobian,
                    dependent_shape=result.shape,
                )
            else:
                if isinstance(jacobian, DiagonalJacobian):
                    jacobian = SparseJacobian(jacobian)
                jaxis = jacobian._get_jaxis(axis)
                # Do it the matrix multiply way (note that the diagonal case invokes the
                # sparse case under the hood).
                if use_dask:
                    result.jacobians[name] = dask.delayed(jacobian.tensordot)(
                        D,
                        axes=[[jaxis], [1]],
                        dependent_unit=result.unit,
                    )
                else:
                    result.jacobians[name] = jacobian.tensordot(
                        D,
                        axes=[[jaxis], [1]],
                        dependent_unit=result.unit,
                    )
        if use_dask:
            result.jacobians = dask.compute(result.jacobians)[0]
    return result


class CubicSplineLinearJacobians:
    """This basically wraps scipy.interpolate.CubicSpline"""

    def __init__(self, x, y, axis=0, bc_type="not-a-knot", extrapolate=None):
        """Setup a dual/unit aware CubicSpline interpolator"""
        if has_jacobians(x):
            raise ValueError("Cannot (yet) handle Jacobians for input x")
        self.x_unit = x.unit
        self.y_unit = y.unit
        self.x_min = np.min(units.Quantity(x))
        self.x_max = np.max(units.Quantity(x))
        self.axis = axis
        self.y_interpolator = interpolate.CubicSpline(
            x.value,
            y.value,
            axis=axis,
            bc_type=bc_type,
            extrapolate=extrapolate,
        )
        if has_jacobians(y):
            self.j_interpolators = dict()
            if "CubicSplineLinearJacobians" in get_config().dask:
                for name, jacobian in y.jacobians.items():
                    self.j_interpolators[name] = dask.delayed(
                        jacobian.linear_interpolator
                    )(x.value, axis)
                self.j_interpolators = dask.compute(self.j_interpolators)[0]
            else:
                for name, jacobian in y.jacobians.items():
                    self.j_interpolators[name] = jacobian.linear_interpolator(
                        x.value, axis
                    )
        else:
            self.j_interpolators = {}
        # Leave a space to cache the derivative interpolators
        self.dydx_interpolator = None

    def __call__(self, x):
        """Return a dual/unit aware spline interpolation"""
        # Make sure x is in the right units and bounded, then take units off
        x_fixed = np.clip(x.to(self.x_unit), self.x_min, self.x_max)
        # Get the interpolated value of y, make it a dual
        y = dlarray(self.y_interpolator(x_fixed.value) << self.y_unit)
        # Now deal with any jacobians with regard to the original y
        if "CubicSplineLinearJacobians" in get_config().dask:
            for name, j_interpolator in self.j_interpolators.items():
                y.jacobians[name] = dask.delayed(j_interpolator)(x_fixed.value)
            y.jacobians = dask.compute(y.jacobians)[0]
        else:
            for name, j_interpolator in self.j_interpolators.items():
                y.jacobians[name] = j_interpolator(x_fixed.value)
        # Now deal with any jacobians with regard to x
        if has_jacobians(x):
            # Get the dydx_interolator if it's not already been generated
            if self.dydx_interpolator is None:
                self.dydx_interpolator = self.y_interpolator.derivative()
            dydx = self.dydx_interpolator(x_fixed) << (self.y_unit / self.x_unit)
            # Now multiply all the dx/dt terms by dy/dx to get dy/dt
            x_new_shape = [1] * y.ndim
            x_new_shape[self.axis] = x.size
            x_fixed = x_fixed.reshape(tuple(x_new_shape))
            y._chain_rule(x_fixed, dydx)
        # Now, if the result doesn't have Jacobians make it not a dual
        if not has_jacobians(y):
            y = units.Quantity(y)
        return y
