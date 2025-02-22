"""A bunch of dual (and unit)-friendly routines for math things I've needed

Typically wraps or replaces scipy/numpy type routines.  Typically with similar names
and/or interfaces, though not always.
"""

from typing import Optional

import mls_scf_tools.njlutil as njlutil
import numpy as np
import scipy.fft as fft
import scipy.interpolate as interpolate
import scipy.special as special
from mls_scf_tools.mls_pint import ureg

from dualpy import DenseJacobian, DiagonalJacobian, SeedJacobian, SparseJacobian
from dualpy.config import get_jacobian_specific_config
from dualpy.duals import dlarray
from dualpy.sparse_helpers import DenselyRearrangedSparseJacobian
from dualpy.user import delete_jacobians, seed, PossibleDual

from .dual_helpers import (
    dedual,
    get_magnitude,
    get_magnitude_and_unit,
    has_jacobians,
    setup_dual_operation,
    to_dimensionless,
)


def solve_quadratic(a, b, c, sign=1):
    """Solve quadratic equation ax^2+bx+c=0 returning jacobians"""
    # Use Muller's equation for stability when a=0
    a_, b_, c_, aj, bj, cj, _ = setup_dual_operation(a, b, c)
    d_ = np.sqrt(b_**2 - 4 * a_ * c_)
    x_ = -2 * c_ / (b_ + sign * d_)
    any_j = aj or bj or cj
    if any_j:
        x = dlarray(x_)
    else:
        x = x_
    if any_j:
        # Precompute some terms
        scale = -1.0 / (2 * a_ * x_ + b_)
        if aj:
            x_2 = x_**2
        else:
            x_2 = None
        for name, jacobian in aj.items():
            x.jacobians[name] = jacobian.premultiply_diagonal(x_2 * scale)
        for name, jacobian in bj.items():
            if name in x.jacobians:
                x.jacobians[name] += jacobian.premultiply_diagonal(x_ * scale)
            else:
                x.jacobians[name] = jacobian.premultiply_diagonal(x_ * scale)
        for name, jacobian in cj.items():
            if name in x.jacobians:
                x.jacobians[name] += jacobian.premultiply_diagonal(scale)
            else:
                x.jacobians[name] = jacobian.premultiply_diagonal(scale)
    return x


# Note that astropy doesn't supply this routine so, unlike sin,
# cos, etc. this leapfrongs straight to scipy.special, for now.
# That may give us problems down the road.
def wofz(z):
    """Dual and unit friendly version of the Fadeeva function"""
    z = to_dimensionless(z)
    z_ = dedual(z)
    # pylint: disable-next=no-member
    out_ = special.wofz(get_magnitude(z_))
    if not has_jacobians(z):
        return out_
    out = dlarray(out_)
    # The derivative actually comes out of the definition of the
    # Fadeeva function pretty easily
    c = 2j / np.sqrt(np.pi)
    # pylint: disable-next=protected-access
    out._chain_rule(z, c - 2 * z_ * out_)
    return out


def voigt_profile(x, sigma, gamma):
    """Dual and unit friendly version of the voight function"""
    z = (x + gamma * 1j) / (sigma * np.sqrt(2))
    w = wofz(z) * ureg.dimensionless
    return (np.real(w.magnitude) * w.units) / (sigma * np.sqrt(2 * np.pi))


def multi_newton_raphson(
    x0,
    func,
    args=None,
    kwargs=None,
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
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    # Define some prefixes we'll use to track Jacobians
    j_name_x = "_mnr_x"
    # Take off any input jacobians.
    x = delete_jacobians(x0)
    # Note if there are jacobians on the target, and take them off in any case
    y_has_jacobians = has_jacobians(y)
    if y is not None:
        y_ = delete_jacobians(y)
    # Do the same for args and kwargs
    args_ = [delete_jacobians(arg) for arg in args]
    kwargs_ = {name: delete_jacobians(kwarg) for name, kwarg in kwargs.items()}
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
        x = x.variable - delta_x
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
            x.jacobians[j_name] = jacobian.premultiply_diagonal(j_reciprocal)
    return x


def interp1d(
    x,
    y,
    kind="linear",
    axis=-1,
    # pylint: disable-next=redefined-outer-name
    copy=True,
    bounds_error=None,
    fill_value=np.nan,
    assume_sorted=False,
):
    """A dual/units.Quantity wrapper for scipy.interpoalte.interp1d"""
    if has_jacobians(x):
        raise ValueError("dualpy.interp1d cannot (yet) handle Jacobians for x-old")
    x_magnitude_dedualed, x_unit = get_magnitude_and_unit(dedual(x))
    y_magnitude_dedualed, y_unit = get_magnitude_and_unit(dedual(y))
    y_interpolator = interpolate.interp1d(
        x_magnitude_dedualed,
        y_magnitude_dedualed,
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
                x_magnitude_dedualed,
                jacobian.data,
                kind=kind,
                axis=jacobian.get_jaxis(axis),
                copy=copy,
                bounds_error=bounds_error,
                fill_value=(0, 0),  # fill_value,
                assume_sorted=assume_sorted,
            )

    # Result - interpolator function
    def result(x_new):
        """Interpolator from dualpy.interp1d"""
        x_new_magnitude_dedualed = get_magnitude(x_new.to(x_unit))
        if has_jacobians(x_new):
            raise ValueError("dualpy.interp1d cannot (yet?) handle Jacobians on x-new")
        y_new = y_interpolator(x_new_magnitude_dedualed) * y_unit
        if has_jacobians(y):
            y_new = dlarray(y_new)
            for name, j_interpolator in j_interpolators.items():
                j_original = y.jacobians[name]
                new_data = j_interpolator(x_new_magnitude_dedualed)
                y_new.jacobians[name] = DenseJacobian(
                    source=new_data,
                    template=j_original,
                    dependent_shape=y_new.shape,
                )
        return y_new

    # End of result interpolator function.
    return result


def construct_rfft_matrix(n_in: int):
    """Constructs a dense rfft matrix of a given dimensionality"""
    n_out = n_in // 2 + 1
    p, q = np.mgrid[0:n_out, 0:n_in]
    c = -2j * np.pi / n_in
    return np.exp(c * p * q)


def construct_irfft_matrix(n_in: int):
    """Constructs a dense irfft matrix of a given dimensionality"""
    n_out = 2 * (n_in - 1)
    p, q = np.mgrid[0:n_out, 0:n_in]
    c = 2 * np.pi / n_out
    irfft_matrix = 2 * np.cos(c * p * q) / n_out
    irfft_matrix[:, 0] *= 0.5
    irfft_matrix[:, -1] *= 0.5
    return irfft_matrix


def rfft(x, axis=-1, workers=None):
    """Compute the 1-D discrete Fourier Transform for real input (includes duals)"""
    if workers is None:
        workers = njlutil.get_n_workers()
    x_magnitude, x_unit = get_magnitude_and_unit(dedual(x))
    result = fft.rfft(x_magnitude, axis=axis, workers=workers) * x_unit
    if has_jacobians(x):
        # Preparet the result
        result = dlarray(result)
        # Dense Jacobian's are simply handled as their own fourier transform.  Sparse
        # ones dictate different possible approaches.
        #
        # The first uses a matrix multiply (granted a multiplication by a matrix whose
        # properties allow for the efficiencies implicit in the FFT algorithm, but may
        # not be usefully exploited when chain ruling with a sparse Jacobians.
        #
        # The second gathers the non fft'd axes together, on the (user's) presumption
        # that the structure will be the same for each row
        rfft_matrix = None
        # Now loop over the Jacobians and deal with them.
        for name, jacobian in x.jacobians.items():
            this_config = get_jacobian_specific_config(
                "sparse_jacobian_fft_strategy", name
            )
            if isinstance(jacobian, DiagonalJacobian) or isinstance(
                jacobian, SeedJacobian
            ):
                jacobian = SparseJacobian(jacobian)
            # Possibly convert sparse Jacobians to dense if the config says we should
            if isinstance(jacobian, SparseJacobian) and this_config == "dense":
                jacobian = DenseJacobian(jacobian)
            if isinstance(jacobian, DenseJacobian):
                jaxis = jacobian.get_jaxis(axis)
                jfft = fft.rfft(jacobian.data, axis=jaxis, workers=workers)
                result.jacobians[name] = DenseJacobian(
                    jfft,
                    template=jacobian,
                    dependent_shape=result.shape,
                    dependent_unit=x_unit,
                )
            elif isinstance(jacobian, SparseJacobian):
                jaxis = jacobian.get_jaxis(axis)
                if this_config == "matrix-multiply":
                    # Compute the rfft matrix if we haven't already
                    if rfft_matrix is None:
                        rfft_matrix = construct_rfft_matrix(x.shape[axis])
                    # Do it the matrix multiply way (note that the diagonal case invokes the
                    # sparse case under the hood).
                    result.jacobians[name] = jacobian.rtensordot(
                        rfft_matrix,
                        axes=[[1], [jaxis]],
                        dependent_unit=x_unit,
                    )
                elif this_config == "gather":
                    rearranged_jacobian = DenselyRearrangedSparseJacobian(
                        jacobian, promoted_axis=jaxis
                    )
                    fft_result = fft.rfft(
                        rearranged_jacobian.matrix, axis=0, workers=workers
                    )
                    result.jacobians[name] = rearranged_jacobian.undo(
                        fft_result, dependent_unit=x_unit
                    )
                else:
                    raise ValueError(
                        f"Invalid sparse_jacobian_fft_strategy for {name}: {this_config}"
                    )
            else:
                raise TypeError(
                    f"Unable to handle FFT for Jacobian of type {type(jacobian)}"
                )
    return result


def irfft(x, axis=-1, workers=None):
    """Compute 1-D discrete inverse Fourier Transform giving real result (with duals)"""
    if workers is None:
        workers = njlutil.get_n_workers()
    x_magnitude, x_unit = get_magnitude_and_unit(x)
    result = fft.irfft(x_magnitude, axis=axis, workers=workers) * x_unit
    if has_jacobians(x):
        # Preparet the result
        result = dlarray(result)
        # Dense Jacobian's are simply handled as their own fourier transform.  Sparse
        # ones dictate different possible approaches.
        #
        # The first uses a matrix multiply (granted a multiplication by a matrix whose
        # properties allow for the efficiencies implicit in the FFT algorithm, but may
        # not be usefully exploited when chain ruling with a sparse Jacobians.
        #
        # The second gathers the non fft'd axes together, on the (user's) presumption
        # that the structure will be the same for each row
        irfft_matrix = None
        for name, jacobian in x.jacobians.items():
            this_config = get_jacobian_specific_config(
                "sparse_jacobian_fft_strategy", name
            )
            if isinstance(jacobian, DiagonalJacobian) or isinstance(
                jacobian, SeedJacobian
            ):
                jacobian = SparseJacobian(jacobian)
            # Possibly convert sparse Jacobians to dense if the config says we should
            if isinstance(jacobian, SparseJacobian) and this_config == "dense":
                jacobian = DenseJacobian(jacobian)
            if isinstance(jacobian, DenseJacobian):
                jaxis = jacobian.get_jaxis(axis)
                jfft = fft.irfft(jacobian.data, axis=jaxis, workers=workers)
                result.jacobians[name] = DenseJacobian(
                    jfft,
                    template=jacobian,
                    dependent_shape=result.shape,
                    dependent_unit=x_unit,
                )
            elif isinstance(jacobian, SparseJacobian):
                jaxis = jacobian.get_jaxis(axis)
                if this_config == "matrix-multiply":
                    # Construct the irfft_matrix if we haven't already
                    irfft_matrix = construct_irfft_matrix(x.shape[axis])
                    result.jacobians[name] = jacobian.tensordot(
                        irfft_matrix,
                        axes=[[jaxis], [1]],
                        dependent_unit=x_unit,
                    )
                elif this_config == "gather":
                    rearranged_jacobian = DenselyRearrangedSparseJacobian(
                        jacobian, promoted_axis=jaxis
                    )
                    fft_result = fft.irfft(
                        rearranged_jacobian.matrix, axis=0, workers=workers
                    )
                    result.jacobians[name] = rearranged_jacobian.undo(
                        fft_result, dependent_unit=x_unit
                    )
                else:
                    raise ValueError(
                        f"Invalid sparse_jacobian_fft_strategy for {name}: {this_config}"
                    )
            else:
                raise TypeError(
                    f"Unable to handle FFT for Jacobian of type {type(jacobian)}"
                )
    return result


class CubicSplineWithJacobians:
    """This basically wraps scipy.interpolate.CubicSpline adding units/jacobians

    See arguments/documentation for scipy.interpolate.CubicSpline.  However, note the
    spline_jacobians option, which employs splines for the Jacobians as well as the
    values themselves.  The default is to use linear interpolation for the Jacobians.
    """

    def __init__(
        self,
        x,
        y,
        axis=0,
        bc_type="not-a-knot",
        extrapolate=None,
        spline_jacobians=False,
    ):
        """Setup a dual/unit aware CubicSpline interpolator"""
        x_ = dedual(x)
        y_ = dedual(y)
        x_magnitude, self.x_unit = get_magnitude_and_unit(x_)
        y_magnitude, self.y_unit = get_magnitude_and_unit(y_)
        self.x_min = np.min(x_)
        self.x_max = np.max(x_)
        self.axis = axis
        self.y_interpolator = interpolate.CubicSpline(
            x_magnitude,
            y_magnitude,
            axis=axis,
            bc_type=bc_type,
            extrapolate=extrapolate,
        )
        if not spline_jacobians and bc_type == "periodic":
            extrapolate = "periodic"
        # Deal with any input Jacobians on x
        if has_jacobians(x):
            self.jx_interpolators = dict()
            for name, jacobian in x.jacobians.items():
                if not spline_jacobians:
                    self.jx_interpolators[name] = jacobian.linear_interpolator(
                        x_magnitude,
                        axis,
                        extrapolate=extrapolate,
                    )
                else:
                    self.jx_interpolators[name] = jacobian.spline_interpolator(
                        x_magnitude,
                        axis,
                        bc_type=bc_type,
                        extrapolate=extrapolate,
                    )
        else:
            self.jx_interpolators = {}
        # Deal with any input Jaobians on y
        if has_jacobians(y):
            self.jy_interpolators = dict()
            for name, jacobian in y.jacobians.items():
                if not spline_jacobians:
                    self.jy_interpolators[name] = jacobian.linear_interpolator(
                        x_magnitude,
                        axis,
                        extrapolate=extrapolate,
                    )
                else:
                    self.jy_interpolators[name] = jacobian.spline_interpolator(
                        x_magnitude,
                        axis,
                        bc_type=bc_type,
                        extrapolate=extrapolate,
                    )
        else:
            self.jy_interpolators = {}
        # Leave a space to cache the derivative interpolators
        self.dydx_interpolator = None

    def __call__(self, x):
        """Return a dual/unit aware spline interpolation"""
        # Make sure x is in the right units and bounded, then take units off
        x_out = np.clip(x.to(self.x_unit), self.x_min, self.x_max)
        x_out_magnitude_dedualed = get_magnitude(dedual(x_out))
        # Get the interpolated value of y, make it a dual
        y = dlarray(self.y_interpolator(x_out_magnitude_dedualed) * self.y_unit)
        # Now deal with any jacobians with regard to the original y
        for name, jy_interpolator in self.jy_interpolators.items():
            y.jacobians[name] = jy_interpolator(x_out_magnitude_dedualed)
        # Now work out if we're going to need dydx, construct it if so.
        if has_jacobians(x_out) or self.jx_interpolators:
            # Construct the interpolator if we're going to need it
            if self.dydx_interpolator is None:
                self.dydx_interpolator = self.y_interpolator.derivative()
            # Invoke it for this x
            dydx = self.dydx_interpolator(x_out_magnitude_dedualed) * (
                self.y_unit / self.x_unit
            )
        else:
            dydx = None
        # Now deal with any jacobians with regard to the output x, first identify a
        # padded out shape for x_new that includes dummys for the other dimensions in y.
        if has_jacobians(x_out) or self.jx_interpolators:
            padded_x_out_shape = [1] * y.ndim
            padded_x_out_shape[self.axis] = x.size
            padded_x_out_shape = tuple(padded_x_out_shape)
        if has_jacobians(x_out):
            x_out_reshaped = x_out.reshape(padded_x_out_shape)
            y._chain_rule(x_out_reshaped, dydx, add=True)
        # Now deal with any Jacobians with regard to the input x
        for name, jx_interpolator in self.jx_interpolators.items():
            jx_interpolated = jx_interpolator(x_out_magnitude_dedualed)
            jx_interpolated = jx_interpolated.reshape(
                padded_x_out_shape, order="K", parent_flags=y.flags
            )
            jx_interpolated = jx_interpolated.broadcast_to(y.shape)
            jy_contribution = jx_interpolated.premultiply_diagonal(dydx)
            if name in y.jacobians:
                y.jacobians[name] -= jy_contribution
            else:
                y.jacobians[name] = -jy_contribution
        # Now, if the result doesn't have Jacobians make it not a dual
        if not has_jacobians(y):
            y = dedual(y)
        return y


def simpson_nonuniform(
    x: PossibleDual, f: PossibleDual, axis: Optional[int] = 0
) -> PossibleDual:
    """Dual/unit friendly implementation of Simpson rule for irregularly spaced data.

    This is adapted from the example given in Wikipedia

    Arguments
    ---------
    x : PossibleDual
        Independent variable
    f : PossibleDual
        Function values to integrate
    axis : Optional[int]
        The axis to integrate over, defaults to 0

    Returns
    -------
    integral : PossibleDual
        Result

    """

    # Define a function to make keys
    def get_nd_key(key):
        """Returns a key tuple with the input in the "axis" axis, all others full slices"""
        result = [slice(None)] * f.ndim
        result[axis] = key
        return tuple(result)

    # Set up stuff related to x
    if x.ndim != 1:
        raise ValueError("Wrong shape for x")
    n = x.shape[0] - 1  # n is the number of panels, not the nuber of points
    h = np.diff(x)
    h0, h1 = h[0:-1:2], h[1::2]
    hph, hdh, hmh = h1 + h0, h1 / h0, h1 * h0
    # Construct the keys for accessing f along the axis
    fi_key = get_nd_key(slice(1, -1, 2))
    fi_p1_key = get_nd_key(slice(2, None, 2))
    fi_m1_key = get_nd_key(slice(0, -2, 2))
    # Construct a key needed in the case where f is multi-dimensional to make things
    # broadcastable
    h_fix = [np.newaxis] * f.ndim
    h_fix[axis] = slice(None)
    h_fix = tuple(h_fix)
    h0, h1 = h0[h_fix], h1[h_fix]
    hph, hdh, hmh = hph[h_fix], hdh[h_fix], hmh[h_fix]
    # Construct the first part of the result (ony part needed if n is even)
    result = np.sum(
        (hph / 6)
        * (
            (2 - hdh) * f[fi_m1_key]
            + (hph**2 / hmh) * f[fi_key]
            + (2 - 1 / hdh) * f[fi_p1_key]
        ),
        axis=axis,
    )
    # Do corrections for the n is odd case.
    if n % 2 == 1:
        h0, h1 = h[n - 2], h[n - 1]
        # Get keys for accessing the n, n-1, and n-2 values of f
        fn_key = get_nd_key(n)
        fn_m1_key = get_nd_key(n - 1)
        fn_m2_key = get_nd_key(n - 2)
        # Tweak the result
        result += f[fn_key] * (2 * h1**2 + 3 * h0 * h1) / (6 * (h0 + h1))
        result += f[fn_m1_key] * (h1**2 + 3 * h1 * h0) / (6 * h0)
        result -= f[fn_m2_key] * h1**3 / (6 * h0 * (h0 + h1))
    return result
