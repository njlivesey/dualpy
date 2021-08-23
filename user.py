import numpy as np
import astropy.units as units
import scipy.special as special
import scipy.constants as constants
import scipy.interpolate as interpolate
import scipy.fft as fft
from typing import Union

from .jacobians import (
    BaseJacobian,
    DenseJacobian,
    DiagonalJacobian,
    SparseJacobian,
    matrix_multiply_jacobians,
)
from .duals import dlarray
from .dual_helpers import _setup_dual_operation

__all__ = [
    "CubicSpline",
    "PossibleDual",
    "_seed_dense",
    "_seed_diagonal",
    "_seed_sparse",
    "compute_jacobians_numerically",
    "cumulative_trapezoid",
    "eliminate_jacobians",
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


def eliminate_jacobians(a):
    """Demote argument from dual back to astropy.Quantity if needed"""
    if isinstance(a, dlarray):
        return units.Quantity(a)
    else:
        return a


def seed(value, name, force=False, overwrite=False, reset=False):
    # In some senses, this is the most important routine in the package,
    # as it's probably the only one most users will knowingly invoke on a
    # regular basis.  It takes an astropy.Quantity and adds a diagonal
    # unit jacobian for it.  From that point on, anything computed from
    # the resulting dual will track Jacobians appropriately.
    """Return a dual for a quantity populated with a unitary Jacobian matrix"""

    if type(value) is dlarray:
        if not force:
            raise ValueError("Proposed seed is already a dual (set force?)")
        if name in value.jacobians and not overwrite:
            raise ValueError(
                f"Proposed seed already has a jacobian named '{name}'"
                + " (set overwrite as well as force?)"
            )
    if type(value) is not dlarray or reset:
        out = dlarray(value)
    else:
        out = value
    jacobian = DiagonalJacobian(
        np.ones(out.shape),
        dependent_unit=value.unit,
        independent_unit=value.unit,
        dependent_shape=value.shape,
        independent_shape=value.shape,
    )
    out.jacobians[name] = jacobian
    return out


# These two are used for testing purposes, to explore functionality
# perhaps not otherwise ventured into.
def _seed_dense(value, name, **kwargs):
    result = seed(value, name, **kwargs)
    result.jacobians[name] = DenseJacobian(result.jacobians[name])
    return result


def _seed_sparse(value, name, **kwargs):
    result = seed(value, name, **kwargs)
    result.jacobians[name] = SparseJacobian(result.jacobians[name])
    return result


# And this one for completeness
def _seed_diagonal(value, name, **kwargs):
    return seed(value, name, **kwargs)


def compute_jacobians_numerically(func, args=None, kwargs=None, plain_func=None):
    # Take a function and set of arguments, run the function once with
    # analytical Jacobians, then perturb each seeded element in turn
    # to compute equivalent numerical Jacobians.  This is used for
    # testing the analytical Jacobian calcualtions.  If "func" cannot
    # be called for non-duals (e.g., voigt_profile exists for ndarray
    # and dlarray, but not units.Quantity) then the optional
    # plain_func argument provides a non-dual compatible routine
    # (presumably a wrapper that promotes one or more arguments to
    # dual to then invoke func on).
    #
    # First compute the unperturbed result
    if plain_func is None:
        plain_func = func
    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()
    result_a = func(*args, **kwargs)
    result0 = units.Quantity(result_a)
    result_n = dlarray(result0)
    #
    # Now combine the args and kwargs into one set of iterable items
    # and names (which are none for the args)
    all_args = []
    all_arg_names = []
    for a in args:
        all_args.append(a)
        all_arg_names.append(None)
    for n, a in kwargs.items():
        all_args.append(a)
        all_arg_names.append(n)
    # Now create a version of these arguments where all duals have
    # been demoted to regular quantities
    all_args_no_duals = []
    seed_names = []
    for a in all_args:
        if isinstance(a, dlarray):
            # The only duals allowed are seeds, let's check thats the
            # name
            if len(a.jacobians) == 0:
                continue
            if len(a.jacobians) != 1:
                raise ValueError("Inputs can only have one Jacobian")
            name = list(a.jacobians.keys())[0]
            j = a.jacobians[name]
            # Check that the Jacobian is square, if so, consider that
            # "good enough". Could check for digaonal but that means
            # we can't use this whole routine to do some of the
            # testing we'd like to do.
            if j.dependent_shape != j.independent_shape:
                raise ValueError("Jacobian is not square")
            all_args_no_duals.append(units.Quantity(a))
            seed_names.append(name)
        else:
            all_args_no_duals.append(a)
            seed_names.append(None)
    # Now take those dual-less arguments and redistribute them into
    # args and kwargs again.
    args_no_duals = []
    kwargs_no_duals = {}
    for n, a in zip(all_arg_names, all_args_no_duals):
        if n is None:
            args_no_duals.append(a)
        else:
            kwargs_no_duals[n] = a
    # Now define our perturbations
    finfo = np.finfo(np.float32)
    ptb_f = np.sqrt(finfo.eps)
    ptb_a = ptb_f
    # Now, iterate over all our arguments
    for a, a_nd in zip(all_args, all_args_no_duals):
        # For the seeds we'll go through them one by one and perturb them
        if isinstance(a, dlarray):
            name = list(a.jacobians.keys())[0]
            template = a.jacobians[name]
            # Create the 2D matrix that will contain the numerical Jacobians
            jacobian = np.ndarray((result0.size, a.size))
            # There may be a more pythonic way to do this, but for now this works.
            for i in np.arange(a.size):
                # Perturb one element, call the function, put the
                # original value back and note the results.
                a_nd_flat = a_nd.reshape(-1)
                oldV = a_nd_flat[i]
                dx = np.maximum(np.abs(oldV * ptb_f), (ptb_a << oldV.unit))
                a_nd_flat[i] += dx
                resultP = plain_func(*args_no_duals, **kwargs_no_duals)
                a_nd_flat[i] = oldV
                dResult = resultP - result0
                jacobian[:, i] = (dResult / dx).value.ravel()
            # Store the Jacobian
            target_shape = result0.shape + template.independent_shape
            jacobian = np.reshape(jacobian, target_shape)
            jacobian = DenseJacobian(
                data=jacobian,
                template=template,
                dependent_shape=result0.shape,
                dependent_unit=dResult.unit,
                independent_unit=dx.unit,
            )
            result_n.jacobians[name] = jacobian
    # Now we're done, I think
    return result_a, result_n


def solve_quadratic(a, b, c, sign=1):
    """Solve quadratic equation ax^2+bx+c=0 returning jacobians"""
    # Use Muller's equation for stability when a=0
    a_, b_, c_, aj, bj, cj, out = _setup_dual_operation(a, b, c)
    d_ = np.sqrt(b_ ** 2 - 4 * a_ * c_)
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
            x_2 = x_ ** 2
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


def _tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    """
    Cumulatively integrate y(x) using the composite trapezoidal rule.
    Adapted from scipy routine of the same name, but for duals/astropy quantities

    Parameters
    ----------
    y : array_like
        Values to integrate.
    x : array_like, optional
        The coordinate to integrate along. If None (default), use spacing `dx`
        between consecutive elements in `y`.
    dx : float, optional
        Spacing between elements of `y`. Only used if `x` is None.
    axis : int, optional
        Specifies the axis to cumulate. Default is -1 (last axis).
    initial : scalar, optional
        If given, insert this value at the beginning of the returned result.
        Typically this value should be 0. Default is None, which means no
        value at ``x[0]`` is returned and `res` has one element less than `y`
        along the axis of integration.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.

    See Also
    --------
    numpy.cumsum, numpy.cumprod
    quad: adaptive quadrature using QUADPACK
    romberg: adaptive Romberg quadrature
    quadrature: adaptive Gaussian quadrature
    fixed_quad: fixed-order Gaussian quadrature
    dblquad: double integrals
    tplquad: triple integrals
    romb: integrators for sampled data
    ode: ODE integrators
    odeint: ODE integrators

    Examples
    --------
    >>> from scipy import integrate
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-2, 2, num=20)
    >>> y = x
    >>> y_int = integrate.cumulative_trapezoid(y, x, initial=0)
    >>> plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')
    >>> plt.show()

    """
    if x is None:
        d = dx
    else:
        if x.ndim == 1:
            d = np.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the " "same as y.")
        else:
            d = np.diff(x, axis=axis)

        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError(
                "If given, length of x along axis must be the " "same as y."
            )

    nd = len(y.shape)
    slice1 = _tupleset((slice(None),) * nd, axis, slice(1, None))
    slice2 = _tupleset((slice(None),) * nd, axis, slice(None, -1))
    res = np.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)
    if initial is not None:
        if not np.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")

        shape = list(res.shape)
        shape[axis] = 1
        res = np.concatenate([np.full(shape, initial, dtype=res.dtype), res], axis=axis)

    return res


def has_jacobians(a):
    """Return true if a is a dual with Jacobians"""
    if not isinstance(a, dlarray):
        return False
    return a.hasJ()


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
        y_ = eliminate_jacobians(y)
    # Do the same for args and kwargs
    args_have_jacobians = False
    args_ = []
    for arg in args:
        if has_jacobians(arg):
            args_have_jacobians = True
            arg = eliminate_jacobians(arg)
        args_.append(arg)
    kwargs_have_jacobians = False
    kwargs_ = {}
    for name, kwarg in kwargs.items():
        if has_jacobians(kwarg):
            kwargs_have_jacobians = True
            kwarg = eliminate_jacobians(kwarg)
        kwargs_[name] = kwarg
    # Other starup issues
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
        delta_y = eliminate_jacobians(delta_y)
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
    # function on more time, at our solution x, but without any Jacobian tracking for t
    # in the solution x itself, but intrinsically keeping all the Jacobian tracking in
    # the other arguments and anything related to t buried in f itself.  This gives us
    # the sum of all those other terms.  We can then subtract that from any dy/dt from
    # the target, and use our knowledge of df/dx to get dx/dt.

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
        result = dlarray(result)
        for name, jacobian in x.jacobians.items():
            jacobian = DenseJacobian(jacobian)
            jaxis = jacobian._get_jaxis(axis)
            jfft = fft.rfft(jacobian.data, axis=jaxis)
            result.jacobians[name] = DenseJacobian(
                jfft, template=jacobian, dependent_shape=result.shape
            )
    return result


def irfft(x, axis=-1):
    """Compute the 1-D discrete inverse Fourier Transform for real input (includes duals)"""
    result = fft.irfft(np.array(x), axis=axis) << x.unit
    if has_jacobians(x):
        result = dlarray(result)
        for name, jacobian in x.jacobians.items():
            jacobian = DenseJacobian(jacobian)
            jaxis = jacobian._get_jaxis(axis)
            jfft = fft.irfft(jacobian.data, axis=jaxis)
            result.jacobians[name] = DenseJacobian(
                jfft, template=jacobian, dependent_shape=result.shape
            )
    return result


class CubicSpline:
    """This basically wraps scipy.interpolate.CubicSpline"""

    def __init__(self, x, y, axis=0, bc_type="not-a-knot", extrapolate=None):
        """Setup a dual/unit aware CubicSpline interpolator"""
        if has_jacobians(x):
            raise ValueError("Cannot (yet) handle Jacobians for input x")
        self.x_unit = x.unit
        self.y_unit = y.unit
        self.x_min = np.min(x)
        self.x_max = np.max(x)
        self.y_interpolator = interpolate.CubicSpline(
            x.value,
            y.value,
            axis=axis,
            bc_type=bc_type,
            extrapolate=extrapolate,
        )
        if has_jacobians(y):
            self.j_interpolators = dict()
            self.j_templates = dict()
            for name, jacobian in y.jacobians.items():
                # Not going to try anything silly in terms of non-dense jacobians
                jacobian = DenseJacobian(jacobian)
                self.j_interpolators[name] = interpolate.CubicSpline(
                    x.value,
                    jacobian.data,
                    axis=jacobian._get_jaxis(axis),
                    bc_type=bc_type,
                    extrapolate=extrapolate,
                )
                self.j_templates[name] = BaseJacobian(template=jacobian)
        else:
            self.j_interpolators = {}
        # Leave a space to cache the derivative interpolators
        self.dydx_interpolator = None

    def __call__(self, x):
        """Return a dual/unit aware spline interpolation"""
        # Make sure x is in the right units and bounded, then take units off
        x_fixed = np.clip(x.to(self.x_unit), self.x_min, self.x_max).value
        # Get the intpolated value of y, make it a dual
        y = dlarray(self.y_interpolator(x_fixed) << self.y_unit)
        # Now deal with any jacobians with regard to the original y
        for name, j_interpolator in self.j_interpolators.items():
            y.jacobians[name] = DenseJacobian(
                template=self.j_templates[name],
                dependent_shape=y.shape,
                data=j_interpolator(x_fixed),
            )
        # Now deal with any jacobians with regard to x
        if has_jacobians(x):
            # Get the dydx_interolator if it's not already been generated
            if self.dydx_interpolator is None:
                self.dydx_interpolator = self.y_interpolator.derivative()
            dydx = self.dydx_interpolator(x_fixed)
            # Now multiply all the dx/dt terms by dy/dx to get dy/dt
            y._chain_rule(x, dydx)
        # Now, if the result doesn't have Jacobians make it not a dual
        if not has_jacobians(y):
            y = units.Quantity(y)
        return y
