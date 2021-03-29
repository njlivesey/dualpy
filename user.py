import numpy as np
import astropy.units as units
import scipy.special as special
import scipy.constants as constants
import copy

from .jacobians import DenseJacobian, DiagonalJacobian, SparseJacobian
from .duals import dlarray
from .dual_helpers import _setup_dual_operation

__all__ = [
    "_seed_dense",
    "_seed_diagonal",
    "_seed_sparse",
    "compute_jacobians_numerically",
    "cumulative_trapezoid",
    "has_jacobians",
    "multi_newton_raphson",
    "seed",
    "solve_quadratic",
    "voigt_profile",
]


def seed(value, name, force=False, overwrite=False, reset=False):
    # In some senses, this is the most important routine in the package,
    # as it's probably the only one most users will knowingly invoke on a
    # regular basis.  It takes an astropy.Quantity and adds a diagonal
    # unit jacobian for it.  From that point on, anything computed from
    # the resulting dual will track Jacobians appropriately.
    """Rebturn a dual for a quantity populated with a unitary Jacobian matrix"""

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
    # Remove Jacobians from x0 and add our own
    jname = "_mnr_x"
    # Take off any input jacobians.
    x = units.Quantity(x0)
    if y is not None:
        y_ = units.Quantity(y)
    # Other starup issues
    i = 0
    finish = False
    reason = ""
    while (i < max_iter or max_iter < 0) and not finish:
        # Seed our special Jacobian
        x = seed(x, jname)
        if y is None:
            delta_y = func(x, *args, **kwargs)
        else:
            delta_y = func(x, *args, **kwargs) - y_
        if dy_tolerance is not None:
            finish = np.all(abs(delta_y) < dy_tolerance)
            reason = "dy"
        # Compute the Newton-Raphson step, drop the Jacibians we don't want to carry
        # round from iteration to interation
        j = delta_y.jacobians[jname].extract_diagonal()
        delta_x = units.Quantity(delta_y) / j
        if dx_tolerance is not None:
            finish = finish or np.all(abs(delta_x < dx_tolerance))
            reason = "dx"
        x = units.Quantity(x) - delta_x
        i += 1
    if reason == "":
        reason = "max_iter"

    # OK, we're done iterating.  Now we have to think about any Jacobians on the output
    if has_jacobians(y):
        raise NotImplementedError(
            "Not coded up the output Jacobian capability for multi_newton_raphson yet"
        )
    return x
