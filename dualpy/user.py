import copy
from typing import Union

import astropy.units as units
import mls_scf_tools.njlutil as njlutil
import numpy as np
import pint

# pylint: disable-next=no-name-in-module
import scipy.fft as fft
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.special as special
from mls_scf_tools.mls_pint import ureg
from numpy.typing import NDArray

from .dual_helpers import (
    dedual,
    get_magnitude,
    get_magnitude_and_unit,
    has_jacobians,
    setup_dual_operation,
    to_dimensionless,
)
from .duals import dlarray
from .jacobians import DenseJacobian, DiagonalJacobian, SeedJacobian, SparseJacobian
from .config import get_jacobian_specific_config

from .sparse_helpers import DenselyRearrangedSparseJacobian

__all__ = [
    "CubicSplineWithJacobians",
    "DroppedJacobianWarning",
    "PossibleDual",
    "delete_jacobians",
    "get_jacobians_for_function_inverse",
    "interp1d",
    "irfft",
    "multi_newton_raphson",
    "rfft",
    "seed",
    "simpson",
    "solve_quadratic",
    "voigt_profile",
    "wofz",
]

PossibleDual = Union[units.Quantity, pint.Quantity, NDArray, dlarray]


class DroppedJacobianWarning(Warning):
    pass


def seed(
    value,
    name,
    force=False,
    overwrite=False,
    reset=False,
    initial_type="seed",
    **kwargs,
):
    # In some senses, this is the most important routine in the package, as it's
    # probably the only one most users will knowingly invoke on a regular basis.  It
    # takes an array-like quantity unit jacobian for it.  From that point on, anything
    # computed from the resulting dual will track Jacobians appropriately.
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
    if hasattr(value, "_dual_seed_"):
        # pylint: disable-next=protected-access
        return value._dual_seed_(
            name,
            force=force,
            overwrite=overwrite,
            reset=reset,
            initial_type=initial_type,
        )
    # Otherwise, kwargs is illegal
    if kwargs:
        raise ValueError("No additional arguements to seed allowed for this quantity")
    # Do some error checking in the case where it's already a dual
    if isinstance(value, dlarray):
        if not force and not reset:
            raise ValueError("Proposed seed is already a dual (set force or reset?)")
        if name in value.jacobians and not overwrite and not reset:
            raise ValueError(
                f"Proposed seed already has a jacobian named '{name}'"
                + " (set overwrite as well as force?)"
            )
    # If we're good to go, then act accordingly
    if not isinstance(value, dlarray):
        out = dlarray(value)
    else:
        out = value
        if reset:
            # Blow away any previous Jacobians
            out.jacobians = {}
    # Create the Jacobian as diaongal initially
    try:
        out_shape = out.shape
    except AttributeError:
        out_shape = tuple()
    # Work out the dtype
    try:
        dtype = value.dtype
    except AttributeError:
        dtype = None
    jacobian = SeedJacobian(
        source=np.ones(out_shape, dtype=dtype),
        dependent_unit=out._dependent_unit,
        independent_unit=out._dependent_unit,
        dependent_shape=out_shape,
        independent_shape=out_shape,
    )
    # Possibly cast it to other forms
    if initial_type == "seed":
        pass
    elif initial_type == "diagonal":
        jacobian = DiagonalJacobian(jacobian)
    elif initial_type == "sparse":
        jacobian = SparseJacobian(jacobian)
    elif initial_type == "dense":
        jacobian = DenseJacobian(jacobian)
    else:
        raise ValueError(f"Illegal initial_type ({initial_type})")
    out.jacobians[name] = jacobian
    return out


def delete_jacobians(a, *names, wildcard=None, remain_dual=False, **kwargs):
    """Remove all or selected Jacobians from a quantity, if non left de-dual

    Not that, unlike the method of the same name, this does not operate in place,
    instead it returns a copy of the input with the named jacobians removed.

    Arguments:
    ----------
    a : array_like (notably dual)
        Quantity from which Jacobians are to be removed.  If it has a
        _dual_delete_jacobians_ method, then that is invoked to do this work instead.
    names : sequence[str]
        Sequence of named Jacobians to delete.  If absent (and no wildcard is suppled)
        then all the Jacobians are deleted.
    wildcard : str, optional
        A unix-style wildcard identifying Jacobians to deleted.
    remain_dual : bool=False
        If, after deleting Jacobians, none are left, then this method will demote the
        dual back to a non-dual array, unless this argument is set to True
    **kwargs : dict, optional
        Other arguments that may be passed to any _dual_delete_jacobians_ method

    Returns:
    --------

    result : various types possible
        Input with named Jacobians deleted.
    """
    # First see if this quantity has a __delete_jacobians__ method.  If so, use it.
    if hasattr(a, "_dual_delete_jacobians_"):
        return a._dual_delete_jacobians_(
            *names, wildcard=wildcard, remain_dual=remain_dual, **kwargs
        )
    # Otherwise, this is a dlarray (or quacks like one), delete the jacobians ourselves.
    if kwargs:
        raise ValueError(
            "No additional arguments allowed to delete_jacobians for this quantity"
        )
    # Check if this has a delete_jacobians method, and invoke that if so
    if hasattr(a, "delete_jacobians"):
        result = copy.copy(a)
        result.jacobians = copy.copy(a.jacobians)
        result.delete_jacobians(*names, wildcard=wildcard)
        # Now possibly demote back to non-dual array if merited and desired
        if not result.jacobians and not remain_dual:
            result = result.variable
        return result
    else:
        # OK, this isn't a dual, return the input unaffected
        return a


def solve_quadratic(a, b, c, sign=1):
    """Solve quadratic equation ax^2+bx+c=0 returning jacobians"""
    # Use Muller's equation for stability when a=0
    a_, b_, c_, aj, bj, cj, out = setup_dual_operation(a, b, c)
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
    out._chain_rule(z, c - 2 * z_ * out_)
    return out


def voigt_profile(x, sigma, gamma):
    z = (x + gamma * 1j) / (sigma * np.sqrt(2))
    w = wofz(z) * ureg.dimensionless
    return (np.real(w.magnitude) * w.units) / (sigma * np.sqrt(2 * np.pi))


def get_jacobians_for_function_inverse(
    y_target, y_solution, dx_key: str, ignored_jkeys: list[str] = []
) -> dict:
    """Compute Jacobians corresponding to the inverse of a function

    This is a helper function for inverting other functions.  Assume the calling code
    has arrived at the solution to y=f(x), for a given y (in other words a value of
    x_solution that satisfies [to the users' satisfaction] y_target = f(x_solution)).
    Given y_target, and y_solution(=f(x_solution)), this function provides a Jacobian
    dictionary corresponding to d(x_solution)/d(y_target).

    The complication arrives because additional Jacobians on f(x) result may emerge
    through supplied additonal arguments to function (positional or keyword), or simply
    because of the instrinsic nature of the function itself.

    This is not as complicated as it might sound at first.  The algorithm employed is
    described in the comments.

    Paramters:
    ----------

    y_target: array-like (including dual)
        The result of f(x) that the calling code found an x_solution such that
        f(x_solution) = y_target (or ~=, depending on the users' preferences).

    y_solution: array-like (including dual)
        The result of f(x_solution) that includes Jacobians not only for the input
        x_solution (designated by the key dx_key), but also any Jacobians arising from
        other arguments to f, or just intrinsically bt f's very nature.

    dx_key: str
         The key in the y.jacobians that corresponds to (partial) dy/dx

    ignored_jkeys: list[str]
         Other jacobian keys that should be ignored

    Note that the Jacobian pointed to by x_key is intended to be a true partial
    derivative purely for the x term.  In other words, it must not account for any
    dependence of x on anything else.

    """

    # This is acutally both more and less complicated than it seems.  We're after
    # dx_solution/dt where t is a quantity we're differentiating by, and ALL DERIVATIVES
    # IN THESE COMMENTS ARE PARTIAL.  The complexity arrives from the fact that there
    # may be contributions to df/dt from the other arguments to f, or indeed to
    # something buried in f itself.  I spent a lot of time thinking I'd have to compute
    # these terms one by one (and unbury them, which was annoying).  However, we can
    # work out what the sum of these is by calling the function at our solution x, but
    # without any Jacobian tracking for t in the solution x itself, but intrinsically
    # keeping all the Jacobian tracking in the other arguments and anything related to t
    # buried in f itself.  This gives us the sum of all those other terms.  We can then
    # subtract that from any dy/dt from the target, and use our knowledge of df/dx to
    # get dx/dt. So...
    #
    x_solution_jacobians = {}
    # Now, as x_solution only has Jacobians with respect to x, this means that any
    # Jacobians that emerge with respect to anything else must have come from other
    # (possibly hidden) arguments to the function. That is they are the sum of
    # df/db*db/dy, where the derivatives here are partial, so we'll need to factor
    # them in.
    #
    # We're after dx/dt, get it as [dy_target/dt-(all df/db*db/dt)]/(dy/dx), start
    # with dy_target/dt terms.
    accumulator = {}
    if has_jacobians(y_target):
        for name, jacobian in y_target.jacobians.items():
            accumulator[name] = jacobian
    # Now subtract all the df/db-related terms (skip the one that is the specific
    # dy_solution/dx term)
    if has_jacobians(y_solution):
        for name, jacobian in y_solution.jacobians.items():
            # Work out whether this is the dy/dt term or one of the other ones.
            if name != dx_key and name not in ignored_jkeys:
                if name in accumulator:
                    accumulator[name] -= jacobian
                else:
                    accumulator[name] = -jacobian
    # OK, now we can compute the Jacobians for the solution
    j_reciprocal = 1.0 / y_solution.jacobians[dx_key].extract_diagonal()
    for j_name, jacobian in accumulator.items():
        x_solution_jacobians[j_name] = jacobian.premultiply_diagonal(j_reciprocal)
    return x_solution_jacobians


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


def simpson(y, x=None, dx=1.0, axis=-1, even="avg"):
    """Integrate y(x) using samples along the given axis and the composite
    Simpson's rule. If x is None, spacing of dx is assumed.
    If there are an even number of samples, N, then there are an odd
    number of intervals (N-1), but Simpson's rule requires an even number
    of intervals. The parameter 'even' controls how this is handled.

    Note that this simply wraps scipy.integrate.simpson (handles duals and units)

    Parameters
    ----------
    y : array_like
        Array to be integrated.
    x : array_like, optional
        If given, the points at which `y` is sampled.
    dx : float, optional
        Spacing of integration points along axis of `x`. Only used when
        `x` is None. Default is 1.
    axis : int, optional
        Axis along which to integrate. Default is the last axis.
    even : str {'avg', 'first', 'last'}, optional
        'avg' : Average two results:1) use the first N-2 intervals with
                  a trapezoidal rule on the last interval and 2) use the last
                  N-2 intervals with a trapezoidal rule on the first interval.
        'first' : Use Simpson's rule for the first N-2 intervals with
                a trapezoidal rule on the last interval.
        'last' : Use Simpson's rule for the last N-2 intervals with a
               trapezoidal rule on the first interval.

    """
    # Prepare all the operands
    y_, x_, dx_, yj, xj, dxj, out = setup_dual_operation(y, x, dx)

    if has_jacobians(x):
        raise ValueError("Cannot (yet?) have Jacobians on the integration x term")
    if has_jacobians(dx):
        raise ValueError("Cannot (yet?) have Jacobians on the integration dx term")
    if x is None:
        x_unit = dx.unit
    else:
        x_unit = x.unit
    # Do the raw integration calculation
    result_ = integrate.simpson(y_, x_, dx_, axis=axis, even=even) * y_.unit * x_unit
    if has_jacobians(y):
        result = dlarray(result_)
        for key, jacobian in y.jacobians.items():
            result.jacobians[key] = jacobian.simpson(x_, dx_, axis=axis, even=even)
    else:
        result = result_
    return result
