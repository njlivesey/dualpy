"""Provides routines supporting dualpy users directly"""

import copy
from abc import abstractmethod
from typing import Optional, Union

import astropy.units as units
import numpy as np
import pint

# pylint: disable-next=no-name-in-module
from numpy.typing import NDArray

from dualpy.dual_helpers import has_jacobians
from dualpy.duals import dlarray
from dualpy.jacobians import (
    DenseJacobian,
    DiagonalJacobian,
    SeedJacobian,
    SparseJacobian,
)

PossibleDual = Union[units.Quantity, pint.Quantity, NDArray, dlarray]


class DualableMixin:
    """This is a mixin that flags a class that has its own dual methods

    The intent is to ensure that all the required methods are indeed provided
    """

    @abstractmethod
    def _dual_seed_(
        self,
        name: str,
        force: Optional[bool] = False,
        overwrite: Optional[bool] = False,
        reset: Optional[bool] = False,
        initial_type: Optional[str] = None,
    ):  # -> Self
        pass

    @abstractmethod
    def _dual_has_jacobians_(self) -> bool:
        pass

    @abstractmethod
    def _dual_dedual_(self):  # -> Self:
        pass

    @abstractmethod
    def _dual_delete_jacobians_(
        self,
        *names: str,
        wildcard: Optional[str] = None,
        remain_dual: Optional[bool] = False,
        **kwargs,
    ):  # -> Self
        pass


class DroppedJacobianWarning(Warning):
    """Issued when an operation drops the Jacobians for a dual"""


def seed(
    value,
    name,
    force=False,
    overwrite=False,
    reset=False,
    initial_type="seed",
    **kwargs,
):
    """Return a dual for a quantity populated, with a unitary Jacobian matrix

    In some senses, this is the most important routine in the dualpy package, as it's
    probably the only one most users will knowingly invoke on a regular basis.  It takes
    an array-like quantity unit jacobian for it.  From that point on, anything computed
    from the resulting dual will track Jacobians appropriately.

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
    # First see if the quantity has a _dual_seed_ method, and, if so, invoke that to do
    # the work.
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
        # pylint: disable=protected-access
        dependent_unit=out._dependent_unit,
        independent_unit=out._dependent_unit,
        # pylint: enable=protected-access
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
    # First see if this quantity has a _delete_jacobians_ method.  If so, use it.
    if hasattr(a, "_dual_delete_jacobians_"):
        # pylint: disable-next=protected-access
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


def apply_sparse_threshold(x: PossibleDual, percent_threshold: Optional[float] = None):
    """Densify jacobians that are denser than a given percent sparsity"""
    # Do nothing if there are no Jacobians for the argument
    if not has_jacobians(x):
        return x
    if percent_threshold is None:
        percent_threshold = 10.0
    # Make a copy
    result: dlarray = copy.copy(x)
    result.jacobians = copy.copy(result.jacobians)
    for key, jacobian in result.jacobians.items():
        if jacobian.percent_full > percent_threshold:
            result.jacobians[key] = DenseJacobian(jacobian)
    return result


def get_jacobians_for_function_inverse(
    y_target, y_solution, dx_key: str, ignored_jkeys: list[str] = None
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
    if ignored_jkeys is None:
        ignored_jkeys = []
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
