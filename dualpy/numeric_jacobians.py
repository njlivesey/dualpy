"""A module for computing numeric Jacobians

This is intended for validating code using dualpy.  It computes Jacobians on arguments
to a function by perturbation.  See the documentation for compute_numeric_jacobians
below.

"""

from dataclasses import dataclass
from typing import Any, Union
import collections
import copy
import inspect
import logging
import numpy as np
from tqdm import tqdm

from .duals import dlarray
from .jacobians import SeedJacobian, DiagonalJacobian, DenseJacobian, SparseJacobian
from .locate_objects import LocatedObjectIterator

__all__ = [
    "compute_numeric_jacobians",
]

logger = logging.getLogger(__name__)


def compute_numeric_jacobians(
    func,
    args=None,
    kwargs=None,
    plain_func=None,
    include_diagonal_jacobians=False,
    include_all_jacobians=False,
):
    """Compute Jacobians by perturbation for comparison to analytical

    Take a function and set of arguments, run the function once with
    analytical Jacobians, then perturb each seeded element in turn
    to compute equivalent numeric Jacobians.  This is used for
    testing the analytical Jacobian calcualtions.  If "func" cannot
    be called for non-duals (e.g., voigt_profile exists for ndarray
    and dlarray, but not units.Quantity) then the optional
    plain_func argument provides a non-dual compatible routine
    (e.g., a wrapper that promotes one or more arguments to
    dual to then invoke func on).

    Arguments
    ---------
    func - callable
        The function that provides analytical Jacobians and that is to be called as many
        times as needed to provide numeric Jacobians
    *args - sequencek
        The arguments to func
    **kwargs - dict
        Any keyword arguments to func
    plain_func: callable, optional
        A version of func that works for non-dual arguments if func is not suitable
        for that

    Returns
    -------
    analytical_result, numeric_result: duals
        The result of calling func with args/kwargs with Jacobians computed
        analytically and numericly, respecively.

    Raises
    ------
    ValueError if an input contains confusing input Jacobians

    Notes
    -----

    The code first searches the inputs for any quantities that have SeedJacobians
    (elements that have zero on the diagonal can be skipped to reduce the number of
    calls.  These are then perturbed one by one and the function called again
    repeatedly.  Any Jacobians in the result from the first (analytical) run are noted
    and replaced with corresponding numerical values based on the perturbed runs.

    Classes wanting to participate in the computation of numerical Jacobians, through
    being an input or an output to the routine being tested, should include an attribute
    _numeric_jacobian_support, which tests True.  If it is a list of strings, then it
    conveys the subset of the attributes that are to be considered when looking for
    duals that can participate in the numeric Jacobians.

    """
    # ----------------------------------------------- Initial setup
    if plain_func is None:
        plain_func = func
    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()
    included_jacobian_types = [SeedJacobian]
    if include_diagonal_jacobians:
        included_jacobian_types = [SeedJacobian, DiagonalJacobian]
    if include_all_jacobians:
        included_jacobian_types += [
            SeedJacobian,
            DiagonalJacobian,
            DenseJacobian,
            SparseJacobian,
        ]
    dx_scale_factor = 1e6
    #
    # ----------------------------------------------- Examine inputs
    #
    # Combine the args and kwargs into one set of iterable items and names (which are
    # none for the args)
    all_args = []
    all_arg_names = []
    for arg in args:
        all_args.append(arg)
        all_arg_names.append(None)
    for n, arg in kwargs.items():
        all_args.append(arg)
        all_arg_names.append(n)
    # Now loop over all of these argument and search for duals in them, preprocesing
    # them if needed.  We'll create a copy of the argument list with all seeds removed
    all_deseeded_args = []
    # For each argument, we'll keep information about any duals
    all_arg_trees = []
    # And any seeds
    all_arg_seed_names = []
    # We'll keep track of the names of all the seeds to complain about duplicates
    all_seed_names = []
    # Keep track of how many perturbations we're going to make
    n_elements = 0
    for arg in all_args:
        # Find all the duals in this argument
        dual_tree = LocatedObjectIterator(arg, targets=dlarray)
        # And note this information
        all_arg_trees.append(dual_tree)
        if not dual_tree:
            # If there are none, simply append this argument to the unseeded argument
            # list unmodified
            all_deseeded_args.append(arg)
            # And note that there are no seeds in this one
            all_arg_seed_names.append([])
        else:
            # Otherwise this argument contains duals, see if any contain SeedJacobians.
            any_seed = False
            for dual_arg in dual_tree(arg):
                for jacobian in dual_arg.jacobians.values():
                    if type(jacobian) in included_jacobian_types:
                        any_seed = True
            if not any_seed:
                # If there are no seeds, again, as if there are no duals, append this
                # argument to the unseeded list unmodified and note that there are no
                # seeds in this argument
                all_deseeded_args.append(arg)
                all_arg_seed_names.append([])
            else:
                # Go through and note the seeds in this argument
                these_seeds = []
                for dual_arg in dual_tree(arg):
                    for seed_name, jacobian in dual_arg.jacobians.items():
                        if type(jacobian) in included_jacobian_types:
                            # This is a seed, check it's not a duplicate
                            if seed_name in all_seed_names:
                                raise ValueError(f"Duplicate seed name {seed_name}")
                            # Now check that the values are all unity or zero, require
                            # exact match, surely that's OK even for floating point.
                            diagonal = jacobian.extract_diagonal()
                            if not np.all((diagonal == 0.0) | (diagonal == 1.0)):
                                raise ValueError(
                                    "The putative Jacobian has (diagonal) "
                                    "entries other than zero or one"
                                )
                            # Note this seed
                            all_seed_names.append(seed_name)
                            these_seeds.append(seed_name)
                            # Add the number of elements this involves
                            n_elements += int(np.sum(diagonal))
                all_arg_seed_names.append(these_seeds)
                # OK, we're going to need a de-seeded copy of this argument, so make a
                # deep copy of it, then go through and delete all the seeds
                arg_deseeded = copy.deepcopy(arg)
                for dual_arg in dual_tree(arg_deseeded):
                    for seed_name in these_seeds:
                        if seed_name in dual_arg.jacobians:
                            del dual_arg.jacobians[seed_name]
                all_deseeded_args.append(arg_deseeded)
    if len(all_seed_names) == 0:
        raise ValueError("No seeded arguments")
    #
    # ----------------------------------------------- Setup result
    #
    # Call the function first and get the analytical Jacobians.
    result_a = func(*args, **kwargs)
    # Find the duals in the result
    result_tree = LocatedObjectIterator(result_a, targets=dlarray)
    # Make a deep copy of the result for storing the numeric Jacobians
    result_n = copy.deepcopy(result_a)
    # And another one with all the Jacobians of any kind removed, to use as the
    # unperturbed result.
    result_0 = copy.deepcopy(result_a)
    # Set any Jacobians corresponding to seeds in the output_n to dense zero.  Delete
    # all the jacobians in output_0
    for output_n, output_0 in result_tree(result_n, result_0):
        for key in all_seed_names:
            if key in output_n.jacobians:
                output_n.jacobians[key] = DenseJacobian(
                    template=output_n.jacobians[key]
                )
        output_0.jacobians = {}
    #
    # ----------------------------------------------- Get ready for perturbations
    #
    # Redistribute deseeded arguments backto args/kwargs
    args_deseeded = []
    kwargs_desseded = {}
    for n, arg in zip(all_arg_names, all_deseeded_args):
        if n is None:
            args_deseeded.append(arg)
        else:
            kwargs_desseded[n] = arg
    #
    # ----------------------------------------------- Perturb each argument in turn
    #
    with tqdm(total=n_elements) as bar:
        for (arg, arg_deseeded, arg_tree, seed_names) in zip(
            all_args,
            all_deseeded_args,
            all_arg_trees,
            all_arg_seed_names,
        ):
            # Skip any arguments that do not contain duals
            if not arg_tree:
                continue
            # Check out the iterations - keep needing to debug
            # for seed_name in seed_names:
            #     print(seed_name)
            # for original_dual, deseeded_dual in iterate_nj_tree(
            #         arg_tree, sources=(arg, arg_deseeded)):
            #     print(original_dual.shape, deseeded_dual.shape)

            # Now iterate over all the duals in this argument
            for seed_name, (original_dual, deseeded_dual) in zip(
                seed_names, arg_tree(arg, arg_deseeded)
            ):
                seed_jacobian = original_dual.jacobians[seed_name]
                seed_jacobian_values = seed_jacobian.extract_diagonal().ravel()
                for i in range(original_dual.size):
                    # If this entry of the seed is zero, we're skipping this element
                    if seed_jacobian_values[i] == 0.0:
                        continue
                    # Get the unraveld indices for this element
                    ii = np.unravel_index(i, shape=deseeded_dual.shape)
                    # Record the original value then perturb
                    original = deseeded_dual[ii]
                    dx = (
                        np.spacing(np.array(original)) * dx_scale_factor
                    ) * original.units
                    deseeded_dual[ii] += dx
                    # Finally, we invoke the function
                    result_p = plain_func(*args_deseeded, **kwargs_desseded)
                    bar.update()
                    # Put the unperturbed value back
                    deseeded_dual[ii] = original
                    # Now we have to loop over all the Jacobian-containing outputs and
                    # note the impact of this perturbation in them.  Loop over the
                    # container for the numerical Jacobians, and the perturbed and
                    # unperturbed result and insert the results for any Jacobians with
                    # respect to this seed.
                    for output_n, output_p, output_0 in result_tree(
                        result_n, result_p, result_0
                    ):
                        if seed_name in output_n.jacobians:
                            j = output_n.jacobians[seed_name]
                            # Compute the Jacobian column
                            delta_output = output_p - output_0
                            delta_output = (
                                delta_output.ravel().to(j.dependent_unit).magnitude
                            )
                            dx_value = dx.to(j.independent_unit).magnitude
                            column = delta_output / dx_value
                            # Insert it
                            j.data2d[:, i] = column

    # Now we're done, I think
    return result_a, result_n
