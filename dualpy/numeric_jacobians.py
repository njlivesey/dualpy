"""A module for computing Jacobians numerically

This is intended for validating code using dualpy.  It computes Jacobians on arguments
to a function by perturbation.  There is an old routine that works well for more
conventional functions that receive and return duals as arguments.  The newer routine is
more flexible and can come with more complex inputs/outputs.
"""

import astropy.units as units
import collections
import copy
import inspect
import numpy as np

from .duals import dlarray
from .jacobians import DenseJacobian


def compute_jacobians_numerically_original(
    func,
    args=None,
    kwargs=None,
    plain_func=None,
):
    """Compute Jacobians by perturbation for comparison to analytical

    Take a function and set of arguments, run the function once with
    analytical Jacobians, then perturb each seeded element in turn
    to compute equivalent numerical Jacobians.  This is used for
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
        times as needed to provide numerical Jacobians
    *args - sequence
        The arguments to func
    **kwargs - dict
        Any keyword arguments to func
    plain_func: callable, optional
        A version of func that works for non-dual arguments if func is not suitable
        for that

    Returns
    -------
    analytical_result, numerical_result: duals
        The result of calling func with args/kwargs with Jacobians computed
        analytically and numerically, respecively.

    Raises
    ------
    ValueError if an input contains more than one Jacobian or any Jacobian
    is non-square.
    """
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
            jacobian = np.ndarray((result0.size, a.size), dtype=result0.dtype)
            # There may be a more pythonic way to do this, but for now this works.
            for i in np.arange(a.size):
                print(f"Perturbation {i+1}/{a.size}")
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


def dissect_object_for_duals(a, depth=0, memo=None):
    """Descend through an object looking for and noting any duals

    Arguments
    ---------
    a : anything
        Object to be dissected

    Returns
    -------
    result : list
        Tree describing where to find the duals
    """
    prefix = ".." * (depth + 1) + " "
    if memo is None:
        memo = set()
    if depth > 100:
        raise ValueError(prefix + "Too deep")
    if type(a) in memo:
        print(prefix + "Type recursion detected")
        return None
    memo = copy.copy(memo)
    memo.add(type(a))
    # If this is a dual, then return a leaf node to that effect
    if isinstance(a, dlarray):
        return ("dual", a)
    # If it's a boring base class return None (and avoid a whole load of annoying
    # recursion)
    if isinstance(a, (int, bool, float, str, property, type, str)):
        return None
    other_missable_types = [
        "NoneType",
        "module",
        "numpy.ndarray",
        "ctypes",
        "astropy.units.quantity.Quantity",
        "Quantity",
        "memoryview",
    ]
    a_type = str(type(a))
    for omt in other_missable_types:
        if omt in a_type:
            return None
    print(prefix + f"Evaluating object of type {type(a)}")
    # Otherwise it might be a collection or object that contains duals.  Setup to explor
    # that possibility.
    object_family = None
    entries = []
    # First explore the list/tuple possibility
    if isinstance(a, collections.abc.Sequence) and not isinstance(a, str):
        for i, entry in enumerate(a):
            dissection = dissect_object_for_duals(entry, depth + 1, memo)
            if dissection is not None:
                entries.append([i, dissection])
        object_family = "Sequence"
        print(prefix + "Passed sequence!")
    # Now see if it's a dict-like
    if object_family is None:
        try:
            for key, item in a.items():
                dissection = dissect_object_for_duals(item, depth + 1, memo)
                if dissection is not None:
                    entries.append([key, dissection])
            object_family = "dict-like"
            print(prefix + "Passed dict")
        except AttributeError:
            print(prefix + f"Failed dict, yet {type(a)}, {isinstance(a, dict)}")
            pass
    # Now see if it's an object with attributes
    if object_family is None:
        print(prefix + "OK, considering as object")
        object_family = "object"
        members = inspect.getmembers(a)
        for name, value in members:
            # Skip dunder methods/attributes
            if name.startswith("__"):
                continue
            # Skip methods
            if callable(value):
                continue
            # Skip properties
            if isinstance(value, property):
                continue
            # Only attributes remain, query those, dissecting downwards
            print(prefix + f"  Considering {name}, {type(value)}")
            dissection = dissect_object_for_duals(value, depth + 1, memo)
            if dissection is not None:
                entries.append([name, dissection])
    print(prefix + f"This has been identified as {object_family}")
    # Now if we have no entries return None
    if not object_family or not entries:
        return None
    else:
        return [object_family] + entries
