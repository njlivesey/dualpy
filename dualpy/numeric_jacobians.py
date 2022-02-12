"""A module for computing numeric Jacobians

This is intended for validating code using dualpy.  It computes Jacobians on arguments
to a function by perturbation.  There is an old routine that works well for more
conventional functions that receive and return duals as arguments.  The newer routine is
more flexible and can come with more complex inputs/outputs.

Classes wanting to support computation of numeric Jacobians on them, or more
specifically, their attributes should include the following method:

    def _numeric_jacobian_support(self, **kwargs):
        '''Wraps dualpy.generic_numeric_jacobian_support, see documentation'''
        from dualpy import generic_numeric_jacobian_support
        generic_numeric_jacobian_support(self, **kwargs)

If they wish only a subset of their attributes to be considered when evaluating numeric
Jacobians, they should set their own _numeric_jacobian_specific_attributes attribute to
be a list of the names of those attributes.

"""

import astropy.units as units
import collections
import copy
import inspect
import numpy as np

from .duals import dlarray
from .user import delete_jacobians, has_jacobians
from .jacobians import DenseJacobian, SeedJacobian

__all__ = [
    "compute_numeric_jacobians",
    "generic_numeric_jacobian_support",
]


def compute_numeric_jacobians(
    func,
    args=None,
    kwargs=None,
    plain_func=None,
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
    *args - sequence
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
    ValueError if an input contains more than one Jacobian or any Jacobian
    is non-square.

    Notes
    -----

    If the return value of the invoked function has the method
    _numeric_jacobian_support, then that is invoked to compute the jacobian-element =
    delta_y / delta_x term.  The method should be structured as

    def _numeric_jacobian_support(self, result_perturbed, independent_name,
        independent_element, perturbation, initialize=False)

    The method is expected to record the numeric Jacobians internally.  That Jacobian
    should be obtained by comparing the values in "self" to those in result_perturbed,
    which correspond to the output from the function in question with the given element
    (ravelled) of a given named input perturbed by a given amount.  If called with
    initialize set true, then this indicates that self, which will be the result of the
    analytical Jacobian calculation is to have its Jacobians set to dense zero arrays.

    Most likely, classes that supply this capability will simply invoke the
    generic_numeric_jacobian_support routine defined below.

    Such cases are noted internall using the opaque_result flag.

    """
    # First compute the unperturbed result
    if plain_func is None:
        plain_func = func
    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()
    result_a = func(*args, **kwargs)
    result0 = delete_jacobians(result_a)
    # See if this result is of a class that has it's own support for computing numeric
    # jacobians
    opaque_result = hasattr(result0, "_numeric_jacobian_support")
    if not opaque_result:
        # OK, so the results are something we can interpert ourselves and difference to
        # get numeric Jacobians.
        try:
            result_n = dlarray(result0)
        except TypeError:
            raise ValueError(
                "Unable to handle function return (cannot convert to dual)"
            )
    else:
        # In thise case, the output of the function is opaque to us, and we need to
        # invoke a method to insert the Jacobians from the result of perturbed runs.
        # Initialize that process.
        result_n = copy.deepcopy(result_a)
        result_n._numeric_jacobian_support(initialize=True)
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
            # The only duals that we will actually perturb are those that are digaonal
            # and seeds (we will allow zero entries on the diagonal, which will signify
            # a specific element to skip).
            found_one = False
            for name, jacobian in a.jacobians.items():
                # We'll only consider DiagonalJacobians
                if not isinstance(jacobian, SeedJacobian):
                    continue
                if found_one:
                    raise ValueError("More than one SeedJacobian in input arguments")
                found_one = True
                # Now check that the values are all unity or zero, require exact match,
                # surely that's OK even for floating point.
                if not np.all((jacobian.data == 0.0) | (jacobian.data == 1.0)):
                    raise ValueError(
                        "The putative SeedJacobian has entries other than zero or one"
                    )
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
            seed_jacobian = a.jacobians[name]
            seed_jacobian_values = seed_jacobian.data.ravel()
            if not opaque_result:
                # Create the 2D matrix that will contain the numeric Jacobians
                jacobian = np.ndarray((result0.size, a.size), dtype=result0.dtype)
            # There may be a more pythonic way to do this, but for now this works.
            for i in range(a.size):
                # If this entry of the seed is zero, we're skipping this element
                if seed_jacobian_values[i] == 0.0:
                    continue
                # Perturb one element, call the function, put the
                # original value back and note the results.
                a_nd_flat = a_nd.reshape(-1)
                oldV = a_nd_flat[i]
                dx = np.maximum(np.abs(oldV * ptb_f), (ptb_a << oldV.unit))
                a_nd_flat[i] += dx
                result_perturbed = plain_func(*args_no_duals, **kwargs_no_duals)
                a_nd_flat[i] = oldV
                if opaque_result:
                    # If this result class provides its own method for computing
                    # numeric Jacobian elements, then invoke that.
                    result_n._numeric_jacobian_support(
                        result_perturbed=result_perturbed,
                        independent_name=name,
                        independent_element=i,
                        perturbation=dx,
                        initialize=False,
                    )
                else:
                    # Otherwise, we can do it ourselves
                    delta_result = result_perturbed - result0
                    jacobian[:, i] = (delta_result / dx).value.ravel()
            # Store the Jacobian (if it's up to us to do so)
            if not opaque_result:
                target_shape = result0.shape + seed_jacobian.independent_shape
                jacobian = np.reshape(jacobian, target_shape)
                jacobian = DenseJacobian(
                    data=jacobian,
                    template=seed_jacobian,
                    dependent_shape=result0.shape,
                    dependent_unit=delta_result.unit,
                    independent_unit=dx.unit,
                )
                result_n.jacobians[name] = jacobian
    # Now we're done, I think
    return result_a, result_n


def _consider_member(name, value, result_entries, keys=None):
    """A helper routine for generic_numeric_jacobian_support"""
    # Skip dunders and callables
    if name.startswith("__") or callable(value):
        return
    # OK, we might be adding it, so think how we'll describe that addition
    if keys is None:
        keys = []
    # full_name = name + "".join([f"[{key}]" for key in keys])
    # If it's a dual add it
    if isinstance(value, dlarray):
        # print(f"Found {full_name} as dual")
        result_entries.append([name] + keys)
        return
    # Otherwise if it provides a _numeric_jacobian_support method, add it
    if hasattr(value, "_numeric_jacobian_support"):
        # print(f"Found {full_name} as supporting numeric Jacobians")
        result_entries.append([name] + keys)
        return
    # Otherwise see if it's a collection of some kind, note that this doesn't consider
    # nested collections, though I guess it could if we tried hard.  First try
    # lists/tuples
    if isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
        # print(f"Found {full_name} as sequence")
        for key, item in enumerate(value):
            _consider_member(name, item, result_entries, keys=keys + [key])
        return
    # Now consider dicts
    if isinstance(value, collections.Mapping):
        # print(f"Found {name} as mapping")
        for key, item in value.items():
            _consider_member(name, item, result_entries, keys=keys + [key])


def generic_numeric_jacobian_support(
    result,
    result_perturbed=None,
    independent_name=None,
    independent_element=None,
    perturbation=None,
    initialize=False,
    specific_attributes=None,
):
    """A generic routine for providing support for numeric jacobians.

    See the documentation of compute_jacobians_numericly, above.
    """
    if not initialize:
        if any(
            [
                arg is None
                for arg in [
                    result_perturbed,
                    independent_name,
                    independent_element,
                    perturbation,
                ]
            ]
        ):
            raise ValueError("Some arguments are missing but initialize is not set")
    # Work out what attributes to attempt to handle if we've not been given them
    # explicitly.  These constitute the dependent variables output from the function.
    # It's possibly they were supplied directly.  If not, check if our type has a list
    # of such things.
    if specific_attributes is None:
        specific_attributes = getattr(
            result, "_numeric_jacobian_specific_attributes", None
        )
    # Now we're going to form our result_entries, which is a list of lists, with the
    # outer list corresponding to each item in the result that is a dual or indiciates
    # it supports numerical Jacobians.  The lists for each of those item are the
    # attribute name followed by any __getitem__ keys needed to further extract them.
    if specific_attributes is not None:
        # In this case, there are no __getitem__ keys
        result_entries = [
            [specific_attribute] for specific_attribute in specific_attributes
        ]
    else:
        result_entries = []
        members = inspect.getmembers(result)
        for name, value in members:
            # Consider this candidate, adding it if it's a type we can deal with (or is
            # a collection that includes things we can deal with)
            _consider_member(name, value, result_entries)

    # Now loop over the attributes/dependent-variables
    for result_entry in result_entries:
        # Identify the attribute
        attribute_name = result_entry[0]
        attribute_keys = result_entry[1:]
        # Extract it, and possible extract it from result_perturbed
        attribute = getattr(result, attribute_name)
        if result_perturbed is not None:
            attribute_perturbed = getattr(result_perturbed, attribute_name)
        else:
            attribute_perturbed = None
        # Now descend through any keys
        for key in attribute_keys:
            attribute = attribute[key]
            if result_perturbed is not None:
                attribute_perturbed = attribute_perturbed[key]
        # If it has a _numeric_jacobian_support method, then invoke that
        if hasattr(attribute, "_numeric_jacobian_support"):
            # Note that initize could be True here, in which case everything else is
            # None, hence the third argument to getattr.
            attribute._numeric_jacobian_support(
                result_perturbed=attribute_perturbed,
                independent_name=independent_name,
                independent_element=independent_element,
                perturbation=perturbation,
                initialize=initialize,
            )
            continue
        # Otherwise, if it doesn't have Jacobians, then skip it
        if not has_jacobians(attribute):
            continue
        # Otherwise, it does have Jacobians, so we'll try to attend to it ourselves
        if initialize:
            # This is the first time round, zero out the Jacobians we're computing
            for jacobian_name, jacobian in attribute.jacobians.items():
                new_jacobian = jacobian.to_dense()
                new_jacobian.data[...] = 0.0
                attribute.jacobians[jacobian_name] = new_jacobian
        else:
            # Otherwise, this is a particular perturbation, deal with that.
            # Get a view of the Jacobian array as a 2D matrix
            jacobian = attribute.jacobians[independent_name]
            delta_y = attribute_perturbed - units.Quantity(attribute)
            jacobian.data2d[:, independent_element] = delta_y.ravel() / perturbation
