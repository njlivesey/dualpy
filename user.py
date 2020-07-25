import numpy as np

from .jacobians import *
from .duals import *

__all__ = ["seed", "_seed_dense", "_seed_sparse", "_seed_diagonal", "compute_jacobians_numerically"]

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
                f"Proposed seed already has a jacobian named '{name}'" +
                " (set overwrite as well as force?)")
    if type(value) is not dlarray or reset:
        out = dlarray(value)
    else:
        out = value
    jacobian=dljacobian_diagonal(
        np.ones(out.shape), dependent_unit=value.unit, independent_unit=value.unit,
        dependent_shape=value.shape, independent_shape=value.shape)
    out.jacobians[name]=jacobian
    return out

# These two are used for testing purposes, to explore functionality
# perhaps not otherwise ventured into.
def _seed_dense(value, name, **kwargs):
    result = seed(value, name, **kwargs)
    result.jacobians[name] = dljacobian_dense(result.jacobians[name])
    return result

def _seed_sparse(value, name, **kwargs):
    result = seed(value, name, **kwargs)
    result.jacobians[name] = dljacobian_sparse(result.jacobians[name])
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
        plain_func=func
    if args is None:
        args=tuple()
    if kwargs is None:
        kwargs=dict()
    result_a=func(*args, **kwargs)
    result0=units.Quantity(result_a)
    result_n=dlarray(result0)
    #
    # Now combine the args and kwargs into one set of iterable items
    # and names (which are none for the args)
    all_args=[]
    all_arg_names=[]
    for a in args:
        all_args.append(a)
        all_arg_names.append(None)
    for n, a in kwargs.items():
        all_args.append(a)
        all_arg_names.append(n)
    # Now create a version of these arguments where all duals have
    # been demoted to regular quantities
    all_args_no_duals=[]
    seed_names=[]
    for a in all_args:
        if isinstance(a, dlarray):
            # The only duals allowed are seeds, let's check thats the
            # name
            if len(a.jacobians) == 0:
                continue
            if len(a.jacobians) != 1:
                raise ValueError("Inputs can only have one Jacobian")
            name=list(a.jacobians.keys())[0]
            j=a.jacobians[name]
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
    args_no_duals=[]
    kwargs_no_duals={}
    for n, a in zip(all_arg_names, all_args_no_duals):
        if n is None:
            args_no_duals.append(a)
        else:
            kwargs_no_duals[n]=a
    # Now define our perturbations
    finfo=np.finfo(0.0)  # result0[0].value
    ptb_f=np.sqrt(finfo.eps)
    ptb_a=ptb_f
    # Now, iterate over all our arguments
    for a, a_nd in zip(all_args, all_args_no_duals):
        # For the seeds we'll go through them one by one and perturb them
        if isinstance(a, dlarray):
            name=list(a.jacobians.keys())[0]
            template = a.jacobians[name]
            # Create the 2D matrix that will contain the numerical Jacobians
            jacobian=np.ndarray((result0.size, a.size))
            # There may be a more pythonic way to do this, but for now this works.
            for i in np.arange(a.size):
                # Perturb one element, call the function, put the
                # original value back and note the results.
                a_nd_flat=a_nd.reshape(-1)
                oldV=a_nd_flat[i]
                dx=np.maximum(np.abs(oldV*ptb_f), (ptb_a << oldV.unit))
                a_nd_flat[i] += dx
                resultP=plain_func(*args_no_duals, **kwargs_no_duals)
                a_nd_flat[i]=oldV
                dResult=resultP - result0
                jacobian[:, i]=(dResult/dx).value.ravel()
            # Store the Jacobian
            target_shape = result0.shape + template.independent_shape
            jacobian = np.reshape(jacobian, target_shape)
            jacobian = dljacobian_dense(
                data=jacobian, template=template, dependent_shape=result0.shape)

            result_n.jacobians[name]=jacobian
    # Now we're done, I think
    return result_a, result_n
