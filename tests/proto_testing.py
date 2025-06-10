"""A sketch of a testing module for duaply"""

from typing import Callable, Sequence, Mapping
from dataclasses import dataclass
import pint
import numpy as np

ureg = pint.UnitRegistry()
# OR this might need to be:
#    from mls_scf_tools.mls_pint import ureg

# A reminder that "Sequence" means the argument can be a list or tuple or similar.
# Similarly, "Mapping" means a dict-like quantity.


@dataclass
class DualpyTestArgument:
    """Describes one argument to a function to be tested"""

    not_scalar: bool = False  # Argument cannot be a scalar
    not_pint: bool = False  # Argument cannot be a pint Quantity
    not_negative: bool = False  # Argument cannot be a negative number
    not_zero: bool = False  # Argument cannot be zero
    allowed_ndims: Sequence[int] = None  # List of ndims the argument can have
    allowed_units: Sequence[pint.Unit] = None  # List of units the argument must have
    # Down the road, I could imagine some arguments related to dtype (particularly for
    # complex numbers, but we'll skip that for now.)


@dataclass
class DualpyTest:
    """Describes the tests to perform on a dualpy function"""

    # The function to test
    function: Callable
    # Now a function that describes the Jacobians
    jacobian_function: Callable
    # The ndarray/pint/dlarray arguments and keyword-arguments for the function, stored
    # as DualpyTestArgument objects, see above.
    args: Sequence[DualpyTestArgument] = None
    kwargs: Mapping[DualpyTestArgument] = None
    # Any other arguments or keyword arguments that are needed, or optional.  If the
    # function can be run with and without these arguments, then the intent is to create
    # two similar DualpyTest instances, one with, and one without, as the presence or
    # absence of these extra arguments may have implications for the properties of the
    # array-like arguments
    extra_args: Sequence = None
    extra_kwargs: Sequence = None
    # Now some details about relationships among the arguments
    arrays_need_same_shape: bool = False
    arrays_must_be_broadcastable: bool = False
    arrays_need_same_units: bool = False
    # Down the road I can imagine other rules, but I'm not sure how we'd implement them
    # (e.g., matrix multiply has specific rules on shape)

    def perform(self):
        """Performs a given test"""

        # Assemble the arguments.  Gather scalar, ndarray, pint incarnations of each
        # argument, along with dual-wrapped version of each.  Look at the ndims
        # arguments and make 1, 2, and 3 dimensional arguments (that should be enough).
        # The arrays don't need to be huge, but each dimension should probably be
        # different.  How about the 3D array be shape (3,5,7)?

        # Assemble the individual calls to the function.  Look at
        # arrays_need_same_shape, arrays_must_be_broadcastable etc. and use them to
        # narrow down the combinations we're going to call the function with.  Look at
        # itertools.product, you can use that to get the fully-populated set of combinations
        # e.g., add(a,p); add(a,q); add(b,p); add(b,q) etc.  Then winnow them down.

        # Be sure that the you include tests where one argument is a dual and the other
        # not (for the binary operations that is), as well as ones where both are (cases
        # where neither are are not dualpy tests.)

        # Call the function with each set of arguments (don't forget to supply the extra
        # args/kwargs).

        # Check that the answers from the ualled calls are np.allclose to those from the
        # dualled ones.

        # Then we're going to want some code to check on the Jacobians. We can deal with
        # that down the road, once you've got this first bit going.

        pass


def example():
    """An example set of implementations"""
    test_exp = DualpyTest(
        function=np.exp,
        jacobian_function=np.exp,
        args=[DualpyTestArgument(allowed_units=ureg.dimensionless)],
    )
    test_sin = DualpyTest(
        function=np.exp,
        jacobian_function=np.cos,
        args=[DualpyTestArgument(allowed_units=ureg.rad)],
    )

    for test in [test_exp, test_sin]:
        test.perform()
