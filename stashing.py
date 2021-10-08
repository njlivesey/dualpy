"""A module for supporting the stashing of Jacobians"""

import astropy.units as units

from .duals import dlarray


class Stash:
    """Used to stash and then re-apply Jacobian information

    Of most use for tasks involving sparse Jacobians
    """

    def __init__(self):
        """Create an empty stash"""
        self._all_stashed_jacobians = {}

    def __str__(self):
        return str(self._all_stashed_jacobians)

    def record_and_reseed(self, dual, name, initial_type="diagonal", replace=False):
        """Stash the jacobians in a dual and reseed it"""
        from .user import seed

        # Possibly complain if we've already got a stash with this name
        if name in self._all_stashed_jacobians() and not replace:
            raise ValueError(
                f'Already have stashed Jacobians for {name}, set "replace"?'
            )
        # Save the Jacobians for this in a stash
        self._all_stashed_jacobians[name] = dual.jacobians
        # Return a freshly seeded version of the dual
        return seed(dual, name, initial_type=initial_type, reset=True)

    def unstash(self, dual):
        """Remap Jacobians in a dual according to this stash"""
        original_jacobians = dual.jacobians
        # Create a jacobian-less version of dual for the result
        result = dlarray(units.Quantity(dual))
        result.jacobians = {}
        # Now loop over the original Jacobians and find out what they map to
        for original_name, original_jacobian in original_jacobians.items():
            # See if this Jacbian is for an independent variable that is in our stash
            if original_name in self._all_stashed_jacobians:
                # If it is, we want to replace with with mapped versions of ourself,
                # which is just the chain rule, and actually just a matrix multiply of
                # the data2Ds for each Jacobian.
                stashed_jacobians = self._all_stashed_jacobians[original_name]
                for component_name, component_jacobian in stashed_jacobians.items():
                    result.jacobians[
                        component_name
                    ] = original_jacobian.matrix_multiply(component_jacobian)
            else:
                # If this Jacobian does not refer to an independent variable in our
                # stash, then simply copy it over.
                result.jacobians[original_name] = original_jacobian
