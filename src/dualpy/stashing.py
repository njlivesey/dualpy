"""A module for supporting the stashing of Jacobians"""

import astropy.units as units

from .duals import dlarray
from .user import has_jacobians
from .jacobian_helpers import jacobian_2d_matrix_multiply


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
        if name in self._all_stashed_jacobians and not replace:
            raise ValueError(
                f'Already have stashed Jacobians for {name}, set "replace"?'
            )
        # Save the Jacobians for this in a stash
        self._all_stashed_jacobians[name] = dual.jacobians
        # Return a freshly seeded version of the dual
        return seed(dual, name, initial_type=initial_type, reset=True)

    def unstash(self, dual, collisions="forbid"):
        """Remap Jacobians in a dual according to this stash.

        Parameters
        ----------

        dual: dlarray
            Dual that is to have its Jacobians updated

        collisions: str, default="forbid"
            Action to take if more than one Jacbian result in the same name.  Default is
            "forbid", which raises an error.  "overwrite" replaces earlier Jacobians
            with any new ones.  "add" accumulates contributions.

        Result
        ------

        Returns an updated version of the dual. Is pass through if result is not a dual.

        """
        # If there are no Jacobians to update, simply act as a passthrough.
        if not has_jacobians(dual):
            return dual
        original_jacobians = dual.jacobians
        # Create a jacobian-less version of dual for the result
        raise NotImplementedError("Not working, line below this message needs update for pint etc.")
        result = dlarray(units.Quantity(dual))
        result.jacobians = {}
        # Now loop over the original Jacobians and find out what they map to
        for original_name, original_jacobian in original_jacobians.items():
            # See if this Jacbian is for an independent variable that is in our stash
            if original_name in self._all_stashed_jacobians:
                # If it is, we want to replace (or perhaps add) with with mapped
                # versions of ourself, which is just the chain rule, and actually just a
                # matrix multiply of the data2Ds for each Jacobian.
                stashed_jacobians = self._all_stashed_jacobians[original_name]
                for component_name, component_jacobian in stashed_jacobians.items():
                    new_matrix = jacobian_2d_matrix_multiply(
                        original_jacobian, component_jacobian
                    )
                    add_matrices = False
                    if component_name in result.jacobians:
                        if collisions == "forbid":
                            raise ValueError(
                                f"Collision for component {component_name}"
                            )
                        elif collisions == "add":
                            add_matrices = True
                        elif collisions == "overwrite":
                            pass
                        else:
                            raise ValueError(
                                f"Illegal value for collisions: {collisions}"
                            )
                    if add_matrices:
                        result.jacobians[component_name] += new_matrix
                    else:
                        result.jacobians[component_name] = new_matrix
            else:
                # If this Jacobian does not refer to an independent variable in our
                # stash, then simply copy it over.
                result.jacobians[original_name] = original_jacobian
        return result

