"""Defines the Unitless class that's a placeholder for duals with no units"""

__all__ = ["Unitless"]


class Unitless:
    """Identifies, and handles algebra for, non-existent units"""

    def __str__(self):
        return "Unitless"

    def __repr__(self):
        return "Unitless"

    def __mul__(self, other):
        return other

    def __rmul__(self, other):
        return other

    def __imul__(self, other):
        return other

    def __truediv__(self, other):
        try:
            return other ** (-1)
        except TypeError:
            return self

    def __rtruediv__(self, other):
        return other

    def __itruediv__(self, other):
        return other ** (-1)

    # Will probably need more
    def __eq__(self, other):
        return isinstance(other, Unitless)
