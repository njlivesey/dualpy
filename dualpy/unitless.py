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

    def __truediv__(self, other):
        try:
            return 1.0 / other
        except TypeError:
            return self

    def __rtruediv__(self, other):
        return other

    # Will probably need more
