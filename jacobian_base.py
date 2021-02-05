"""The base class for the jacobians"""
import copy
import numpy as np

from .jacobian_helpers import _broadcasted_shape, _prepare_jacobians_for_binary_op

__all__ = ["BaseJacobian"]


class BaseJacobian(object):

    """This is a container for a jacobian "matrix".  The various child
    classes store the information as either diagonal, a dense array,
    or a sparse array.

    """

    def __init__(
        self,
        template=None,
        dependent_unit=None,
        independent_unit=None,
        dependent_shape=None,
        independent_shape=None,
    ):
        """Define a new jacobian"""

        def pick(*args):
            return next((item for item in args if item is not None), None)

        # Set up the core metadata
        if isinstance(template, BaseJacobian):
            self.dependent_unit = pick(dependent_unit, template.dependent_unit)
            self.independent_unit = pick(independent_unit, template.independent_unit)
            self.dependent_shape = pick(dependent_shape, template.dependent_shape)
            self.independent_shape = pick(independent_shape, template.independent_shape)
        else:
            self.dependent_unit = dependent_unit
            self.independent_unit = independent_unit
            self.dependent_shape = tuple(dependent_shape)
            self.independent_shape = tuple(independent_shape)
        # Do a quick piece of housekeepting
        if type(self.dependent_shape) != tuple:
            self.dependent_shape = tuple(self.dependent_shape)
        if type(self.independent_shape) != tuple:
            self.independent_shape = tuple(self.independent_shape)
        # Now derive a bunch of metadata from that
        self.shape = tuple(self.dependent_shape + self.independent_shape)
        self.dependent_size = int(np.prod(self.dependent_shape))
        self.independent_size = int(np.prod(self.independent_shape))
        self.size = self.dependent_size * self.independent_size
        self.dependent_ndim = len(self.dependent_shape)
        self.independent_ndim = len(self.independent_shape)
        self.ndim = self.dependent_ndim + self.independent_ndim
        self.shape2d = (self.dependent_size, self.independent_size)
        self._dummy_dependent = (1,) * self.dependent_ndim
        self._dummy_independent = (1,) * self.independent_ndim

    def __str__(self):
        return (
            f"Jacobian of type {type(self)}\n"
            + f"Dependent shape is {self.dependent_shape} <{self.dependent_size}>\n"
            + f"Independent shape is {self.independent_shape}"
            + f"<{self.independent_size}>\n"
            + f"Combined they are {self.shape} <{self.size}>\n"
            + f"Dummies are {self._dummy_dependent} and {self._dummy_independent}\n"
            + f"Units are d<{self.dependent_unit}>/d<{self.independent_unit}> = "
            + f"{(self.dependent_unit/self.independent_unit).decompose()}"
        )

    def __repr__(self):
        return self.__str__()

    def __neg__(self):
        return type(self)(-self.data, template=self)

    def __add__(self, other):
        s_, o_, result_type = _prepare_jacobians_for_binary_op(self, other)
        return result_type(data=s_ + o_, template=self)

    def __subtract__(self, other):
        s_, o_, result_type = _prepare_jacobians_for_binary_op(self, other)
        return result_type(data=s_ - o_, template=self)

    def __lshift__(self, unit):
        result = copy.copy(self)  # or should this be deepcopy
        result.dependent_unit = unit

    def real(self):
        return type(self)(np.real(self.data), template=self)

    # This routine is called by the child classes to set up for a
    # premul_diag.  It works out the units issues and sets up for
    # broadcasting.
    def _prepare_premul_diag(self, diag):
        # print (f"Asked for premul_diag on {self}\n.... with {diag.shape}")
        if hasattr(diag, "unit"):
            dependent_unit = diag.unit * self.dependent_unit
            diag_ = diag.value
        else:
            dependent_unit = self.dependent_unit
            diag_ = diag
        dependent_shape = _broadcasted_shape(self.dependent_shape, diag_.shape)
        # print (f"Will return dependent_shape={dependent_shape}")
        return diag_, dependent_unit, dependent_shape

    def flatten(self, order="C"):
        """flatten a jacobian"""
        return self.reshape((self.dependent_size,), order=order)

    def nan_to_num(self, copy=True, nan=0.0, posinf=None, neginf=None):
        return self.__class__(
            template=self,
            data=np.nan_to_num(
                self.data, copy=copy, nan=nan, posinf=posinf, neginf=neginf
            ),
        )

    def to(self, unit):
        """Change the dependent_unit for a Jacobian"""
        if unit == self.dependent_unit:
            return self
        scale = self.dependent_unit._to(unit) * (unit / self.dependent_unit)
        # print (f"Scaling from {self.dependent_unit} to {unit}, factor={scale}")
        return self.scalar_multiply(scale)

    def decompose(self):
        """Decompose the dependent_unit for a Jacobian"""
        raise NotImplementedError("Should not be needed")
        print(f"In decompose comes in with {self.dependent_unit}")
        unit = self.dependent_unit.decompose()
        print(f"In decompose, try to give unit {unit}")
        result = self.to(unit)
        print(f"In decompose goes out with {self.dependent_unit}")
        return result

    def scalar_multiply(self, scale):
        """Multiply Jacobian by a scalar"""
        self.dependent_unit *= scale.unit
        self.data *= scale.value
        return self
