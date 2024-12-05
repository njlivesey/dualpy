"""Module defining the BaseJacobian class from which other types of Jacobian are descended"""

from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
import copy
import warnings
from collections.abc import Sequence
from abc import abstractmethod

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .dual_helpers import (
    get_unit_conversion_scale,
    get_magnitude_and_unit,
)

from .jacobian_helpers import (
    broadcasted_shape,
    prepare_jacobians_for_binary_op,
    GenericUnit,
)

if TYPE_CHECKING:
    from sparse_jacobians import SparseJacobian
    from dense_jacobians import DenseJacobian


@dataclass
class DTypeVessel:
    """Purely contains a dtype for the data field in BaseJacobian

    I did it this way to avoid having jacobian.dtype and jacobian.data.dtype that I need
    to try to keep in lockstep. Add shape to keep the error checking happy
    """

    dtype: DTypeLike
    shape: tuple = tuple()


class BaseJacobian(object):
    """This is a container for a jacobian "matrix".  The various child
    classes store the information as either diagonal, a dense array, or a sparse array.
    They all share information on the shape and units (if applicable) of the dependent
    and independent terms in the Jacobian."""

    @abstractmethod
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        source: ArrayLike | "BaseJacobian" = None,  # pylint: disable=unused-argument
        template: "BaseJacobian" = None,
        dependent_unit: GenericUnit = None,
        independent_unit: GenericUnit = None,
        dependent_shape: Sequence = None,
        independent_shape: Sequence = None,
        dtype: DTypeLike = None,
    ):
        """Populate the basic contents of a BaseJacobian

        Parameters
        ----------
        template : BaseJacobian or child thereof, optional
            If supplied, can be source of shape and units information
        source : ArrayLike or another Jacobian
            Source for Jacobian data (mostly ignored in base class, just the dtype, if
            any, is noted)
        dependent_unit : GenericUnit, optional
            Units for the dependent quantity
        independent_unit : GenericUnit, optional
            Units for the independent quantity
        dependent_shape : Sequence, optional
            Shape for the dependent quantity
        independent_shape : Sequence, optional
            Shape for the independent quantity
        dtype : DTypeLike, optional
            dtype for the data values
        """

        # Define a quick helper rutine
        def pick(*args):
            """Return the first non-none value in the supplied arguments or None"""
            return next((item for item in args if item is not None), None)

        # Set up the core metadata.  First try to populate it from the template if
        # supplied (letting supplied arguments take precedence)
        try:
            source_dtype = source.dtype
        except AttributeError:
            source_dtype = None
        if template:
            self.dependent_unit = pick(dependent_unit, template.dependent_unit)
            self.independent_unit = pick(independent_unit, template.independent_unit)
            self.dependent_shape = pick(dependent_shape, template.dependent_shape)
            self.independent_shape = pick(independent_shape, template.independent_shape)
            dtype = pick(dtype, template.dtype, source_dtype)
        else:
            # Otherwise, just get it from the arguments
            self.dependent_unit = dependent_unit
            self.independent_unit = independent_unit
            self.dependent_shape = dependent_shape
            self.independent_shape = independent_shape
            dtype = pick(dtype, source_dtype)
        # Do a quick piece of housekeepting
        self.dependent_shape = tuple(self.dependent_shape)
        self.independent_shape = tuple(self.independent_shape)
        # Store the dtype in data, see documentation for DTypeVessel above
        self.data = DTypeVessel(dtype=dtype)
        # Now derive a bunch of metadata from the other parameters
        self.shape = tuple(self.dependent_shape + self.independent_shape)
        self.dependent_size = int(np.prod(self.dependent_shape))
        self.independent_size = int(np.prod(self.independent_shape))
        self.size = self.dependent_size * self.independent_size
        self.dependent_ndim = len(self.dependent_shape)
        self.independent_ndim = len(self.independent_shape)
        self.independent_ndim_axis_corrrection = self.independent_ndim
        self.ndim = self.dependent_ndim + self.independent_ndim
        self.shape_2d = (self.dependent_size, self.independent_size)
        self._dummy_dependent = (1,) * self.dependent_ndim
        self._dummy_independent = (1,) * self.independent_ndim

    def __str__(self):
        """Return a string describing the Jacobians

        Note that this doesn't print the values, arguably a deviation from standard
        python, but perhaps forgivable.
        """
        # Run a check on the Jacobian while we're at it.
        result = (
            f"Jacobian of type {type(self)}\n"
            f"Dependent shape is {self.dependent_shape} <{self.dependent_size:,}>\n"
            f"Independent shape is {self.independent_shape}"
            f" <{self.independent_size:,}>\n"
            f"Combined they are {self.shape} <{self.size:,}>\n"
            f"Dummies are {self._dummy_dependent} and {self._dummy_independent}\n"
            f"Units are d<{self.dependent_unit}>/d<{self.independent_unit}> = "
            f"<{(self.dependent_unit/self.independent_unit)}>"
        )
        if self.data is not None:
            result += f"\ndata is {type(self.data)}({list(self.data.shape)}, dtype={self.data.dtype})"
        else:
            result += "\ndata is None"
        return result

    def _check(
        self,
        name: str = None,
        jname: str = None,
        dependent_shape: tuple[int] = None,
        dependent_unit=None,
    ):
        """Perform fundamental checkes on a BaseJacobian

        Parameters
        ----------
        name : str, optional
            An optional name to attach to error messages
        """
        # Check that the array sizes and shapes are consistent
        if name is None:
            name = "<unnamed-variable>"
        if jname is None:
            jname = "<unnamed-jacobian>"
        assert self.dependent_shape + self.independent_shape == self.shape, (
            f"Shape mismatch for {name} Jacobian {jname}, "
            f"{self.dependent_shape} + {self.independent_shape} != {self.shape}"
        )
        assert np.prod(self.dependent_shape) == self.dependent_size, (
            f"Dependent size mismatch for {name} Jacobian {jname}, "
            f"product({self.dependent_shape}) != {self.dependent_size}"
        )
        assert np.prod(self.independent_shape) == self.independent_size, (
            f"Independent size mismatch for {name} Jaobian {jname}, "
            f"product({self.independent_shape}) != {self.independent_size}"
        )
        assert (
            np.prod(self.shape) == self.size
        ), f"Overall size mismatch for {name} Jacobian {jname}, product({self.shape}) != {self.size}"
        # Check that the dependent_shape is correct
        if dependent_shape is not None:
            assert (
                self.dependent_shape == dependent_shape
            ), f"dependent_shape mismatch for {name} Jacobian {jname}: {self.dependent_shape} vs. {dependent_shape} expected"
        # Check that the dependent unit is correct
        if dependent_unit is not None:
            assert (
                self.dependent_unit == dependent_unit
            ), f"dependent_unit mismatch for {name} Jacobian {jname}: {self.dependent_unit} vs. {dependent_unit} expected"
        # Check that data is not from the base class
        assert not isinstance(
            self.data, DTypeVessel
        ), "This seems to be a base Jacobian (data is type DTypeVessel)"

    def __repr__(self):
        """Return a representation of the Jacobians"""
        return self.__str__()

    @abstractmethod
    def get_data_nd(self, form: str = None) -> ArrayLike:
        """Get mutli-dimensional version of Jacobian's data"""

    @abstractmethod
    def get_data_2d(self, form: str = None) -> ArrayLike:
        """Get 2D version of Jacobian's data"""

    @abstractmethod
    def get_data_diagonal(self) -> ArrayLike:
        """Get diagonal version of Jacobian's data

        This should only work for DiagonalJacobians"""

    @property
    def dtype(self):
        """Returns the dtype for a Jacobian"""
        return self.data.dtype

    def __neg__(self):
        """Unary negative for Jacobian"""
        # pylint: disable-next=invalid-unary-operand-type
        return type(self)(source=-self.data, template=self)

    def __pos__(self):
        """Unary positive for Jacobian"""
        return self

    def real(self):
        """Return real part of the Jacobians"""
        return type(self)(source=np.real(self.data), template=self)

    # pylint: disable=import-outside-toplevel
    def __add__(self, other: "BaseJacobian"):
        """Add two Jacobians"""
        from .dense_jacobians import DenseJacobian

        self_data, other_data, result_type = prepare_jacobians_for_binary_op(
            self, other
        )
        result_data = self_data + other_data
        if result_type is DenseJacobian:
            result_data = np.reshape(np.array(result_data), self.shape)
        return result_type(source=result_data, template=self)

    # pylint: disable=import-outside-toplevel
    def __sub__(self, other: "BaseJacobian"):
        """Subtract two Jacobians"""
        from .dense_jacobians import DenseJacobian

        self_data, other_data, result_type = prepare_jacobians_for_binary_op(
            self, other
        )
        result_data = self_data - other_data
        if result_type is DenseJacobian:
            result_data = np.reshape(np.array(result_data), self.shape)
        return result_type(source=result_data, template=self)

    def _force_unit(self, unit):
        """Force a jacobian to take the given dependent_unit"""
        result = copy.copy(self)
        result.dependent_unit = unit

    def get_jaxis(
        self,
        axis: int | Sequence | None,
        none: str = "none",
    ):
        """Make (-ve) axis arguments in reduce-type operations valid for jacobians

        Negative axis requests count backwards from the last index, but the Jacobians
        have the independent_shape appended to their shape, so we need to correct for
        that (or not if its positive).

        The none argument says what to do if the supplied axis is missing/None.  See
        below.

        Parameters
        ----------
        axis : int
            axis for operations ()
        none : str, optional
            Says what should be done if the supplied axis is none.  A variety of options
            are possible, reflecting the diversity of actions that various numpy
            routines take in this eventuality.
             - "none" : return None
             - "flatten": return zero
             - "first" : return the first axis (i.e., zero)
             - "last" : return the final axis (ndim - 1)
             - "all" : return all the dependent axies [0, 1, 2, ...]
             - "transpose" : as "all", but return them in reverse order.
        Returns
        -------
        result : int
            The axis appropriately correctly
        """
        if axis is None:
            if none == "none":
                return None
            if none == "flatten":
                return 0
            if none == "first":
                return 0
            if none == "last":
                return self.dependent_ndim - 1
            if none == "all":
                return tuple(range(self.dependent_ndim))
            if none == "transpose":
                return tuple(range(self.dependent_ndim)[::-1])
            else:
                raise ValueError(
                    '"none" argument must be one of '
                    '"none", "flatten", "first", "last", "all", or "transpose"'
                )
        else:
            # Let non-negative axis requests pass through, correct negative ones to be
            # the right positive number (dependent dimensions only)
            try:
                # First the cases where axis is a sequency of axes
                return tuple(
                    a if a >= 0 else self.ndim + a - self.independent_ndim for a in axis
                )
            except TypeError:
                # Otherwise axis is a scalar
                return axis if axis >= 0 else self.ndim + axis - self.independent_ndim

    # def _slice_axis(self, axis, s, none="none"):
    #     """Return a key that has full slices for all axes, but s for axis"""
    #     axis = self.get_jaxis(axis, none=none)
    #     if axis is None:
    #         raise ValueError("Axis cannot be None in this context")
    #     return [slice(None)] * axis + [s] + [slice(none)] * (self.ndim - axis - 1)

    def _prepare_premultiply_diagonal(
        self,
        diagonal: ArrayLike | GenericUnit,
    ) -> tuple[ArrayLike | None, GenericUnit, tuple[int]]:
        """Called by child classes to set up for a premultiply_diagonal.  It works out the
        units issues and sets up for broadcasting.

        Parameters
        ----------
        diagonal : ArrayLike | GenericUnit
            Term to multiply the diagonal of this Jacobian by

        Returns
        -------
        diagonal_values : ArrayLike
            The diagonal as an array (stripped of units etc.)
        dependent_unit : Unit-like (pint/astropy/Unitless)
            The dependent unit for the result
        dependent_shape: tuple[int]
            The dependent shape for the result
        """
        from astropy import units
        import pint
        from .dual_helpers import isunit

        # If we're actually mulitplying by a unit we have a different set of returns
        if isunit(diagonal):
            return None, self.dependent_unit * diagonal, self.dependent_shape
        # Otherwise, setup to mutiply by this diagonal
        if isinstance(diagonal, units.Quantity):
            dependent_unit = diagonal.unit * self.dependent_unit
            diag_ = diagonal.value
        elif isinstance(diagonal, pint.Quantity):
            dependent_unit = diagonal.units * self.dependent_unit
            diag_ = diagonal.magnitude
        else:
            dependent_unit = self.dependent_unit
            diag_ = diagonal
        # OK, we've worked out the units, now attend to shape
        try:
            dependent_shape = broadcasted_shape(self.dependent_shape, diag_.shape)
        except AttributeError:
            dependent_shape = self.dependent_shape
        return diag_, dependent_unit, dependent_shape

    @abstractmethod
    def reshape(
        self,
        new_dependent_shape: tuple,
        order: str,
        parent_flags,
    ) -> BaseJacobian:
        """Stub for reshaping a Jacobian (dependent shape only)

        Parameters
        ----------
        new_shape : tuple
            The new shape
        order : str
            "F", "C", "A", see documentation for numpy
        parent_flags : numpy.ndarray.flags
            The flags for the parent quantity for which these are jacobians

        Returns
        -------
        BaseJacobian
            Result
        """

    def flatten(
        self,
        order: str,
        parent_flags,
    ) -> BaseJacobian:
        """Flatten the Jacobian

        Parameters
        ----------
        order : str
            "F", "C", "A", see documentation for numpy
        parent_flags : numpy.ndarray.flags
            The flags for the parent quantity for which these are jacobians

        Returns
        -------
        BaseJacobian
            Result
        """
        return self.reshape(
            new_dependent_shape=(self.dependent_size,),
            order=order,
            parent_flags=parent_flags,
        )

    def ravel(
        self,
        order: str,
        parent_flags,
    ) -> BaseJacobian:
        """Dependent variable has been ravelled, make Jacobians match

        Parameters
        ----------
        order : str
            "F", "C", "A", see documentation for numpy
        parent_flags : numpy.ndarray.flags
            The flags for the parent quantity for which these are jacobians

        Returns
        -------
        BaseJacobian
            Result
        """
        if order == "K":
            order = "A"
        reverse = (order == "C" and not parent_flags.c_contiguous) or (
            order == "F" and not parent_flags.f_contiguous
        )
        if reverse:
            raise NotImplementedError("F-contiguous dlarrays have not been tested")
            # input_jacobian = self.transpose(None, self.dependent_shape[::-1])
        else:
            input_jacobian = self
        # We'll retain C ordering for the independent variable
        return input_jacobian.reshape(
            new_dependent_shape=(input_jacobian.dependent_size,),
            order="C",
            parent_flags=parent_flags,
        )

    def nan_to_num(
        self,
        copy: bool = True,  # pylint: disable=redefined-outer-name
        nan: int | float = 0.0,
        posinf: int | float = None,
        neginf=int | float,
    ) -> BaseJacobian:
        """Implements nan_to_num for Jacobians

        Parameters
        ----------
        copy : bool, optional
            Whether to create a copy of `x` (True) or to replace values in-place
            (False). The in-place operation only occurs if casting to an array does not
            require a copy. Default is True.

        nan : int, float, optional
            Value to be used to fill NaN values. If no value is passed then NaN values
            will be replaced with 0.0.

        posinf : int, float, optional
           Value to be used to fill positive infinity values. If no value is passed
            then positive infinity values will be replaced with a very large number.

        neginf : int, float, optional
            Value to be used to fill negative infinity values. If no value is passed
            then negative infinity values will be replaced with a very small (or
            negative) number.

        Returns
        -------
        result : BaseJacobian
        """
        return type(self)(
            template=self,
            source=np.nan_to_num(
                self.data,
                copy=copy,
                nan=nan,
                posinf=posinf,
                neginf=neginf,
            ),
        )

    def to(self, unit: GenericUnit) -> BaseJacobian:
        """Change the dependent_unit for a Jacobian

        Parameters
        ----------
        unit : GenericUnit
            Unit to set the dependent unit to

        Returns
        -------
        result : BaseJacobian
            Jacobian with new dependent_unit
        """
        if unit == self.dependent_unit:
            return self
        scale = get_unit_conversion_scale(self.dependent_unit, unit)
        return self.scalar_multiply(scale)

    def to_dense(self) -> DenseJacobian:
        """Return a dense version of self"""
        from .dense_jacobians import DenseJacobian

        return DenseJacobian(source=self)

    def to_sparse(self) -> SparseJacobian:
        """Retrun a sparse version of self"""
        from .sparse_jacobians import SparseJacobian

        return SparseJacobian(source=self)

    def scalar_multiply(self, scale: float | int) -> BaseJacobian:
        """Multiply Jacobian by a scalar"""
        magnitude, units = get_magnitude_and_unit(scale)
        return type(self)(
            template=self,
            source=self.data * magnitude,
            dependent_unit=self.dependent_unit * units,
        )

    def independents_compatible(self, other: "BaseJacobian") -> bool:
        """Return true if the independent variables for two jacobians are compatible"""
        if self.independent_shape != other.independent_shape:
            return False
        if self.independent_unit != other.independent_unit:
            return False
        return True

    def _preprocess_getsetitem_key(self, key: tuple | int) -> tuple:
        """Get key into the right shape for the dependent variable.

        Parameters
        ----------
        key : tuple or int
            The argument to getitem

        Returns
        -------
        result : tuple
            The key in the correct shape for the dependent variable

        Raises
        ------
        ValueError
            If there is some unepxected argument in key
        """
        # Most likely key is a tuple
        if isinstance(key, tuple):
            # If it's too short, then add a ..., unless we already have one, in which
            # case we're fine.
            if len(key) < self.dependent_ndim and Ellipsis not in key:
                key = key + (Ellipsis,)
            # If it's too long, then we have to hope that that is because the user has
            # added some np.newaxis terms.
            elif len(key) > self.dependent_ndim:
                n_extra = len(key) - self.dependent_ndim
                if sum(x is np.newaxis or x is Ellipsis for x in key) != n_extra:
                    raise ValueError(
                        "Dual key for getitem/setitem has extra entries "
                        "that are not np.newaxis (i.e., not None) or Ellipsis"
                    )
            # Otherwise, it's fine as is.
        else:
            if self.dependent_ndim > 1:
                key = (key, Ellipsis)
            else:
                # Don't say "tuple(key)" here, as it will convert a list to a tuple,
                # which is not what we want.
                key = (key,)
        return key

    def matrix_multiply(self, other: "BaseJacobian"):
        """Matrix multiply Jacobian with another (other on the right)"""
        from .dense_jacobians import DenseJacobian
        from .sparse_jacobians import SparseJacobian
        from .diagonal_jacobians import DiagonalJacobian

        # Check that the dimensions and units are agreeable
        if self.independent_shape != other.dependent_shape:
            raise ValueError("Shape mismatch for dense Jacobian matrix multiply")
        if self.independent_unit != other.dependent_unit:
            raise ValueError("Units mismatch for dense Jacobian matrix multiply")
        # Recast any diagonal Jacobians into sparse
        final_self = self
        if isinstance(self, DiagonalJacobian):
            final_self = SparseJacobian(self)
        if isinstance(other, DiagonalJacobian):
            other = SparseJacobian(other)
        # Decide what our result type will be
        if isinstance(final_self, SparseJacobian) and isinstance(other, SparseJacobian):
            result_type = SparseJacobian
        else:
            result_type = DenseJacobian
        # OK, do the matrix multiplication
        #
        # pylint: disable-next=protected-access
        result_data_2d = final_self._get_data_2d() @ other._get_data_2d()
        # Work out its true shape
        result_shape = self.dependent_shape + other.independent_shape
        if result_type is DenseJacobian:
            result_data = result_data_2d.reshape(result_shape)
        elif result_type is SparseJacobian:
            result_data = result_data_2d
        else:
            assert False, f"Should not have gotten here (result_type={result_type})"
        return result_type(
            source=result_data,
            dependent_shape=self.dependent_shape,
            independent_shape=other.independent_shape,
            dependent_unit=self.dependent_unit,
            independent_unit=other.independent_unit,
        )

    # Potentially temporary guard to get round some issues with pint.
    @property
    def dependent_unit(self) -> GenericUnit:
        """Wrap access to dependent_unit as a poperty"""
        return self._dependent_unit

    @dependent_unit.setter
    def dependent_unit(self, value: GenericUnit):
        """Setter for dependent_unit

        Needed because there are times when pint seems to try to set a unit to a
        (hopefully unit) scalar times a unit.  Fix those.
        """
        import pint

        if isinstance(value, pint.Quantity):
            warnings.warn("Got non-unit for Unit")
            if value.magnitude != 1.0:
                raise ValueError("Cannot handle non-one unit")
            value = value.units
        self._dependent_unit = value


class Base2DExtractor:
    """A base class for n-d to 2-d extractor"""

    def __init__(self, jacobian: BaseJacobian):
        # Store a set of dummy 1D arrays that are the length of each axis
        self.jacobian = jacobian
        self.dummy_arrays = [np.zeros(shape=[n], dtype=int) for n in jacobian.shape]

    @abstractmethod
    def __getitem__(self, key) -> ArrayLike:
        pass

    def preprocess_key(self, key):
        """Preprocess and validate the key passed to a 2D extractor

        The key can only contain slices, and ints (for now at least, no smart indexing
        or Ellipsis.)
        """
        if len(key) != self.jacobian.ndim:
            raise ValueError(f"Two-D extract has wrong key length")
        result_dependent_size = 1
        result_independent_size = 1
        for axis in range(self.jacobian.ndim):
            this_key = key[axis]
            if isinstance(this_key, (int, np.integer)):
                # So, we're just extracting a single index from this Jacobian (perhaps
                # this is the block index for a gather/scatter in moepy)
                pass
            elif isinstance(this_key, slice):
                # Slice the dummy result in the same way
                dummy_result = self.dummy_arrays[axis][this_key]
                # Augment either the dependent or independent size of the result by this
                # length.
                if axis < self.jacobian.dependent_ndim:
                    result_dependent_size *= len(dummy_result)
                else:
                    result_independent_size *= len(dummy_result)
        # Return shape for result
        return [result_dependent_size, result_independent_size]
