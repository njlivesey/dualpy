"""Class for dense jacobians"""

from __future__ import annotations

from typing import Optional

from collections.abc import Sequence

import numpy as np
import scipy.interpolate as interpolate
from numpy.typing import ArrayLike, DTypeLike

from .base_jacobian import BaseJacobian, Base2DExtractor
from .config import get_config
from .dual_helpers import apply_units, has_jacobians
from .jacobian_helpers import (  # array_to_sparse_diagonal,
    linear_interpolation_indices_and_weights,
    prepare_jacobians_for_binary_op,
    GenericUnit,
)

__all__ = [
    "DenseJacobian",
    "DenseJacobianLinearInterpolator",
    "DenseJacobianSplineInterpolator",
]


class DenseJacobian(BaseJacobian):
    """A dljacobian that's a full-on ndarray"""

    def __init__(
        self,
        source: BaseJacobian | ArrayLike = None,
        template: BaseJacobian = None,
        dependent_unit: GenericUnit = None,
        independent_unit: GenericUnit = None,
        dependent_shape: Sequence = None,
        independent_shape: Sequence = None,
        dtype: DTypeLike = None,
    ):
        """Create and populate a DenseJacobian

        Parameters
        ----------
        template : BaseJacobian or child thereof, optional
            If supplied, can be source of shape and units information
        source : ArrayLike or another Jacobian
            Source for Jacobian data
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
        # First, establish the shape, units, etc. of this Jacobian.  See if we can get
        # it from a template (or from source if it is itself a Jacobian.)
        if isinstance(source, BaseJacobian):
            if template is None:
                template = source
            else:
                raise ValueError("Got a Jacobian in source and also a template")
        # OK, now set up the key parameters based on the information in template, if
        # any, and the keword arguments (depdendent_shape etc. etc.)
        super().__init__(
            template=template,
            dependent_unit=dependent_unit,
            independent_unit=independent_unit,
            dependent_shape=dependent_shape,
            independent_shape=independent_shape,
            dtype=dtype,
        )
        # Now try to get the data for this Jacobian, an n-dimensional array
        data = None
        if source is not None:
            if isinstance(source, BaseJacobian):
                # If source is a Jacobian, then get it from that
                data = source.get_data_nd(form="dense")
            else:
                # Otherwise, try to get it from source.  First densify it if it's some kind of
                # sparse representation.
                try:
                    data = np.array(source.todense())
                except AttributeError:
                    data = source
                # Now try to reshape it.
                try:
                    data = np.reshape(data, self.shape)
                except AttributeError:
                    # If that fails then it's probably not any kind of array, perhaps it's a
                    # scalar or something, turn it into an ndarray.
                    data = np.reshape(np.array(source), self.shape)
        # If we weren't able to get anywhere with data, make it an array of zeros.
        if data is None:
            data = get_config("default_zero_array_type")(
                shape=self.shape, dtype=self.data.dtype
            )
        # OK, lodge data in self
        self.data = data
        # Check out the Jacobian to make sure everything is as it should be
        if get_config("check_jacobians"):
            self._check()

    def get_data_nd(self, form: str = None) -> ArrayLike:
        """Return the n-dimensional array of data in self"""
        if form is None or form == "dense":
            return self.data
        elif form == "sparse":
            raise TypeError("Unable to return sparse form of DenseJacobian data")
        else:
            raise ValueError(f"Invalid value for form argument: {form}")

    def get_data_2d(self, form: str = None) -> ArrayLike:
        """Return the 2-dimensional array of the data in self"""
        if form is None or form == "dense":
            return np.reshape(self.data, self.shape_2d)
        elif form == "sparse":
            raise TypeError("Unable to return 2D-sparse form of DenseJacobian data")
        else:
            raise ValueError(f"Invalid value for form argument: {form}")

    def get_data_diagonal(self) -> ArrayLike:
        """Return the diagonal form of the array in self"""
        raise TypeError("Unable to return diagonal from DenseJacobian")

    def get_2d_view(self) -> ArrayLike:
        """Returns a 2-dimensional view into the array of the data in self"""
        return self.data.view().reshape(self.shape_2d)

    def _check(
        self,
        name: str = None,
        jname: str = None,
        dependent_shape: tuple[int] = None,
        dependent_unit=None,
    ):
        """Integrity checks on dense Jacobian

        Parameters
        ----------
        name : str, optional
            Name for the Jacobian for use in error messages
        """
        if name is None:
            name = "<unknown dense Jacobian>"
        # Get the base class to do the main checking
        super()._check(
            name=name,
            jname=jname,
            dependent_shape=dependent_shape,
            dependent_unit=dependent_unit,
        )
        assert isinstance(
            self.data, (np.ndarray, np.generic)
        ), f"Incorrect type for data array in {name}, {type(self.data)}"
        assert (
            self.data.shape == self.shape
        ), f"Array shape mismatch for {name}, {self.data.shape} != {self.shape}"
        assert (
            self.data.size == self.size
        ), f"Array size mismatch for {name}, {self.data.size} != {self.size}"

    def _getjitem(
        self,
        new_dependent_shape: tuple,
        key: tuple | ArrayLike,
    ) -> DenseJacobian:
        """Enacts getitem for dense Jacobians

        Invoked by dlarray.__getitem__

        Parameters
        ----------
        new_shape : tuple
            The new shape of the dependent variable in the

        key : tuple | ArrayLike
            The getitem key

        Returns
        -------
        result : DenseJacobian
            Appropriate subset of the Jacobian
        """
        # Do some preprocessing on the key (basically ensures is the right shape, as
        # sometimes, things like matplotlib add Ellipses etc.)
        key = self._preprocess_getsetitem_key(key)
        # Append full selections for the independent axes
        jkey = key + (slice(None),) * self.independent_ndim
        result_data = self.data[jkey]
        new_full_shape = new_dependent_shape + self.independent_shape
        if result_data.shape != new_full_shape:
            raise NotImplementedError(
                "Looks like we do need to do a reshape after all!"
            )
            # result_ = np.reshape(result_, new_full_shape)
        return DenseJacobian(
            source=result_data, template=self, dependent_shape=new_dependent_shape
        )

    def _setjitem(
        self,
        key: tuple | ArrayLike,
        value: BaseJacobian,
    ):
        """Enacts setitem for dense Jacobians

        Invoked by dlarray.__setitem___

        Parameters
        ----------
        key : tuple | ArrayLike
            The setitem key
        value : BaseJacobian
            The values for this part of the Jacobian to be set to
        """
        if value is not None:
            # Try to get the "output" and "input" Jacobians to be compatible, as we
            # would for a binary operation.
            dummy_self_data, value_data, result_type = prepare_jacobians_for_binary_op(
                self, value
            )
            if result_type != type(self):
                raise TypeError(
                    "Jacobian is not of correct type to receive new contents"
                )
        else:
            raise NotImplementedError("Not sure why we ever need to get here.")
            # value_data = 0.0
        key = self._preprocess_getsetitem_key(key)
        # Do the value insetion
        self.data[key] = value_data

    def get_2d_extractor(self):
        """Provides a way to use getindex directly on the Jacobian returning 2D"""
        return Dense2DExtractor(self)

    def broadcast_to(self, new_dependent_shape: tuple) -> DenseJacobian:
        """Broadcast dense Jacobian to new dependent_shape"""
        # Don't bother doing anything if the shape is already good
        if new_dependent_shape == self.dependent_shape:
            return self
        full_shape = new_dependent_shape + self.independent_shape
        data = np.broadcast_to(self.data, full_shape)
        return DenseJacobian(
            source=data,
            template=self,
            dependent_shape=new_dependent_shape,
        )

    def reshape(
        self,
        new_dependent_shape: tuple,
        order: str,
        parent_flags,
    ) -> DenseJacobian:
        """Reshape a DenseJacobian

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
        DenseJacobian
            Result
        """
        # Don't bother doing anything if the shape is already good
        if new_dependent_shape == self.dependent_shape:
            return self
        if order == "K":
            order = "A"
        reverse = (order == "C" and not parent_flags.c_contiguous) or (
            order == "F" and not parent_flags.f_contiguous
        )
        if reverse:
            input_jacobian = self.transpose(None, self.dependent_shape[::-1])
        else:
            input_jacobian = self
        try:
            full_shape = tuple(new_dependent_shape) + tuple(
                input_jacobian.independent_shape
            )
        except TypeError:
            full_shape = (new_dependent_shape,) + tuple(
                input_jacobian.independent_shape
            )
        data = np.reshape(input_jacobian.data, full_shape, order)
        return DenseJacobian(
            source=data, template=input_jacobian, dependent_shape=new_dependent_shape
        )

    def premultiply_diagonal(self, diagonal: ArrayLike) -> DenseJacobian:
        """Diagonal premulitply for dense Jacobian

        This is a key routine as the vast majority of dual operations involve a
        chain-rule-invoked premultiplication along the dependent axis.  In this case
        diagonal needs to be broadcastable to self.dependent_shape, in which case a simple
        element-wise multiply accomplishes what is needed.

        Parameters
        ----------
        diagonal : ArrayLike
            The array to multiply down the diagonal

        Returns
        -------
        result : DenseJacobian
            The resulting DenseJacobian
        """
        (
            diagonal_data,
            dependent_unit,
            dependent_shape,
        ) = self._prepare_premultiply_diagonal(diagonal)
        # Special case if we're just multiplying by a unit
        if diagonal_data is None:
            return DenseJacobian(self, dependent_unit=dependent_unit)
        try:
            diagonal_data = np.reshape(
                diagonal_data, (diagonal.shape + self._dummy_independent)
            )
            # This will fail for scalars, but that's OK scalars don't
            # need to be handled specially
        except (ValueError, AttributeError):
            pass
        return DenseJacobian(
            diagonal_data * self.data,
            template=self,
            dependent_unit=dependent_unit,
            dependent_shape=dependent_shape,
        )

    def insert(
        self,
        obj: int | slice | Sequence[int],
        values: ArrayLike,
        axis: int,
        dependent_shape: tuple,
    ) -> DenseJacobian:
        """Performms numpy.insert-like actions on DenseJacobian

        Parameters
        ----------
        arr : array_like
            Input array.
        obj : int, slice or sequence of ints
            Object that defines the index or indices before which `values` is
            inserted.
        values : array_like
            Values to insert into `arr`. If the type of `values` is different
            from that of `arr`, `values` is converted to the type of `arr`.
            `values` should be shaped so that ``arr[...,obj,...] = values``
            is legal.
        axis : int, optional
            Axis along which to insert `values`.  If `axis` is None then `arr`
            is flattened first.

        Returns
        -------
        out : SparseJacobian
            A copy of `arr` with `values` inserted.  Note that `insert`
            does not occur in-place: a new array is returned. If
            `axis` is None, `out` is a flattened array.
        """

        if has_jacobians(values):
            raise NotImplementedError(
                "Not implemented the case where a a dual is being inserted"
            )
        jaxis = self.get_jaxis(axis, none="flatten")
        data = np.insert(self.data, obj, 0.0, jaxis)
        return DenseJacobian(data, template=self, dependent_shape=dependent_shape)

    def _reduce(
        self,
        function: callable,
        new_dependent_shape: tuple,
        axis: int = None,
        **kwargs,
    ) -> DenseJacobian:
        """A generic reduction operation for DenseJacobians

        Parameters
        ----------
        function : callable
            The operation to perform (e.g., np.sum)
        All other arguments as the various numpy reduce operations.
        """
        jaxis = self.get_jaxis(axis, none="all")
        return DenseJacobian(
            source=function(self.data, axis=jaxis, **kwargs),
            template=self,
            dependent_shape=new_dependent_shape,
        )

    def sum(self, new_dependent_shape: tuple, **kwargs) -> DenseJacobian:
        """Performs sum for the dense Jacobians.  See numpy help for details"""
        return self._reduce(
            function=np.sum, new_dependent_shape=new_dependent_shape, **kwargs
        )

    def mean(self, new_dependent_shape: tuple, **kwargs) -> DenseJacobian:
        """Performs mean for the dense Jacobians.  See numpy help for details"""
        return self._reduce(
            function=np.mean, new_dependent_shape=new_dependent_shape, **kwargs
        )

    def cumsum(
        self,
        new_dependent_shape: tuple,
        # pylint: disable-next=unused-argument
        strategy: Optional[str] = None,
        **kwargs,
    ) -> DenseJacobian:
        """Performs cumsum for the dense Jacobians.  See numpy help for details"""
        return self._reduce(
            function=np.cumsum, new_dependent_shape=new_dependent_shape, **kwargs
        )

    # pylint: disable=protected-access
    def diff(
        self, dependent_shape, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue
    ) -> DenseJacobian:
        """diff method for dense jacobian"""
        jaxis = self.get_jaxis(axis)
        if prepend is not np._NoValue:
            prepend = np.expand_dims(
                prepend,
                range(self.dependent_ndim, self.dependent_ndim + self.independent_ndim),
            )
        if append is not np._NoValue:
            append = np.expand_dims(
                append,
                range(self.dependent_ndim, self.dependent_ndim + self.independent_ndim),
            )
        result_ = np.diff(self.data, n, jaxis, prepend, append)
        return DenseJacobian(
            source=result_, template=self, dependent_shape=dependent_shape
        )

    def transpose(self, axes, result_dependent_shape) -> DenseJacobian:
        """Transpose DenseJacobian in response to transpose of parent dual"""
        jaxes = self.get_jaxis(axes, none="transpose")
        jaxes = tuple(jaxes) + tuple(range(self.dependent_ndim, self.ndim))
        return DenseJacobian(
            source=self.data.transpose(jaxes),
            template=self,
            dependent_shape=result_dependent_shape,
        )

    def tensordot(
        self, other: BaseJacobian, axes: tuple, dependent_unit: GenericUnit
    ) -> DenseJacobian:
        """Compute self(.)other"""
        import sparse as st  # pylint: disable=import-outside-toplevel

        # Note that axes here must be in the list of two lists form, with no negative
        # numbers.
        n_contractions = len(axes[0])
        # Make sure the axes are all in appropriate form

        # With this order of the tensordot, the annoying thing here is that we actually
        # want our independent dimensions (part of self) at the end, so will have to do
        # a transpose.  Let's do the tensor dot anyway.
        if isinstance(self.data, np.ndarray) and isinstance(other, np.ndarray):
            data = np.tensordot(self.data, other, axes)
        else:
            data = st.tensordot(self.data, other, axes)
        # Move the indepenent axes to the end.  First we want the non-contracted
        # dependent dimensions from self, these are currently at the start
        new_axis_order = list(range(self.dependent_ndim - n_contractions))
        # Then the non-contracted dimensions from other, currently at the end
        new_axis_order += list(
            range(data.ndim - other.ndim + n_contractions, data.ndim)
        )
        # Finally the independent dimensions, currently in the middle
        new_axis_order += list(
            range(
                self.dependent_ndim - n_contractions,
                self.dependent_ndim - n_contractions + self.independent_ndim,
            )
        )
        data = data.transpose(new_axis_order)
        # Do this next bit long hand in scale independent_ndim==0
        result_dependent_shape = data.shape[: data.ndim - self.independent_ndim]
        return DenseJacobian(
            source=data,
            dependent_shape=result_dependent_shape,
            dependent_unit=dependent_unit,
            independent_shape=self.independent_shape,
            independent_unit=self.independent_unit,
        )

    def rtensordot(
        self, other: BaseJacobian, axes: tuple, dependent_unit: GenericUnit
    ) -> DenseJacobian:
        """Compute other(.)self"""
        # This one is actually easier than regular tensordot, because the axes end up in
        # the right order
        data = np.tensordot(other, self.data, axes)
        # Do this next bit long hand in scale independent_ndim==0
        result_dependent_shape = data.shape[: data.ndim - self.independent_ndim]
        return DenseJacobian(
            source=data,
            template=self,
            dependent_shape=result_dependent_shape,
            dependent_unit=dependent_unit,
        )

    def extract_diagonal(self):
        """Extract the diagonal from a dense Jacobian with units"""
        if self.dependent_shape != self.independent_shape:
            raise ValueError("Dense Jacobian is not square")
        result = np.reshape(np.diag(self.get_data_2d()), self.dependent_shape)
        return apply_units(result, self.dependent_unit / self.independent_unit)

    def todensearray(self):
        """Get Jacobian as n-dimensional dense array with units"""
        return apply_units(self.data, self.dependent_unit / self.independent_unit)

    def to2ddensearray(self):
        """Get Jacobian as 2-dimensional dense array with units"""
        return apply_units(
            self.get_data_2d(), self.dependent_unit / self.independent_unit
        )

    def to2darray(self):
        """Get Jacobian as 2-dimensional array with units"""
        return self.to2ddensearray()

    def _join(self, other, location, axis, result_dependent_shape):
        """Insert/append dense Jacobians"""
        n = self.dependent_shape[axis]
        if location < n:
            # This is an insert
            result_ = np.insert(self.data, location, other.data, axis)
        elif location == n:
            # This is an append
            result_ = np.append(self.data, other.data, axis)
        return DenseJacobian(
            result_, template=self, dependent_shape=result_dependent_shape
        )

    def linear_interpolator(
        self,
        x_in,
        axis=-1,
        extrapolate=None,
    ):
        """Return an interpolator for a given Jacobian axis"""
        return DenseJacobianLinearInterpolator(
            self,
            x_in=x_in,
            axis=axis,
            extrapolate=extrapolate,
        )

    def spline_interpolator(
        self,
        x_in,
        axis=-1,
        bc_type="not-a-knot",
        extrapolate=None,
    ):
        """Return an interpolator for a given Jacobian axis"""
        return DenseJacobianSplineInterpolator(self, x_in, axis)


class DenseJacobianLinearInterpolator(object):
    """Interpolates a DenseJacobian along one dependent axis"""

    def __init__(self, jacobian, x_in, axis=-1, extrapolate=None):
        """Setup an interpolator for a given DenseJacobian"""
        self.jacobian = jacobian
        self.jaxis = jacobian.get_jaxis(axis, none="first")
        self.x_in = x_in
        if extrapolate == "periodic":
            raise NotImplementedError(
                "Unable to handle periodic interpolation with Jacobians (yet)"
            )
        self.extrapolate = extrapolate

    def __call__(self, x_out):
        """Inpoterpolate a DenseJacobian to new values along an axis"""
        i_lower, i_upper, w_lower, w_upper = linear_interpolation_indices_and_weights(
            self.x_in,
            x_out,
            extrapolate=self.extrapolate,
        )
        # Set up keys for indexing along the relevant axis
        empty_key = [slice(None)] * self.jacobian.ndim
        lower_key = empty_key.copy()
        upper_key = empty_key.copy()
        lower_key[self.jaxis] = i_lower
        upper_key[self.jaxis] = i_upper
        # Set up a shape for the weights that's broadcastable to the jacobian shape and
        # reshape the weights it
        w_shape = np.ones(self.jacobian.ndim, dtype=int)
        w_shape[self.jaxis] = w_lower.size
        w_lower = np.reshape(w_lower, w_shape)
        w_upper = np.reshape(w_upper, w_shape)
        # Now do the linear interpolation
        data = (
            self.jacobian.data[tuple(lower_key)] * w_lower
            + self.jacobian.data[tuple(upper_key)] * w_upper
        )
        # Prepare the result and return
        new_dependent_shape = data.shape[: self.jacobian.dependent_ndim]
        return DenseJacobian(
            template=self.jacobian,
            source=data,
            dependent_shape=new_dependent_shape,
        )


class DenseJacobianSplineInterpolator(object):
    """Interpolates Jacobian along one dependent axis"""

    def __init__(
        self,
        jacobian,
        x_in,
        axis=-1,
        bc_type="not-a-knot",
        extrapolate=None,
    ):
        """Setup an interpolator for a given DenseJacobian"""

        self.jacobian = jacobian
        self.jaxis = jacobian.get_jaxis(axis, none="first")
        self.interpolator = interpolate.CubicSpline(
            x_in,
            jacobian.data,
            axis=self.jaxis,
            bc_type=bc_type,
            extrapolate=extrapolate,
        )

    def __call__(self, x_out):
        """Inpoterpolate a DenseJacobian to new values along an axis"""
        data = self.interpolator(x_out)
        new_dependent_shape = data.shape[: self.jacobian.dependent_ndim]
        return DenseJacobian(
            template=self.jacobian,
            source=data,
            dependent_shape=new_dependent_shape,
        )


class Dense2DExtractor(Base2DExtractor):
    """A class to directly get a subset from the Jacobian"""

    # The base classes __init__ is sufficient for our purposes.

    def __getitem__(self, key):
        """Get an extract of the Jacobian and return it as a 2D matrix"""
        result_shape = self.preprocess_key(key)
        return self.jacobian.data[key].reshape(result_shape)
