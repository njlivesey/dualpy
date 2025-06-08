"""The class for diagonal jacobians"""

from __future__ import annotations
from typing import Optional

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .base_jacobian import BaseJacobian
from .config import get_config
from .dense_jacobians import DenseJacobian
from .dual_helpers import apply_units, has_jacobians
from .jacobian_helpers import GenericUnit
from .sparse_jacobians import SparseJacobian

__all__ = ["DiagonalJacobian", "SeedJacobian"]


class DiagonalJacobian(BaseJacobian):
    """A class for storing "diagonal" Jacobians

    These are the types where there are no cross-derivatives.  We're just storing
    dy_i/dx_i where y and x have the same shape."""

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
        """Create and populate a DiagonalJacobian

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
            source=source,
            dependent_unit=dependent_unit,
            independent_unit=independent_unit,
            dependent_shape=dependent_shape,
            independent_shape=independent_shape,
            dtype=dtype,
        )
        # Now check that the shapes are identical, the definition of a diagonal Jacobian
        if self.dependent_shape != self.independent_shape:
            raise ValueError("Attempt to create a diagonal Jacobian that is not square")
        # Now try to get the data for this Jacobian into the right shape of array
        data = None
        if source is not None:
            if isinstance(source, BaseJacobian):
                if isinstance(source, DiagonalJacobian):
                    # If source is another DiagonalJacobian, get the data directly from that
                    data = source.data
                else:
                    raise ValueError(
                        "Cannot source a DiagonalJacobian from another subclass of BaseJacobian"
                    )
            else:
                # Otherwise, try to get it from source.  First densify it if it's some kind of
                # sparse representation.
                try:
                    data = source.todense()
                except AttributeError:
                    data = source
                # Now try to reshape it.
                try:
                    data = np.reshape(data, self.dependent_shape)
                except AttributeError:
                    # If that fails then it's probably not any kind of array, perhaps it's a
                    # scalar or something, turn it into an ndarray.
                    data = np.reshape(np.array(data), self.dependent_shape)
        # If we weren't able to get anywhere with data, make it an array of zeros.
        if data is None:
            data = get_config("default_zero_array_type")(
                shape=self.dependent_shape, dtype=self.data.dtype
            )
        # OK, lodge data in self
        self.data = data
        # Check out the Jacobian to make sure everything is as it should be
        if get_config("check_jacobians"):
            self._check()

    def get_data_nd(self, form: str = None) -> ArrayLike:
        """Return the n-dimensional array of data in self"""
        if form is None or form == "dense":
            return np.diag(self.data.ravel()).reshape(self.shape)
        elif form == "sparse":
            return self.to_sparse().get_data_nd()
        else:
            raise ValueError(f"Invalid value for form argument: {form}")

    def get_data_2d(self, form: str = None) -> ArrayLike:
        """Return a 2-dimensional array of data in self"""
        if form is None or form == "dense":
            return self.to_dense().get_data_2d()
        elif form == "sparse":
            return self.to_sparse().get_data_2d()
        else:
            raise ValueError(f"Invalid value for form argument: {form}")

    def get_data_diagonal(self) -> ArrayLike:
        """Return array of data along the diagonal"""
        return self.data

    def _check(
        self,
        name: str = None,
        jname: str = None,
        dependent_shape: tuple[int] = None,
        dependent_unit=None,
    ):
        """Integrity chescks on diagonal Jacobian"""
        if name is None:
            name = "<unknown diagonal Jacobian>"
        super()._check(
            name=name,
            jname=jname,
            dependent_shape=dependent_shape,
            dependent_unit=dependent_unit,
        )
        assert isinstance(
            self.data, (np.ndarray, np.float64)
        ), f"Incorrect type for data array in {name}, {type(self.data)}"
        assert self.dependent_shape == self.independent_shape, (
            f"Non-diagonal shape for diagonal Jacobian {name}"
            f"{self.dependent_shape} != {self.independent_shape}"
        )
        assert self.dependent_size == self.independent_size, (
            f"Non-diagonal size for diagonal Jacobian {name}"
            f"{self.dependent_size} != {self.independent_size}"
        )
        assert self.data.shape == self.dependent_shape, (
            f"Array shape mismatch for {name}, "
            f"{self.data.shape} != {self.dependent_shape}"
        )

    def to_non_diagonal(self) -> DenseJacobian | SparseJacobian:
        """This function takes a DiagonalJacobian that is about to become non-diagonal and
        changes it to another type.  Typically this is a SparseJacobian, but if the
        independent size is small, then DenseJacobian is used instead."""
        if self.independent_size <= 3:
            return DenseJacobian(self)
        else:
            return SparseJacobian(self)

    def _getjitem(
        self,
        new_dependent_shape: tuple,
        key: tuple | ArrayLike,
    ) -> BaseJacobian:
        """Enacts getitem for diagonal Jacobians

        Invoked by dlarray.__getitem__

        Parameters
        ----------
        new_shape : tuple
            The new shape of the dependent variable in the

        key : tuple | ArrayLike
            The getitem key

        Returns
        -------
        result : SparseJacobian
            Appropriate subset of the Jacobian
        """
        # OK, once we extract items, this will no longer be diagonal, so we convert to
        # sparse/dense before doing the subset.
        #
        # pylint: disable=protected-access
        return self.to_non_diagonal()._getjitem(new_dependent_shape, key)

    def _setjitem(
        self,
        key: tuple | ArrayLike,
        value: BaseJacobian,
    ):
        """A setitem type method for diagonal Jacobians"""
        # OK, once we insert items, this will no longer be diagonal, so we convert to
        # sparse/dense before doing the subset
        #
        raise TypeError("Cannot enact setitem on a DiagonalJacobian")

    def get_2d_extractor(self):
        """Provides a way to use getindex directly on the Jacobian returning 2D"""
        return self.to_non_diagonal().get_2d_extractor()

    def broadcast_to(self, new_dependent_shape: tuple) -> BaseJacobian:
        """Broadcast diagonal jacobian to new dependent shape"""
        # OK, once you broadcast a diagonal, it is not longer, strictly speaking, a
        # diagonal So, convert to sparse/dense and broadcast that.  However, don't
        # bother doing anything if there is no actual broadcast going on.
        if new_dependent_shape == self.dependent_shape:
            return self
        return self.to_non_diagonal().broadcast_to(new_dependent_shape)

    def reshape(
        self,
        new_dependent_shape: tuple,
        order: str,
        parent_flags,
    ):
        """Reshape a DiagonalJacobian

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
        # OK, once you reshape a diagonal, it is not longer, strictly speaking, a
        # diagonal So, convert to sparse/dense and reshape that.  However, don't bother
        # doing anything if there is no actual reshape going on.
        if new_dependent_shape == self.dependent_shape:
            return self
        return self.to_non_diagonal().reshape(
            new_dependent_shape,
            order,
            parent_flags,
        )

    def premultiply_diagonal(self, diagonal: ArrayLike) -> DiagonalJacobian:
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
        result : DiagonalJacobian
            The resulting DenseJacobian
        """
        (
            diagonal_data,
            dependent_unit,
            dependent_shape,
        ) = self._prepare_premultiply_diagonal(diagonal)
        # Do something special if we're just multiplying by a unit
        if diagonal_data is None:
            return DiagonalJacobian(self, dependent_unit=dependent_unit)
        # If the diagonal is same shape as our independent shape (in retrospect I don't
        # see how it can't be.)  Perhaps it's something to do with the scalar case.
        if dependent_shape == self.independent_shape:
            return DiagonalJacobian(
                diagonal_data * self.data,
                template=self,
                dependent_unit=dependent_unit,
                dependent_shape=dependent_shape,
            )
        else:
            # Otherwise convert to the favored non-diagonal Jacobian (again, don't see
            # how we end up here).
            raise NotImplementedError(
                "Don't understand how cases can end up here, but retaining"
            )
            # return self.to_non_diagonal().premultiply_diagonal(diagonal)

    def insert(
        self,
        obj: int | slice | Sequence[int],
        values: ArrayLike,
        axis: int,
        dependent_shape: tuple,
    ) -> DiagonalJacobian:
        """Performms numpy.insert-like actions on DiagonalJacobian

        Parameters
        ----------
        arr : array_like
            Input array.
        obj : int, slice or sequence of ints
            Object that defines the index or indices before which `values` is
            inserted.

            .. versionadded:: 1.8.0

            Support for multiple insertions when `obj` is a single scalar or a
            sequence with one element (similar to calling insert multiple
            times).
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
        out : ndarray
            A copy of `arr` with `values` inserted.  Note that `insert`
            does not occur in-place: a new array is returned. If
            `axis` is None, `out` is a flattened array.
        """
        # By construction this is no longer diagonal once inserted to
        # change to sparse/dense and insert there.
        if has_jacobians(values):
            raise NotImplementedError(
                "Not implemented the case where a a dual is being inserted"
            )
        return self.to_non_diagonal().insert(
            obj=obj,
            values=values,
            axis=axis,
            dependent_shape=dependent_shape,
        )

    def _reduce(
        self,
        function: callable,
        new_dependent_shape: tuple,
        **kwargs,
    ) -> DenseJacobian:
        """A generic reduction operation for DiagonalJacobians

        Parameters
        ----------
        function : callable
            The operation to perform (e.g., np.sum)
        All other arguments as the various numpy reduce operations.
        """
        # Any reduction turns the Jacobian to non-diagonal, so do that first and invoke
        # the non-diagonal reduction.
        #
        # This is easier than for the dense and other cases because the Jacobian has the
        # same dimensionality as the dependent and independent variables.
        #
        # pylint: disable=protected-access
        return self.to_non_diagonal()._reduce(
            function=function, new_dependent_shape=new_dependent_shape, **kwargs
        )

    def sum(self, new_dependent_shape: tuple, **kwargs) -> BaseJacobian:
        """Performs sum for the diagonal Jacobians.  See numpy help for details"""
        return self._reduce(
            function=np.sum, new_dependent_shape=new_dependent_shape, **kwargs
        )

    def mean(self, new_dependent_shape: tuple, **kwargs) -> BaseJacobian:
        """Performs mean for the diagonal Jacobians.  See numpy help for details"""
        return self._reduce(
            function=np.mean, new_dependent_shape=new_dependent_shape, **kwargs
        )

    def cumsum(
        self,
        new_dependent_shape: tuple,
        # pylint: disable-next=unused-argument
        strategy: Optional[str] = None,
        **kwargs,
    ) -> BaseJacobian:
        """Performs cumsum for the diagonal Jacobians.  See numpy help for details"""
        return self._reduce(
            function=np.cumsum, new_dependent_shape=new_dependent_shape, **kwargs
        )

    # pylint: disable=protected-access
    def diff(
        self,
        dependent_shape,
        n: int = 1,
        axis: int | tuple = -1,
        prepend=np._NoValue,
        append=np._NoValue,
    ) -> BaseJacobian:
        """diff method for diagonal jacobian"""
        # Again, the result will not be diagonal, so change to sparse/dense and do diff
        # in that space.
        return self.to_non_diagonal().diff(
            dependent_shape=dependent_shape,
            n=n,
            axis=axis,
            prepend=prepend,
            append=append,
        )

    def transpose(self, axes, result_dependent_shape):
        """Transpose DenseJacobian in response to transpose of parent dual"""
        # This is simpler than the Dense case as the Jacobian matrix is the same shape
        # as the dependent and independent variables.
        return DiagonalJacobian(
            source=self.data.transpose(axes),
            template=self,
            dependent_shape=result_dependent_shape,
        )

    def tensordot(self, other: BaseJacobian, axes: tuple, dependent_unit: GenericUnit):
        """Compute self(.)other"""
        # Once we do this, we will no longer be diagonal, so convert to sparse/dense
        return self.to_non_diagonal().tensordot(other, axes, dependent_unit)

    def rtensordot(self, other: BaseJacobian, axes: tuple, dependent_unit: GenericUnit):
        """Compute self(.)other"""
        # Once we do this, we will no longer be diagonal, so convert to sparse/dense
        return self.to_non_diagonal().rtensordot(other, axes, dependent_unit)

    def diagonal(self):
        """Get diagonal elements (shape=dependent_shape)"""
        return apply_units(self.data, (self.dependent_unit / self.independent_unit))

    # The reaons we have extract_diagonal and diagonal is that diagonal is only
    # populated for diagonal Jacobians.  extract_diagonal is populated for all.
    def extract_diagonal(self):
        """Extract the diagonal from a diagonal Jacobian"""
        return self.diagonal()

    def toarray(self):
        """Get self's data as a non-dense ndarray"""
        return self.to_sparse().toarray()

    def todensearray(self):
        """Get self's data as dense ndarray"""
        unit = self.dependent_unit / self.independent_unit
        return apply_units(self.get_data_nd(), unit)

    def to2darray(self):
        """Get self's data as (dense) 2d array"""
        unit = self.dependent_unit / self.independent_unit
        return apply_units(self.get_data_2d(), unit)

    def to2ddensearray(self):
        """Get self's data as dense 2d array"""
        return self.to2darray()

    def linear_interpolator(self, x_in, axis=-1, extrapolate=None):
        """Return an interpolator for a given Jacobian axis"""
        return self.to_non_diagonal().linear_interpolator(
            x_in=x_in,
            axis=axis,
            extrapolate=extrapolate,
        )

    def spline_interpolator(
        self, x_in, axis=-1, bc_type="not_a_knot", extrapolate=None
    ):
        """Return an interpolator for a given Jacobian axis"""
        return self.to_non_diagonal().spline_interpolator(
            x_in=x_in,
            axis=axis,
            bc_type=bc_type,
            extrapolate=extrapolate,
        )


class SeedJacobian(DiagonalJacobian):
    """A specific subclass of DiagonalJacobian for seeds

    There is no substantive difference.  The only intent is to indicate that this is an
    actual seed (it's only use thus far is to help the numeric Jacobian computation code
    identify what are the seeded variables so it can perturb them.

    """
