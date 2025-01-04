"""Class for sparse jacobians"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from scipy import interpolate, sparse

from .base_jacobian import BaseJacobian, Base2DExtractor
from .config import get_config
from .dual_helpers import apply_units, get_magnitude_and_unit, has_jacobians
from .jacobian_helpers import (
    GenericUnit,
    array_to_sparse_diagonal,
    linear_interpolation_indices_and_weights,
    shapes_broadcastable,
)
from .sparse_helpers import (
    DenselyRearrangedSparseJacobian,
    SparselyRearrangedSparseJacobian,
    sparse_tensor_to_csc,
)

__all__ = ["SparseJacobian"]


class SparseJacobian(BaseJacobian):
    """A dual Jacobian that's stored as a scipy 2D sparse array"""

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
        # pylint: disable=import-outside-toplevel
        from .diagonal_jacobians import DiagonalJacobian
        import sparse as sparse_tensor

        # pylint: enable=import-outside-toplevel

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
        # Now try to get the data for this Jacobian, into a sparse csc_array
        if isinstance(source, BaseJacobian):
            if isinstance(source, SparseJacobian):
                data = source.data
            elif isinstance(source, DiagonalJacobian):
                data = array_to_sparse_diagonal(source.data)
            else:
                raise TypeError(f"Cannot initialize SparseJacobian from {type(source)}")
        elif isinstance(source, sparse.csc_array):
            data = source
        elif isinstance(source, sparse_tensor.SparseArray):
            data = sparse_tensor_to_csc(source, dependent_ndim=self.dependent_ndim)
        elif source is None:
            data = sparse.csc_array(self.shape_2d, dtype=self.data.dtype)
        else:
            raise TypeError(
                f"Unable to initialize SparseJacobian with source of type {type(source)}"
            )
        # Check that what we have is going to be of the right shape
        if data.shape != self.shape_2d:
            raise ValueError("Attempt to create jacobian_sparse with wrong-sized input")
        # OK, lodge data in self
        self.data = data
        # Check out the Jacobian to make sure everying is as it should be
        if get_config("check_jacobians"):
            self._check()

    @property
    def percent_full(self):
        """Returns the percentage of non-zero values"""
        return 100.0 * self.data.nnz / (self.dependent_size * self.independent_size)

    def __str__(self):
        """Provide a string summary of a sparse Jacobian"""
        percent = self.percent_full()
        suffix = f" with {self.data.nnz:,} numbers stored ({percent:.2g}%)"
        return super().__str__() + suffix

    def get_data_nd(self, form: str = None) -> ArrayLike:
        """Return the n-dimensional array of data in self"""
        if form is None or form == "sparse":
            return self.as_sparse_tensor()
        elif form == "dense":
            return np.reshape(np.array(self.data.todense()), self.shape)
        else:
            raise ValueError(f"Invalid value for form argument: {form}")

    def get_data_2d(self, form: str = None) -> ArrayLike:
        """Return 2-d (sparse) array of data in self"""
        if form is None or form == "sparse":
            return self.data
        elif form == "dense":
            return self.data.todense()
        else:
            raise ValueError(f"Invalid value for form argument: {form}")

    def get_data_diagonal(self) -> ArrayLike:
        """Return the diagonal form of the array in self"""
        raise TypeError("Not possible to get digonal from SparseJacobian")

    def _check(
        self,
        name: str = None,
        jname: str = None,
        dependent_shape: tuple[int] = None,
        dependent_unit=None,
    ):
        """Integrity checks for sparse Jacobians

        Parameters
        ----------
        name : str, optional
            Name for the Jacobian for use in error messages
        """
        if name is None:
            name = "<unknown sparse Jacobian>"
        # Get the base class to do the main checking
        super()._check(
            name=name,
            jname=jname,
            dependent_shape=dependent_shape,
            dependent_unit=dependent_unit,
        )
        # Convert to csc_array if csc_matrix
        if isinstance(self.data, sparse.csc_matrix):
            warnings.warn(
                "Found a csc_matrix, converting to csc_array", DeprecationWarning
            )
            self.data = sparse.csc_array(self.data)
        assert isinstance(
            self.data, sparse.csc_array
        ), f"Incorrect type for data array in {name}, {type(self.data)}"
        correct_2dshape = (self.dependent_size, self.independent_size)
        assert self.data.shape == correct_2dshape, (
            f"array2d shape mismatch for {name}, "
            f"{self.data.shape} != {correct_2dshape}"
        )

    def _getjitem(
        self, new_dependent_shape: tuple, key: tuple | ArrayLike
    ) -> SparseJacobian:
        """Enacts getitem for sparse Jacobians

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
        key = self._preprocess_getsetitem_key(key)
        # Now we're going to collapse all the dependent key into a 1D index array.
        # There are probably more efficient ways in which to handle this, particular
        # for the cases where key is filled entirely with slices. However, the thing
        # to remember is that there is no shame in handling things that are the
        # length of the dependent vector (or independent one come to that), just
        # avoid things that are the cartesian product of them.  Accordingly this
        # straightforward approach is probably good enough.
        i_full = np.reshape(np.arange(self.dependent_size), self.dependent_shape)
        i_subset = np.ravel(i_full[key])
        dependent_slice = (i_subset,)
        # Note, I originally tried to sidestep the null-operation for the above, but it
        # didn't work when adding np.newaxes.  Simpler just to do it all the time.
        # Append a "get the lot" slice for the independent variable dimension
        jkey = dependent_slice + (slice(None),)
        data = self.data[jkey]
        return SparseJacobian(
            source=data,
            template=self,
            dependent_shape=new_dependent_shape,
        )

    def _setjitem(
        self,
        key: tuple | ArrayLike,
        value: BaseJacobian,
    ):
        """Enacts setitem for sparse Jacobians

        Invoked by dlarray.__setitem___

        Parameters
        ----------
        key : tuple | ArrayLike
            The setitem key
        value : BaseJacobian
            The values for this part of the Jacobian to be set to
        """
        if not isinstance(value, SparseJacobian):
            raise NotImplementedError(
                f"Not implemented SparseJacobian._setjitem for {type(value)}"
            )
        # Key point to recall here is that, for each of the dependent indices in key,
        # we're replacing all of its Jacobians, not just those that are present in
        # value.  Those Jacobian elements in value that are zero should trump anything
        # present in self.  This is an overwrite (for the dependent elements in key) not
        # an add/merge.  First we're going to map between the dependent indices in self
        # and those in value, using key.
        i_dependent_full = np.reshape(
            np.arange(self.dependent_size), self.dependent_shape
        )
        i_dependent_subset = np.ravel(i_dependent_full[key])
        # Now, the dependent index is the "rows" for the sparse Jacobian.  Transform
        # both self and value into lil sparse matrices.
        self_lil = self.data.tolil()
        value_lil = value.data.tolil()
        # Replace all affected rows in self with the rows in value
        for i, (value_row, value_data) in enumerate(
            zip(value_lil.rows, value_lil.data)
        ):
            ii = i_dependent_subset[i]
            self_lil.rows[ii] = value_row
            self_lil.data[ii] = value_data
        self.data = self_lil.tocsc()

    def get_2d_extractor(self):
        """Provides a way to use getindex directly on the Jacobian returning 2D"""
        return Sparse2DExtractor(self)

    def broadcast_to(self, shape):
        """Broadcast the dependent vector part of a sparse Jacobian to another shape"""
        # Don't bother doing anything if the shape is already good
        if shape == self.dependent_shape:
            return self
        if not shapes_broadcastable(self.dependent_shape, shape):
            raise ValueError("Unable to broadcast SparseJacobian to new shape")
        # This one has to be rather inefficient as it turns out.  Down
        # the road we might be able to do something with some kind of
        # index mapping, but for now, just do a replication rather
        # than a broadcast (shame)

        # Get a 1D vector indexing original dependent vector
        i_old = np.arange(self.dependent_size)
        # Turn into an nD array
        i_old = np.reshape(i_old, self.dependent_shape)
        # Broadcast this to the new shape
        i_old = np.broadcast_to(i_old, shape)
        # Now convert to a 1D array
        i_old = np.ravel(i_old)
        # Get a matching i_new array
        i_new = np.arange(i_old.size)
        # Now put a 1 at every [i_new,i_old] point in a sparse matrix
        one = np.ones((i_old.size,), dtype=np.int64)
        broadcast_array = sparse.csc_array(
            (one, (i_new, i_old)), shape=(i_new.size, self.dependent_size)
        )
        # Now do a matrix multiply to accomplish what broadcast tries
        # to do.
        data = broadcast_array @ self.data
        return SparseJacobian(source=data, template=self, dependent_shape=shape)

    def reshape(
        self,
        new_dependent_shape: tuple,
        order: str,
        parent_flags,
    ) -> SparseJacobian:
        """Reshape a SparseJacobian

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
        SparseJacobian
            Result
        """
        # Don't bother doing anything if the shape is already good
        if new_dependent_shape == self.dependent_shape:
            return self
        reverse = (order == "C" and not parent_flags.c_contiguous) or (
            order == "F" and not parent_flags.f_contiguous
        )
        if reverse:
            input_jacobian = self.transpose(None, self.dependent_shape[::-1])
        else:
            input_jacobian = self
        # For the sparse jacobians, which never really have a shape
        # anyway, this simply involves updating the shapes of record
        # (assuming it's a valid one).
        if int(np.prod(new_dependent_shape)) != input_jacobian.dependent_size:
            raise ValueError("Unable to reshape SparseJacobian to new shape")
        return SparseJacobian(
            source=input_jacobian.data,
            template=input_jacobian,
            dependent_shape=new_dependent_shape,
        )

    def premultiply_diagonal(self, diagonal: ArrayLike) -> SparseJacobian:
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
        result : SparseJacobian
            The resulting DenseJacobian
        """
        # This is a hand-woven diagnonal/sparse-csc multiply
        # equivalent to
        #     out = (diagonal matrix) @ self
        # However, when expressed that way it's less efficient then below
        (
            diagonal_data,
            dependent_unit,
            dependent_shape,
        ) = self._prepare_premultiply_diagonal(diagonal)
        # Special case when we're just multiplying by a unit
        if diagonal_data is None:
            return SparseJacobian(self, dependent_unit=dependent_unit)
        # Do a manual broadcast if needed (which, in the case of the
        # Jacobian is actually not a real broadcast, but unfortunately
        # a somewhat wasteful replicate opration.  But since the
        # result will be the size of the broadcasted shape in any
        # case, there is no real loss.
        if dependent_shape != self.dependent_shape:
            # pylint: disable=self-cls-assignment
            self = self.broadcast_to(dependent_shape)
            diagonal_data = np.broadcast_to(diagonal_data, dependent_shape)
        diagonal_data = np.ravel(diagonal_data)

        # out_ = self.data2d.copy()
        # if np.iscomplexobj(diag_) and not np.iscomplexobj(out_):
        #     out_=out_ * complex(1, 0)
        # if len(diag_) == 1:
        #     out_.data *= diag_[0]
        # else:
        #     # out_.data *= diag_[out_.indices]
        #     out_.data *= np.take(diag_, out_.indices)

        if len(diagonal_data) == 1:
            out_data = self.data.data * diagonal_data
        else:
            out_data = self.data.data * np.take(diagonal_data, self.data.indices)
        result = sparse.csc_array(
            (out_data, self.data.indices, self.data.indptr), shape=self.data.shape
        )
        return SparseJacobian(
            source=result,
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
    ) -> SparseJacobian:
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

        # We do this simply by moving the indices around
        # First convert to a coo-based sparse array
        self_coo = self.data.tocoo()
        # OK, unravel the multi index that makes up the rows
        all_indices = list(np.unravel_index(self_coo.row, self.dependent_shape))
        axis = self.get_jaxis(axis, none="zero")
        i = all_indices[axis]
        # Convert insertion requests to list
        try:
            obj_ = obj.tolist()
        except AttributeError:
            try:
                len(obj)
                obj_ = obj
            except TypeError:
                obj_ = [obj]
        # Loop over those insertion requests
        for j in obj_:
            i[j:] += 1
        all_indices[axis] = i
        row = np.ravel_multi_index(all_indices, dependent_shape)
        dependent_size = int(np.prod(dependent_shape))
        result_csc = sparse.csc_array(
            (self_coo.data, (row, self_coo.col)),
            shape=(dependent_size, self.independent_size),
        )
        return SparseJacobian(
            template=self, source=result_csc, dependent_shape=dependent_shape
        )

    def sum(
        self,
        new_dependent_shape: tuple,
        axis: int | tuple = None,
        dtype: DTypeLike = None,
        keepdims: bool = False,
    ) -> BaseJacobian:
        """Perform sum for the sparse Jacobians"""
        # pylint: disable=import-outside-toplevel
        from .dense_jacobians import DenseJacobian

        # Two different approaches, depending on whether axis is supplied
        if axis is None:
            # OK, here we want to sum over all the dependent elements,
            # scipy.sparse can do that.
            data = np.sum(self.data, axis=0, dtype=dtype)
            # Note that result is a dense matrix
            if keepdims:
                result_shape = (1,) * self.dependent_ndim + self.independent_shape
            else:
                result_shape = self.independent_shape
            data = np.reshape(np.asarray(data), result_shape)
            return DenseJacobian(
                template=self, source=data, dependent_shape=new_dependent_shape
            )
        # Otherwise we were supplied an axis to sum over. Here we want to sum over
        # selected indices.  Recall that there is no shame in thinking about things that
        # have the same size as the dependent vector.  To that end, we will formulate
        # this as a sparse-sparse matrix multiply.  In many ways, this operation is the
        # complement of the broadcast operation, so is constructed in a similar manner.
        try:
            iter(axis)
        except TypeError:
            axis = (axis,)
        # Take the orginal shape and replace the summed-over axes with one.  In the case
        # where keepdims is set, that would be independent_shape of course, but we can't
        # rely on that.
        reduced_shape = list(self.dependent_shape)
        # Note that the below code implicitly handles the case where an axis element is
        # negative
        for a in axis:
            reduced_shape[a] = 1
        reduced_size = int(np.prod(reduced_shape))
        # Get a 1D vector indexing over the elements in the desired result vector
        i_reduced = np.arange(reduced_size)
        # Turn into an nD array
        i_reduced = np.reshape(i_reduced, reduced_shape)
        # Broadcast this to the original shape
        i_reduced = np.broadcast_to(i_reduced, self.dependent_shape)
        # Turn this into a 1D vector
        i_reduced = np.ravel(i_reduced)
        # Get a matching index into the original array
        i_original = np.arange(self.dependent_size)
        # Now put a 1 at every [i_reduced, i_original] in a sparse matrix
        one = np.ones((i_original.size,), dtype=np.int64)
        summing_array = sparse.csc_array(
            (one, (i_reduced, i_original)),
            shape=(reduced_size, i_original.size),
        )
        data = summing_array @ self.data
        # Note that by specifying dependent_shape here, supplied
        # by the calling code, we've implicitly taken the value of
        # the keepdims argument into account.
        return SparseJacobian(
            template=self, source=data, dependent_shape=new_dependent_shape
        )

    def mean(
        self,
        new_dependent_shape: tuple,
        axis: int | tuple = None,
        dtype: DTypeLike = None,
        keepdims: bool = False,
    ) -> BaseJacobian:
        """Perform sum for the sparse Jacobians"""
        # pylint: disable=import-outside-toplevel
        from .dense_jacobians import DenseJacobian

        # Two different approaches, depending on whether axis is supplied
        if axis is None:
            # OK, here we want to sum over all the dependent elements,
            # scipy.sparse can do that.
            data = np.mean(self.data, axis=0, dtype=dtype)
            # Note that result is a dense matrix
            if keepdims:
                result_shape = (1,) * self.dependent_ndim + self.independent_shape
            else:
                result_shape = self.independent_shape
            data = np.reshape(np.asarray(data), result_shape)
            return DenseJacobian(
                template=self, source=data, dependent_shape=new_dependent_shape
            )
        # Here we want to sum over selected indices.  Recall that there is no shame in
        # thinking about things that have the same size as the dependent vector.  To
        # that end, we will formulate this as a sparse-sparse matrix multiply.  In many
        # ways, this operation is the complement of the broadcast operation, so is
        # constructed in a similar manner.
        try:
            iter(axis)
        except TypeError:
            axis = (axis,)
        # Take the orginal shape and replace the summed-over axes with one.  In the case
        # where keepdims is set, that would be independent_shape of course, but we can't
        # rely on that.
        reduced_shape = list(self.dependent_shape)
        # Note that the below code implicitly handles the case where an axis element is
        # negative
        original_size = 1
        for a in axis:
            original_size *= reduced_shape[a]
            reduced_shape[a] = 1
        reduced_size = int(np.prod(reduced_shape))
        # Get a 1D vector indexing over the elements in the
        # desired result vector
        i_reduced = np.arange(reduced_size)
        # Turn into an nD array
        i_reduced = np.reshape(i_reduced, reduced_shape)
        # Broadcast this to the original shape
        i_reduced = np.broadcast_to(i_reduced, self.dependent_shape)
        # Turn this into a 1D vector
        i_reduced = np.ravel(i_reduced)
        # Get a matching index into the original array
        i_original = np.arange(self.dependent_size)
        # Now put a 1 at every [i_reduced, i_original] in a sparse matrix
        one = np.ones((i_original.size,), dtype=np.int64)
        summing_array = sparse.csc_array(
            (one, (i_reduced, i_original)),
            shape=(reduced_size, i_original.size),
        )
        data = summing_array @ self.data / original_size
        # Note that by specifying dependent_shape here, supplied
        # by the calling code, we've implicitly taken the value of
        # the keepdims argument into account.
        return SparseJacobian(
            template=self, source=data, dependent_shape=new_dependent_shape
        )

    def cumsum(
        self,
        # pylint: disable-next=unused-argument
        new_dependent_shape: tuple,
        axis: int,
        strategy: Optional[str] = None,
        **kwargs,
    ):
        """Perform cumsum for a sparse Jacobian"""
        # pylint: disable=import-outside-toplevel
        from .dense_jacobians import DenseJacobian

        jaxis = self.get_jaxis(axis, none="flatten")
        # Cumulative sums by definitiona severly reduce sparsity. However, there may be
        # cases where we're effectively doing lots of parallel sums here, so the
        # "off-diagonal blocks" may well still be absent.  Depending on how things work
        # out, ultimately the user may still prefer to just densify, indeed that's
        # proven to be faster for non 3D retrievals.
        if strategy == "dense":
            return DenseJacobian(self).cumsum(
                axis=jaxis, new_dependent_shape=new_dependent_shape, **kwargs
            )
        # The rest don't handle fancy kwargs (as yet)
        if kwargs:
            raise NotImplementedError(
                "Sparse Jacobian cumsum does not support these arguments."
            )
        if strategy == "gather":
            # This gets us a matrix with the jaxis moved to the front, and with the
            # other dimensions densified with only the non-zero columns (optimal in
            # cases where the sparsity pattern is basically the same for all of the
            # instances along axis
            rearranged_jacobian = DenselyRearrangedSparseJacobian(self, jaxis)
            result = np.cumsum(rearranged_jacobian.matrix, axis=0)
            return rearranged_jacobian.undo(result)
        if strategy == "matrix-multiply":
            # Move the axis of interest the the front, and ravel together all the others
            rearranged_jacobian = SparselyRearrangedSparseJacobian(self, jaxis)
            # Create a lower-triangle matrix with ones and zeros and multiply by that.
            lower_triangle_array = sparse.csc_array(
                np.tri(rearranged_jacobian.matrix.shape[0])
            )
            result = lower_triangle_array @ rearranged_jacobian.matrix
            return rearranged_jacobian.undo(result)
        raise ValueError(f"Unrecognized sparse_jacobian_cumsum_strategy: {strategy}")

    def diff(
        self,
        dependent_shape,
        n=1,
        axis=-1,
        # pylint: disable-next=protected-access
        prepend=np._NoValue,
        # pylint: disable-next=protected-access
        append=np._NoValue,
    ):
        """diff method for sparse jacobian"""
        # pylint: disable=import-outside-toplevel
        from .dense_jacobians import DenseJacobian

        # For now at least I'm going to have this go to dense.
        self_dense = DenseJacobian(self)
        result_dense = self_dense.diff(dependent_shape, n, axis, prepend, append)
        return SparseJacobian(result_dense)

    def as_sparse_tensor(self):
        """Return our information as sparse tensor (tensor, not scipy)"""
        # pylint: disable=import-outside-toplevel
        import sparse as sparse_tensor

        self_coo = self.data.tocoo()
        dependent_indices = np.unravel_index(self_coo.row, self.dependent_shape)
        independent_indices = np.unravel_index(self_coo.col, self.independent_shape)
        indices = dependent_indices + independent_indices
        coords = np.stack(indices, axis=0)
        return sparse_tensor.COO(coords, self_coo.data, self.shape)

    def transpose(self, axes, result_dependent_shape):
        """Transpose a sparse jacobian"""
        coo = sparse.coo_array(self.data)
        if axes is None:
            axes = range(self.dependent_ndim)[::-1]
        row_indices = np.unravel_index(coo.row, self.dependent_shape)
        new_indices = [row_indices[i] for i in axes]
        new_row = np.ravel_multi_index(new_indices, result_dependent_shape)
        result_shape2d = (np.prod(result_dependent_shape), self.independent_size)
        result_csc = sparse.csc_array(
            (coo.data, (new_row, coo.col)), shape=result_shape2d
        )
        return SparseJacobian(
            source=result_csc,
            template=self,
            dependent_shape=result_dependent_shape,
        )

    def tensordot(self, other, axes, dependent_unit):
        """Compute self(.)other"""
        # pylint: disable=import-outside-toplevel
        import sparse as sparse_tensor

        from .dense_jacobians import DenseJacobian

        # Convert to sparse tensor form
        self_st = self.as_sparse_tensor()
        # From this point on, this is modeled after DenseJacobian.tensordot
        n_contractions = len(axes[0])
        # With this order of the tensordot, the annoying thing here is that we actually
        # want our independent dimensions (part of self) at the end, so will have to do
        # a transpose.  Let's do the tensor dot anyway.
        result_ = sparse_tensor.tensordot(self_st, other, axes)
        # Move the indepenent axes to the end.  First we want the non-contracted
        # dependent dimensions from self, these are currently at the start
        new_axis_order = list(range(self.dependent_ndim - n_contractions))
        # Then the non-contracted dimensions from other, currently at the end
        new_axis_order += list(
            range(result_.ndim - other.ndim + n_contractions, result_.ndim)
        )
        # Finally the independent dimensions, currently in the middle
        new_axis_order += list(
            range(
                self.dependent_ndim - n_contractions,
                self.dependent_ndim - n_contractions + self.independent_ndim,
            )
        )
        result = np.transpose(result_, new_axis_order)
        # Do this next bit long hand in scale independent_ndim==0
        result_dependent_shape = result_.shape[: result_.ndim - self.independent_ndim]
        if isinstance(result_, np.ndarray):
            return DenseJacobian(
                source=result,
                dependent_shape=result_dependent_shape,
                dependent_unit=dependent_unit,
                independent_shape=self.independent_shape,
                independent_unit=self.independent_unit,
            )
        elif isinstance(result, sparse_tensor.SparseArray):
            dependent_ndim = len(result_dependent_shape)
            result_csc = sparse_tensor_to_csc(result, dependent_ndim=dependent_ndim)
            return SparseJacobian(
                source=result_csc,
                dependent_shape=result_dependent_shape,
                dependent_unit=dependent_unit,
                independent_shape=self.independent_shape,
                independent_unit=self.independent_unit,
            )
        else:
            raise NotImplementedError(f"No way to handle {type(result)}")

    def rtensordot(self, other, axes, dependent_unit):
        """Compute self(.)other"""
        # pylint: disable=import-outside-toplevel
        import sparse as st

        from .dense_jacobians import DenseJacobian

        # Convert to sparse tensor form
        self_st = self.as_sparse_tensor()
        # From this point on, this is modeled after DenseJacobian.rtensordot.  This is
        # actually easier than regular tensordot because the axes end up in the right
        # order.
        data = st.tensordot(other, self_st, axes)
        # Do this next bit long hand in scale independent_ndim==0
        result_dependent_shape = data.shape[: data.ndim - self.independent_ndim]
        if isinstance(data, np.ndarray):
            return DenseJacobian(
                source=data,
                dependent_shape=result_dependent_shape,
                dependent_unit=dependent_unit,
                independent_shape=self.independent_shape,
                independent_unit=self.independent_unit,
            )
        elif isinstance(data, st.SparseArray):
            dependent_ndim = len(result_dependent_shape)
            result_csc = sparse_tensor_to_csc(data, dependent_ndim=dependent_ndim)
            return SparseJacobian(
                source=result_csc,
                dependent_shape=result_dependent_shape,
                dependent_unit=dependent_unit,
                independent_shape=self.independent_shape,
                independent_unit=self.independent_unit,
            )
        else:
            raise NotImplementedError(f"No way to handle {type(data)}")

    def extract_diagonal(self):
        """Extract the diagonal from a sparse Jacobian"""
        if self.dependent_shape != self.independent_shape:
            raise ValueError(
                "Sparse Jacobian is not square "
                + f"{self.dependent_shape};{self.independent_shape}"
            )
        result = np.reshape(self.data.diagonal(), self.dependent_shape)
        return apply_units(result, self.dependent_unit / self.independent_unit)

    def toarray(self):
        """Return data from self as a non-dense tensor or sparse array"""
        if self.independent_ndim != 1 or self.dependent_ndim != 1:
            return apply_units(
                self.as_sparse_tensor(), self.dependent_unit / self.independent_unit
            )
        else:
            return apply_units(self.data, self.dependent_unit / self.independent_unit)

    def todensearray(self):
        """Return data from self as dense array"""
        return apply_units(
            np.reshape(self.data.toarray(), self.shape),
            self.dependent_unit / self.independent_unit,
        )

    def to2ddensearray(self):
        """Return data from self as dense array"""
        return apply_units(
            self.data.toarray(), self.dependent_unit / self.independent_unit
        )

    def to2darray(self):
        """Return data from self as sparse array"""
        return apply_units(self.data, self.dependent_unit / self.independent_unit)

    # pylint: disable=redefined-outer-name
    def nan_to_num(self, copy=True, nan=0.0, posinf=None, neginf=None):
        """Implements numpy.nan_to_num"""
        if copy:
            data = self.data.copy()
        else:
            data = self.data
        data.data = np.nan_to_num(
            data.data, copy=False, nan=nan, posinf=posinf, neginf=neginf
        )
        return type(self)(source=data, template=self)

    def scalar_multiply(self, scale):
        """Multiply Jacobian by a scalar"""
        magnitude, units = get_magnitude_and_unit(scale)
        return SparseJacobian(
            template=self,
            source=self.data * magnitude,
            dependent_unit=self.dependent_unit * units,
        )

    def _join(self, other, location, axis, result_dependent_shape):
        """Insert/append sparse Jacobians"""
        # For now at least, we demand a scalar location here.
        scalar_only = "insert/append only supported for scalar indices, for now"
        try:
            if location.size != 1 or len(location) != 1:
                raise NotImplementedError(scalar_only)
        except (AttributeError, TypeError):
            pass
        nd = self.dependent_ndim
        self_dependent_shape = self.dependent_shape
        if len(other.dependent_shape) != nd:
            other_dependent_shape = list(self_dependent_shape)
            other_dependent_shape[axis] = 1
            if not shapes_broadcastable(other.dependent_shape, other_dependent_shape):
                raise ValueError(
                    "Shapes not broadcastable: "
                    + f"{other.dependent_shape} and {other_dependent_shape}"
                )
        else:
            other_dependent_shape = other.dependent_shape
        n_self = self_dependent_shape[axis]
        n_other = other_dependent_shape[axis]
        # Convert both to coo-based sparse matrices
        self_coo = sparse.coo_array(self.data)
        other_coo = sparse.coo_array(other.data)
        # Unravel the row indices
        self_row_indices = np.unravel_index(self_coo.row, shape=self_dependent_shape)
        other_row_indices = np.unravel_index(other_coo.row, shape=other_dependent_shape)
        # Now create the result indices, join the two sparse arrays, the relevant axis
        # needs renumbering, others can remain the same.
        result_row_indices = list()
        for i, sri, ori in zip(range(nd), self_row_indices, other_row_indices):
            if i == axis:
                self_index_map = np.arange(n_self)
                self_index_map[location:] += n_other
                other_index_map = np.arange(n_other) + location
                result_row_indices.append(
                    np.concatenate((self_index_map[sri], other_index_map[ori]))
                )
            else:
                result_row_indices.append(np.concatenate((sri, ori)))
        # Re-ravel the row indcies
        result_row = np.ravel_multi_index(result_row_indices, result_dependent_shape)
        # The column indices are a straight forward concatenation
        result_col = np.concatenate((self_coo.col, other_coo.col))
        # Merge the two data arrays
        result_coo_data = np.concatenate((self_coo.data, other_coo.data))
        # Create the result
        result_dependent_size = np.prod(result_dependent_shape)
        result_j_shape2d = (result_dependent_size, self.independent_size)
        result_csc = sparse.csc_array(
            (result_coo_data, (result_row, result_col)), result_j_shape2d
        )
        return SparseJacobian(
            result_csc, template=self, dependent_shape=result_dependent_shape
        )

    def linear_interpolator(
        self,
        x_in,
        axis=-1,
        extrapolate=None,  # pylint: disable=unused-argument
    ):
        """Return a linear interpolator for a given Jacobian axis"""
        return SparseJacobianLinearInterpolator(self, x_in, axis)

    def spline_interpolator(
        self,
        x_in,
        axis=-1,
        bc_type="not_a_knot",
        extrapolate=None,
    ):
        """Return a spline interpolator for a given Jacobian axis"""
        return SparseJacobianSplineInterpolator(
            self,
            x_in=x_in,
            axis=axis,
            bc_type=bc_type,
            extrapolate=extrapolate,
        )


class SparseJacobianLinearInterpolator(object):
    """Interpolates a SparseJacobian along one dependent axis"""

    def __init__(
        self,
        jacobian,
        x_in,
        axis=-1,
        extrapolate=False,
    ):
        """Setup an interpolator for a given DenseJacobian"""
        self.jacobian = jacobian
        self.jaxis = jacobian.get_jaxis(axis, none="first")
        self.x_in = x_in
        self.extrapolate = extrapolate
        # Transpose the jacobian to put the interpolating axis in front
        self.rearranged_jacobian = SparselyRearrangedSparseJacobian(
            jacobian,
            promoted_axis=self.jaxis,
        )

    def __call__(self, x_out):
        """Interpolate a SparseJacobian to a new value along an axis"""
        i_lower, i_upper, w_lower, w_upper = linear_interpolation_indices_and_weights(
            self.x_in,
            x_out,
            extrapolate=self.extrapolate,
        )
        # Convert indices and weights to a sparse matrix
        row = np.concatenate([np.arange(x_out.size)] * 2)
        col = np.concatenate([i_lower, i_upper])
        weight = np.concatenate([w_lower, w_upper])
        weight_array = sparse.csc_array(
            (weight, (row, col)), shape=[x_out.size, self.x_in.size]
        )
        result = weight_array @ self.rearranged_jacobian.matrix
        result = self.rearranged_jacobian.undo(result)
        return result


class SparseJacobianSplineInterpolator(object):
    """Interpolates a SparseJacobian along one dependent axis"""

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
        # Transpose the jacobian to put the interpolating axis in front, and then ravel
        # other axes together into a dense matrix.  This assumes that for each "row" the
        # non-zero elements occupy similar columns.
        self.rearranged_jacobian = DenselyRearrangedSparseJacobian(
            jacobian,
            promoted_axis=self.jaxis,
        )
        self.interpolator = interpolate.CubicSpline(
            x_in,
            self.rearranged_jacobian.matrix,
            axis=0,
            bc_type=bc_type,
            extrapolate=extrapolate,
        )

    def __call__(self, x_out):
        """Interpolate a SparseJacobian to a new value along an axis"""
        # Call the interpolator to get an interpolated version of the rearranged dense
        # matrix.
        intermediate = self.interpolator(x_out)
        # Undo the rearrangement and return the result
        return self.rearranged_jacobian.undo(intermediate)


class Sparse2DExtractor(Base2DExtractor):
    """A class to directly get a subset from the Jacobian"""

    def __init__(self, jacobian: SparseJacobian):
        super().__init__(jacobian)
        # Get the Jacobian as a sparse tensor for later efficient use
        self.jacobian_as_sparse_tensor = self.jacobian.as_sparse_tensor()

    def __getitem__(self, key):
        result_shape = self.preprocess_key(key)
        return sparse.csc_array(
            self.jacobian_as_sparse_tensor[key].reshape(result_shape).tocsc()
        )
