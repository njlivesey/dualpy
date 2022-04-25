"""Class for sparse jacobians"""
import scipy.sparse as sparse
import numpy as np
import copy

from .jacobian_helpers import _array_to_sparse_diagonal, _shapes_broadcastable
from .base_jacobian import BaseJacobian
from mls_scf_tools.util import linear_interpolation_indices_and_weights

__all__ = ["SparseJacobian"]


def _rearrange_2d(matrix, original_shapes, axes=None, promote=None, demote=None):
    """Rearrange a 2D sparse matrix representing a tensor into a different axis order

    Parameters
    ----------

    matrix: (sparse) matrix-like
        A 2D sparse matrix (typically csc format) that is really storing a sparse tensor
        with the axes raveled into to sets.

    original_shape: list of two lists of int
        A list of two lists giving the shapes that ravel to make the rows and columns of
        matrix.

    axes: list of two lists of int
        Which axes of the original should constitute the raveling of the result

    promote: A single axis to move to the front

    demote: A single axis to move to the back

    """
    # Work out what we've been asked to do
    n0_in = len(original_shapes[0])
    n1_in = len(original_shapes[1])
    n_in = n0_in + n1_in
    original_shape = original_shapes[0] + original_shapes[1]
    if promote is not None or demote is not None:
        if axes is not None:
            raise ValueError("Cannot set both axes and promote/demote")
        if promote is not None and demote is not None:
            raise ValueError("Cannot set both promote and demote")
        remainder = list(range(len(original_shape)))
        if promote is not None:
            remainder.pop(promote)
            axes = [[promote], remainder]
        if demote is not None:
            remainder.pop(demote)
            axes = [remainder, [demote]]
    n0_out, n1_out = len(axes[0]), len(axes[1])
    n_out = n0_out + n1_out
    if n_out != n_in:
        raise ValueError("Mismatched dimensionality")
    all_axes = axes[0] + axes[1]
    rearranged_shape = [original_shape[i] for i in all_axes]
    rearranged_shape_2d = [
        np.prod(rearranged_shape[:n0_out]),
        np.prod(rearranged_shape[n0_out:]),
    ]
    # Convert the matrix to coo form
    matrix_coo = matrix.tocoo()
    # Now get the raveled indices for all the elements
    i_raveled = np.ravel_multi_index((matrix_coo.row, matrix_coo.col), matrix_coo.shape)
    # Now get these all as unravelled indices across all the axes
    i_all_original = np.unravel_index(i_raveled, original_shape)
    # Now rearrange this list of indices according to the prescription supplied.
    i_all_rearranged = [i_all_original[i] for i in all_axes]
    # Now ravel the new row/column indices
    i0_new = np.ravel_multi_index(i_all_rearranged[:n0_out], rearranged_shape[:n0_out])
    i1_new = np.ravel_multi_index(i_all_rearranged[n0_out:], rearranged_shape[n0_out:])
    # Make the new matrix - this is what does the actual rearrangement
    result = sparse.csc_matrix(
        (matrix_coo.data, (i0_new, i1_new)), shape=rearranged_shape_2d
    )
    undo_order = [None] * n_in
    for i, a in enumerate(all_axes):
        undo_order[a] = i
    undo = [undo_order[:n0_in], undo_order[n0_in:]]
    return (
        result,
        [rearranged_shape[0:n0_out], rearranged_shape[n0_out:]],
        undo,
    )


class SparseJacobian(BaseJacobian):
    """A dljacobian that's stored as sparse 2D array under the hood"""

    def __init__(self, data=None, template=None, **kwargs):
        """Create a new sparse jacobian"""
        from .dense_jacobians import DenseJacobian
        from .diagonal_jacobians import DiagonalJacobian

        # This kind of Jacobian can only be initialized using another
        # sparse jacobian or by a diagonal one.
        if isinstance(data, BaseJacobian):
            if template is None:
                template = data
            else:
                raise ValueError(
                    "Cannot supply template with jacobian data simultaneously"
                )
        super().__init__(template=template, **kwargs)
        if isinstance(data, SparseJacobian):
            data2d_ = data.data2d
        elif isinstance(data, DiagonalJacobian):
            data2d_ = _array_to_sparse_diagonal(data.data.ravel())
        elif isinstance(data, DenseJacobian):
            data2d_ = sparse.csc_matrix(data.data2d)
        elif type(data) is sparse.csc_matrix:
            data2d_ = data
        elif data is None:
            data2d_ = sparse.csc_matrix(self.shape2d)
        else:
            raise TypeError("Values supplied to SparseJacobian are not suitable")
        if data2d_.shape != self.shape2d:
            raise ValueError("Attempt to create jacobian_sparse with wrong-sized input")
        else:
            self.data2d = data2d_

    def __str__(self):
        """Provide a string summary of a sparse Jacobian"""
        percent = (
            100.0 * self.data2d.nnz / (self.dependent_size * self.independent_size)
        )
        suffix = (
            f"\ndata2d is {self.data2d.shape}"
            + f" with {self.data2d.nnz} numbers stored ({percent:.2g}%)"
        )
        return super().__str__() + suffix

    def _check(self, name):
        """Integrity checks for sparse Jacobians"""
        self._check_jacobian_fundamentals(name)
        correct_2dshape = (self.dependent_size, self.independent_size)
        assert self.data2d.shape == correct_2dshape, (
            f"array2d shape mismatch for {name}, "
            f"{self.data2d.shape} != {correct_2dshape}"
        )

    def _getjitem(self, new_shape, key):
        """A getitem type method for sparse Jacobians"""
        key = self._preprocess_getsetitem_key(key)
        # Now we're going to collapse all the dependent key into a 1D index array.
        # There are probably more efficient ways in which to handle this, particular
        # for the cases where key is filled entirely with slices. However, the thing
        # to remember is that there is no shame in handling things that are the
        # length of the dependent vector (or independent one come to that), just
        # avoid things that are the cartesian product of them.  Accordingly this
        # straightforward approach is probably good enough.
        i_full = np.reshape(np.arange(self.dependent_size), self.dependent_shape)
        i_subset = i_full[key].ravel()
        dependent_slice = (i_subset,)
        # Note, I originally tried to sidestep the null-operation for the above, but it
        # didn't work when adding np.newaxes.  Simpler just to do it all the time.
        # Append a "get the lot" slice for the independent variable dimension
        jkey = dependent_slice + (slice(None),)
        result_ = self.data2d.__getitem__(jkey)
        return SparseJacobian(data=result_, template=self, dependent_shape=new_shape)

    def _setjitem(self, key, value):
        """A setitem type method for dense Jacobians"""
        if not isinstance(value, SparseJacobian):
            raise NotImplementedError(
                "Not implemented SparseJacobian._setjitem for {type(value)}"
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
        i_dependent_subset = i_dependent_full[key].ravel()
        # Now, the dependent index is the "rows" for the sparse Jacobian.  Transform
        # both self and value into lil sparse matrices.
        self_lil = self.data2d.tolil()
        value_lil = value.data2d.tolil()
        # Replace all affected rows in self with the rows in value
        for i, (r, d) in enumerate(zip(value_lil.rows, value_lil.data)):
            ii = i_dependent_subset[i]
            self_lil.rows[ii] = r
            self_lil.data[ii] = d
        self.data2d = self_lil.tocsc()

    def broadcast_to(self, shape):
        """Broadcast the dependent vector part of a sparse Jacobian to another shape"""
        # Don't bother doing anything if the shape is already good
        if shape == self.dependent_shape:
            return self
        if not _shapes_broadcastable(self.dependent_shape, shape):
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
        i_old = i_old.ravel()
        # Get a matching i_new array
        i_new = np.arange(i_old.size)
        # Now put a 1 at every [i_new,i_old] point in a sparse matrix
        one = np.ones((i_old.size,), dtype=np.int64)
        M = sparse.csc_matrix(
            (one, (i_new, i_old)), shape=(i_new.size, self.dependent_size)
        )
        # Now do a matrix multiply to accomplish what broadcast tries
        # to do.
        result_ = M @ self.data2d
        return SparseJacobian(data=result_, template=self, dependent_shape=shape)

    def reshape(self, shape, order, parent_flags):
        """Reshape a sparse Jacobian to a new dependent vector"""
        # Don't bother doing anything if the shape is already good
        if shape == self.dependent_shape:
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
        if int(np.prod(shape)) != input_jacobian.dependent_size:
            raise ValueError("Unable to reshape SparseJacobian to new shape")
        return SparseJacobian(
            data=input_jacobian.data2d, template=input_jacobian, dependent_shape=shape
        )

    def premul_diag(self, diag):
        """Dependent-Element by element multiply of an other quantity"""
        # This is a hand-woven diagnonal/sparse-csc multiply
        # equivalent to
        #     out = (diagonal matrix) @ self
        # However, when expressed that way it's less efficient then below
        diag_, dependent_unit, dependent_shape = self._prepare_premul_diag(diag)
        # Do a manual broadcast if needed (which, in the case of the
        # Jacobian is actually not a real broadcast, but unfortunately
        # a somewhat wasteful replicate opration.  But since the
        # result will be the size of the broadcasted shape in any
        # case, there is no real loss.
        if dependent_shape != self.dependent_shape:
            self = self.broadcast_to(dependent_shape)
            diag_ = np.broadcast_to(diag_, dependent_shape)
        diag_ = diag_.ravel()

        # out_ = self.data2d.copy()
        # if np.iscomplexobj(diag_) and not np.iscomplexobj(out_):
        #     out_=out_ * complex(1, 0)
        # if len(diag_) == 1:
        #     out_.data *= diag_[0]
        # else:
        #     # out_.data *= diag_[out_.indices]
        #     out_.data *= np.take(diag_, out_.indices)

        if len(diag_) == 1:
            out_data = self.data2d.data * diag_
        else:
            out_data = self.data2d.data * np.take(diag_, self.data2d.indices)
        out_ = sparse.csc_matrix(
            (out_data, self.data2d.indices, self.data2d.indptr), shape=self.data2d.shape
        )
        return SparseJacobian(
            out_,
            template=self,
            dependent_unit=dependent_unit,
            dependent_shape=dependent_shape,
        )

    def __neg__(self):
        return type(self)(-self.data2d, template=self)

    def real(self):
        return type(self)(np.real(self.data2d), template=self)

    def insert(self, obj, axis, dependent_shape):
        """insert method for sparse Jacobian"""
        # We do this simply by moving the indices around
        # First convert to a coo-based sparse array
        self_coo = self.data2d.tocoo()
        # OK, unravel the multi index that makes up the rows
        all_indices = list(np.unravel_index(self_coo.row, self.dependent_shape))
        axis = self._get_jaxis(axis, none="zero")
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
        result_csc = sparse.csc_matrix(
            (self_coo.data, (row, self_coo.col)),
            shape=(dependent_size, self.independent_size),
        )
        return SparseJacobian(
            template=self, data=result_csc, dependent_shape=dependent_shape
        )

    def sum(self, dependent_shape, axis=None, dtype=None, keepdims=False):
        """Perform sum for the sparse Jacobians"""
        from .dense_jacobians import DenseJacobian

        # Two different approaches, depending on whether axis is supplied
        if axis is None:
            # OK, here we want to sum over all the dependent elements,
            # scipy.sparse can do that.
            result_ = np.sum(self.data2d, axis=0, dtype=dtype)
            # Note that result is a dense matrix
            if keepdims:
                result_shape = (1,) * self.dependent_ndim + self.independent_shape
            else:
                result_shape = self.independent_shape
            result_ = np.reshape(np.asarray(result_), result_shape)
            result = DenseJacobian(
                template=self, data=result_, dependent_shape=dependent_shape
            )
            pass
        else:
            # Here we want to sum over selected indices.  Recall that
            # there is no shame in thinking about things that have the
            # same size as the dependent vector.  To that end, we will
            # formulate this as a sparse-sparse matrix multiply.  In
            # many ways, this operation is the complement of the
            # broadcast operation, so is constructed in a similar
            # manner.
            try:
                iter(axis)
            except TypeError:
                axis = (axis,)
            # Take the orginal shape and replace the summed-over axes
            # with one.  In the case where keepdims is set, that would
            # be independent_shape of course, but we can't rely on
            # that.
            reduced_shape = list(self.dependent_shape)
            # Note that the below code implicitly handles the case
            # where an axis element is negative
            for a in axis:
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
            i_reduced = i_reduced.ravel()
            # Get a matching index into the original array
            i_original = np.arange(self.dependent_size)
            # Now put a 1 at every [i_reduced, i_original] in a sparse matrix
            one = np.ones((i_original.size,), dtype=np.int64)
            M = sparse.csc_matrix(
                (one, (i_reduced, i_original)),
                shape=(reduced_size, i_original.size),
            )
            result_ = M @ self.data2d
            # Note that by specifying dependent_shape here, supplied
            # by the calling code, we've implicitly taken the value of
            # the keepdims argument into account.
            result = SparseJacobian(
                template=self, data=result_, dependent_shape=dependent_shape
            )
        return result

    def cumsum(self, axis, heroic=True):
        """Perform cumsum for a sparse Jacobian"""
        from .dense_jacobians import DenseJacobian
        jaxis = self._get_jaxis(axis, none="flatten")
        # Cumulative sums by definitiona severly reduce sparsity.
        # However, there may be cases where we're effectively doing
        # lots of parallel sums here, so the "off-diagonal blocks" may
        # well still be absent.  Depending on how things work out,
        # ultimately the user may still prefer to just densify, indeed
        # that's proven to be faster for non 3D retrievals.
        if not heroic:
            self_dense = DenseJacobian(self)
            result = self_dense.cumsum(jaxis)
        else:
            # This is much simpler if jaxis is None or (thus and)
            # dependent_ndim==1
            easy = jaxis is None or self.dependent_ndim == 1
            if easy:
                new_csc = self.data2d
                nrows = self.shape[0]
            else:
                # OK, we want to take the current 2D matrix and pull it
                # about so that the summed-over jaxis is now the rows, and
                # the columns are all the remaining dependent axes plus
                # the independent ones.
                new_csc, rearranged_shapes, undo_rearrange = _rearrange_2d(
                    self.data2d,
                    [self.dependent_shape, self.independent_shape],
                    promote=jaxis,
                )
                nrows = self.shape[jaxis]
            # We have two choices here, we could do a python loop to build up and store
            # cumulative sums, but I suspect that, while on paper more efficient
            # (reducing the number of additions, it would be slow in reality. Instead,
            # I'll create lower-triangle matrix with ones and zeros and multiply by
            # that.
            lt = sparse.csc_matrix(np.tri(nrows))
            intermediate = lt @ new_csc
            if easy:
                result = SparseJacobian(template=self, data=intermediate)
            else:
                restored_order = list(range(1, self.ndim))
                restored_order.insert(jaxis, 0)
                result_csc, dummy_result_shapes, dummy_undo = _rearrange_2d(
                    intermediate,
                    rearranged_shapes,
                    undo_rearrange,
                )
                result = SparseJacobian(template=self, data=result_csc)
        return result

    def diff(
        self,
        dependent_shape,
        n=1,
        axis=-1,
        prepend=np._NoValue,
        append=np._NoValue,
    ):
        """diff method for sparse jacobian"""
        from .dense_jacobians import DenseJacobian

        # For now at least I'm going to have this go to dense.
        self_dense = DenseJacobian(self)
        result_dense = self_dense.diff(dependent_shape, n, axis, prepend, append)
        return SparseJacobian(result_dense)

    def as_sparse_tensor(self):
        """Return our information as sparse tensor (tensor, not scipy)"""
        import sparse as st

        self_coo = self.data2d.tocoo()
        dependent_indices = np.unravel_index(self_coo.row, self.dependent_shape)
        independent_indices = np.unravel_index(self_coo.col, self.independent_shape)
        indices = dependent_indices + independent_indices
        coords = np.stack(indices, axis=0)
        return st.COO(coords, self_coo.data, self.shape)

    def transpose(self, axes, result_dependent_shape):
        """Transpose a sparse jacobian"""
        coo = sparse.coo_matrix(self.data2d)
        if axes is None:
            axes = range(self.dependent_ndim)[::-1]
        row_indices = np.unravel_index(coo.row, self.dependent_shape)
        new_indices = [row_indices[i] for i in axes]
        new_row = np.ravel_multi_index(new_indices, result_dependent_shape)
        result_shape2d = (np.prod(result_dependent_shape), self.independent_size)
        new_csc = sparse.csc_matrix(
            (coo.data, (new_row, coo.col)), shape=result_shape2d
        )
        return SparseJacobian(
            new_csc,
            template=self,
            dependent_shape=result_dependent_shape,
        )

    def tensordot(self, other, axes, dependent_unit):
        """Compute self(.)other"""
        import sparse as st
        from .dense_jacobians import DenseJacobian

        # Convert to sparse tensor form
        self_st = self.as_sparse_tensor()
        # From this point on, this is modeled after DenseJacobian.tensordot
        n_contractions = len(axes[0])
        # With this order of the tensordot, the annoying thing here is that we actually
        # want our independent dimensions (part of self) at the end, so will have to do
        # a transpose.  Let's do the tensor dot anyway.
        result_ = st.tensordot(self_st, other, axes)
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
        result_ = np.transpose(result_, new_axis_order)
        result_dependent_shape = result_.shape[: -self.independent_ndim]
        if isinstance(result_, np.ndarray):
            return DenseJacobian(
                data=result_,
                dependent_shape=result_dependent_shape,
                dependent_unit=dependent_unit,
                independent_shape=self.independent_shape,
                independent_unit=self.independent_unit,
            )
        elif isinstance(result_, st.COO) or isinstance(result_, st.GCXS):
            if isinstance(result_, st.GCXS):
                result_ = result_.tocoo()
            dependent_ndim = len(result_dependent_shape)
            coords = np.split(result_.coords, result_.ndim)
            dependent_coords = [c[0] for c in coords[:dependent_ndim]]
            independent_coords = [c[0] for c in coords[dependent_ndim:]]
            dependent_indices = np.ravel_multi_index(
                dependent_coords, result_dependent_shape
            )
            independent_indices = np.ravel_multi_index(
                independent_coords, self.independent_shape
            )
            dependent_size = np.prod(result_dependent_shape)
            result_csc = sparse.csc_matrix(
                (result_.data, (dependent_indices, independent_indices)),
                shape=[dependent_size, self.independent_size],
            )
            return SparseJacobian(
                data=result_csc,
                dependent_shape=result_dependent_shape,
                dependent_unit=dependent_unit,
                independent_shape=self.independent_shape,
                independent_unit=self.independent_unit,
            )
        else:
            raise NotImplementedError(f"No way to handle {type(result_)}")

    def rtensordot(self, other, axes, dependent_unit):
        """Compute self(.)other"""
        import sparse as st
        from .dense_jacobians import DenseJacobian

        # Convert to sparse tensor form
        self_st = self.as_sparse_tensor()
        # From this point on, this is modeled after DenseJacobian.rtensordot.  This is
        # actually easier than regular tensordot because the axes end up in the right
        # order.
        result_ = st.tensordot(other, self_st, axes)
        result_dependent_shape = result_.shape[: -self.independent_ndim]
        if isinstance(result_, np.ndarray):
            return DenseJacobian(
                data=result_,
                dependent_shape=result_dependent_shape,
                dependent_unit=dependent_unit,
                independent_shape=self.independent_shape,
                independent_unit=self.independent_unit,
            )
        elif isinstance(result_, st.COO) or isinstance(result_, st.GCXS):
            if isinstance(result_, st.GCXS):
                result_ = result_.tocoo()
            dependent_ndim = len(result_dependent_shape)
            coords = np.split(result_.coords, result_.ndim)
            dependent_coords = [c[0] for c in coords[:dependent_ndim]]
            independent_coords = [c[0] for c in coords[dependent_ndim:]]
            dependent_indices = np.ravel_multi_index(
                dependent_coords, result_dependent_shape
            )
            independent_indices = np.ravel_multi_index(
                independent_coords, self.independent_shape
            )
            dependent_size = np.prod(result_dependent_shape)
            result_csc = sparse.csc_matrix(
                (result_.data, (dependent_indices, independent_indices)),
                shape=[dependent_size, self.independent_size],
            )
            return SparseJacobian(
                data=result_csc,
                dependent_shape=result_dependent_shape,
                dependent_unit=dependent_unit,
                independent_shape=self.independent_shape,
                independent_unit=self.independent_unit,
            )
        else:
            raise NotImplementedError(f"No way to handle {type(result_)}")
        
    def extract_diagonal(self):
        """Extract the diagonal from a sparse Jacobian"""
        if self.dependent_shape != self.independent_shape:
            raise ValueError(
                "Sparse Jacobian is not square "
                + f"{self.dependent_shape};{self.independent_shape}"
            )
        result_ = np.reshape(self.data2d.diagonal(), self.dependent_shape)
        return result_ << (self.dependent_unit / self.independent_unit)

    def todensearray(self):
        from .dense_jacobians import DenseJacobian

        self_dense = DenseJacobian(self)
        return self_dense.todensearray()

    def to2ddensearray(self):
        return self.data2d.toarray() << (self.dependent_unit / self.independent_unit)

    def to2darray(self):
        return self.data2d << (self.dependent_unit / self.independent_unit)

    def nan_to_num(self, copy=True, nan=0.0, posinf=None, neginf=None):
        if copy:
            data2d = self.data2d.copy()
        else:
            data2d = self.data2d
        data2d.data = np.nan_to_num(
            data2d.data, copy=False, nan=nan, posinf=posinf, neginf=neginf
        )
        return self.__class__(template=self, data=data2d)

    def scalar_multiply(self, scale):
        """Multiply Jacobian by a scalar"""
        self.dependent_unit *= scale.unit
        self.data2d *= scale.value
        return self

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
            if not _shapes_broadcastable(other.dependent_shape, other_dependent_shape):
                raise ValueError(
                    "Shapes not broadcastable: "
                    + f"{other.dependent_shape} and {other_dependent_shape}"
                )
        else:
            other_dependent_shape = other.dependent_shape
        n_self = self_dependent_shape[axis]
        n_other = other_dependent_shape[axis]
        # Convert both to coo-based sparse matrices
        self_coo = sparse.coo_matrix(self.data2d)
        other_coo = sparse.coo_matrix(other.data2d)
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
        result_csc = sparse.csc_matrix(
            (result_coo_data, (result_row, result_col)), result_j_shape2d
        )
        return SparseJacobian(
            result_csc, template=self, dependent_shape=result_dependent_shape
        )

    def linear_interpolator(self, x_in, axis=-1):
        """Return an interpolator for a given Jacobian axis"""
        return SparseJacobianLinearInterpolator(self, x_in, axis)


class SparseJacobianLinearInterpolator(object):
    """Interpolates a SparseJacobian along one dependent axis"""

    def __init__(self, jacobian, x_in, axis=-1):
        """Setup an interpolator for a given DenseJacobian"""
        self.jacobian = jacobian
        self.jaxis = jacobian._get_jaxis(axis, none="first")
        self.x_in = x_in
        # Transpose the jacobian to put the interpolating axis in front
        self.rearranged, self.rearranged_shapes, self.undo_reordering = _rearrange_2d(
            jacobian.data2d,
            [jacobian.dependent_shape, jacobian.independent_shape],
            promote=self.jaxis,
        )

    def __call__(self, x_out):
        """Interpolate a SparseJacobian to a new value along an axis"""
        i_lower, i_upper, w_lower, w_upper = linear_interpolation_indices_and_weights(
            self.x_in, x_out
        )
        # Convert indices and weights to a sparse matrix
        row = np.concatenate([np.arange(x_out.size)] * 2)
        col = np.concatenate([i_lower, i_upper])
        weight = np.concatenate([w_lower, w_upper])
        W = sparse.csc_matrix((weight, (row, col)), shape=[x_out.size, self.x_in.size])
        result_rearranged = W @ self.rearranged
        rearranged_shapes = copy.deepcopy(self.rearranged_shapes)
        rearranged_shapes[0][0] = x_out.size
        result, result_shape, dummy_undo = _rearrange_2d(
            result_rearranged,
            rearranged_shapes,
            axes=self.undo_reordering,
        )
        new_dependent_shape = result_shape[0]
        return SparseJacobian(
            template=self.jacobian, data=result, dependent_shape=new_dependent_shape
        )
