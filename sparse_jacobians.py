"""Class for sparse jacobians"""
import scipy.sparse as sparse
import numpy as np

from .jacobian_helpers import _array_to_sparse_diagonal, _shapes_broadcastable
from .base_jacobian import BaseJacobian

__all__ = ["SparseJacobian"]


class SparseJacobian(BaseJacobian):
    """A dljacobian that's stored as sparse 2D array under the hood"""

    def __init__(self, data, template=None, **kwargs):
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
        else:
            raise TypeError("Values supplied to SparseJacobian are not suitable")
        if data2d_.shape != self.shape2d:
            # print (data2d_.shape, self.shape2d)
            raise ValueError("Attempt to create jacobian_sparse with wrong-sized input")
        else:
            self.data2d = data2d_

    def __str__(self):
        """Provide a string summary of a sparse Jacobian"""
        suffix = (
            f"\ndata2d is {self.data2d.shape}"
            + f" with {self.data2d.nnz} numbers stored"
        )
        return super().__str__() + suffix

    def _getjitem(self, new_shape, key):
        """A getitem type method for sparse Jacobians"""
        from .dense_jacobians import DenseJacobian

        # OK, getjitem for sparse is a bit more complicated than for
        # the others.  This is mainly because matplotlib seems to put
        # in some funny getitem requests to its arguments.  If that
        # happens we'll fall back to dense and provide that as the
        # result.  scipy.sparse is fussier about indexing than regular
        # arrays too, so there's some choreography associated with
        # that too.
        if self.dependent_ndim > 1:
            # This thing to remember is that there is no shame in
            # handling things that are the length of the dependent
            # vector (or independent one come to that), just avoid
            # things that are the cartesian product of them.
            iold = np.reshape(np.arange(self.dependent_size), self.dependent_shape)
            inew = iold.__getitem__(key).ravel()
            dependent_slice = (inew,)
        else:
            dependent_slice = key
        # OK, so now we have the option to fall back to a dense case
        # because matplotlib makes some strange __getitem__ requests
        # that give odd errors down the road.
        # Append a "get the lot" slice for the independent variable dimension
        try:
            jSlice = dependent_slice + (slice(None),)
        except TypeError:
            jSlice = (dependent_slice, slice(None))
        try:
            if len(jSlice) > 2:
                raise TypeError("Dummy raise to fall back to dense")
            result_ = self.data2d.__getitem__(jSlice)
            return SparseJacobian(
                data=result_, template=self, dependent_shape=new_shape
            )
        except TypeError:
            # warnings.warn("SparseJacobian._getjitem had to fall back to dense")
            self_dense = DenseJacobian(self)
            return self_dense._getjitem(new_shape, key)

    def _setjitem(self, key, value):
        """A setitem type method for dense Jacobians"""
        raise NotImplementedError(
            "Not (yet) written the setitem capability for sparse Jacobians"
        )

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
        iold = np.arange(self.dependent_size)
        # Turn into an nD array
        iold = np.reshape(iold, self.dependent_shape)
        # Broadcast this to the new shape
        iold = np.broadcast_to(iold, shape)
        # Now convert to a 1D array
        iold = iold.ravel()
        # Get a matching inew array
        inew = np.arange(iold.size)
        # Now put a 1 at every [inew,iold] point in a sparse matrix
        one = np.ones((iold.size,), dtype=np.int64)
        M = sparse.csc_matrix(
            sparse.coo_matrix(
                (one, (inew, iold)), shape=(inew.size, self.dependent_size)
            )
        )
        # Now do a matrix multiply to accomplish what broadcast tries
        # to do.
        result_ = M @ self.data2d
        return SparseJacobian(data=result_, template=self, dependent_shape=shape)

    def reshape(self, shape, order="C"):
        """Reshape a sparse Jacobian to a new dependent vector"""
        # Don't bother doing anything if the shape is already good
        if shape == self.dependent_shape:
            return self
        # For the sparse jacobians, which never really have a shape
        # anyway, this simply involves updating the shapes of record
        # (assuming it's a valid one).
        if int(np.prod(shape)) != self.dependent_size:
            raise ValueError("Unable to reshape SparseJacobian to new shape")
        return SparseJacobian(data=self.data2d, template=self, dependent_shape=shape)

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
        if axis is None:
            axis_ = 0
        else:
            axis_ = axis
        i = all_indices[axis_]
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
        all_indices[axis_] = i
        row = np.ravel_multi_index(all_indices, dependent_shape)
        dependent_size = int(np.prod(dependent_shape))
        self_coo = sparse.coo_matrix(
            (self_coo.data, (row, self_coo.col)),
            shape=(dependent_size, self.independent_size),
        )
        return SparseJacobian(
            template=self, data=self_coo.tocsc(), dependent_shape=dependent_shape
        )

    def sum(self, dependent_shape, axis=None, dtype=None, keepdims=False):
        """Perform sum for the sparse Jacobians"""
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
            # be in dependent_shape of course, but we can't rely on
            # that.
            reduced_shape = list(self.dependent_shape)
            # print(f"Start with {self}")
            # print(f"Summing over {axis}")
            # Note that the below code implicitly handles the case
            # where an axis element is negative
            for a in axis:
                reduced_shape[a] = 1
            reduced_size = int(np.prod(reduced_shape))
            # print(f"End with {reduced_shape}, {type(reduced_shape)}")
            # Get a 1D vector indexing over the elements in the
            # desired result vector
            ireduced = np.arange(reduced_size)
            # Turn into an nD array
            ireduced = np.reshape(ireduced, reduced_shape)
            # Broadcast this to the original shape
            ireduced = np.broadcast_to(ireduced, self.dependent_shape)
            # Turn this into a 1D vector
            ireduced = ireduced.ravel()
            # Get a matching index into the original array
            ioriginal = np.arange(self.dependent_size)
            # Now put a 1 at every [ireduced, ioriginal] in a sparse matrix
            one = np.ones((ioriginal.size,), dtype=np.int64)
            M = sparse.csc_matrix(
                sparse.coo_matrix(
                    (one, (ireduced, ioriginal)), shape=(reduced_size, ioriginal.size)
                )
            )
            result_ = M @ self.data2d
            # Note that by specifying dependent_shape here, supplied
            # by the calling code, we've implicitly taken the value of
            # the keepdims argument into account.
            result = SparseJacobian(
                template=self, data=result_, dependent_shape=dependent_shape
            )
            pass
        return result

    def cumsum(self, axis, heroic=False):
        """Perform cumsum for a sparse Jacobian"""
        # Cumulative sums by definitiona severly reduce sparsity.
        # However, there may be cases where we're effectively doing
        # lots of parallel sums here, so the "off-diagonal blocks" may
        # well still be absent.  Depending on how things work out,
        # ultimately the user may still prefer to just densify, indeed
        # that's proven to be faster thus far, and so is default.
        if not heroic:
            self_dense = DenseJacobian(self)
            result = self_dense.cumsum(axis)
        else:
            # This is much simpler if axis is None or (thus and)
            # dependent_ndim==1
            easy = axis is None or self.dependent_ndim == 1
            if easy:
                new_csc = self.data2d
                nrows = self.shape[0]
            else:
                # OK, we want to take the current 2D matrix and pull it
                # about so that the summed-over axis is now the rows, and
                # the columns are all the remaining dependent axes plus
                # the independent ones.  First, work out the shape of the
                # shuffled array (and the 2D matrix it will be thought of
                # as)
                shape_shuff = list(self.shape)
                shape_shuff.insert(0, shape_shuff.pop(axis))
                shape_shuff_2d = (shape_shuff[0], int(np.prod(shape_shuff[1:])))
                # Turn the matrix into coo
                self_coo = self.data2d.tocoo()
                # Now get the raveled indices for all the elements
                i = np.ravel_multi_index((self_coo.row, self_coo.col), self_coo.shape)
                # Now get these all as unravelled indices across all the axes
                i = list(np.unravel_index(i, self.shape))
                # Now rearrange this list of indices to pull the target one to the
                # front.  Do this by popping the axis in question into a row axis
                row = i.pop(axis)
                # And merging the remainder into a column index
                col = np.ravel_multi_index(i, shape_shuff[1:])
                # Make this a new coo matrix - this is what actually
                # performs the transpose
                new_coo = sparse.coo_matrix(
                    (self_coo.data, (row, col)), shape=shape_shuff_2d
                )
                # Turn this to a csc matrix, this will be what we cumsum over the rows
                new_csc = new_coo.tocsc()
                nrows = shape_shuff[0]
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
                # Now we need to transpose this back to the original
                # shape.  Reverse the steps above.
                intermediate_coo = intermediate.tocoo()
                # First ravel the combined row/column index
                i = np.ravel_multi_index(
                    (intermediate_coo.row, intermediate_coo.col), intermediate_coo.shape
                )
                # Now get these all as unravelled indices across the board
                i = list(np.unravel_index(i, shape_shuff))
                # Now rearrange this to put things back in their proper place
                i.insert(axis, i.pop(0))
                # Now ravel them all into one index again
                i = np.ravel_multi_index(i, self.shape)
                # And now make row and column indices out of them
                row, col = np.unravel_index(i, self.shape2d)
                result_coo = sparse.coo_matrix(
                    (intermediate_coo.data, (row, col)), shape=self.shape2d
                )
                result = SparseJacobian(template=self, data=result_coo.tocsc())
        return result

    def diff(self, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
        """diff method for sparse jacobian"""
        # For now at least I'm going to have this go to dense.
        self_dense = SparseJacobian(self)
        return self_dense.diff(n, axis, prepend, append)

    def extract_diagonal(self):
        """Extract the diagonal from a sparse Jacobian"""
        if self.dependent_shape != self.independent_shape:
            raise ValueError("Sparse Jacobian is not square")
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
