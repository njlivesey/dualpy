"""Some classes and routines to help deal with sparse Jacobians"""

import copy
from abc import abstractmethod
from typing import Sequence, Optional, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

if TYPE_CHECKING:
    from .sparse_jacobians import SparseJacobian

# --------------------------------------------------------------------- Routines
#
# First we have some low-level helper routines that manipulate sparse matrices,
# rearranging axes etc. While these can indeed be run by any user, if you're using them
# to specifically handle Jacobians, you would probably be better off using the classes
# that wrap these in Jacobian-aware framework, below.


def rearrange_2d(
    matrix: sparse.spmatrix,
    original_shapes: Sequence[int],
    axes: Optional[Sequence] = None,
    promote: Optional[int] = None,
    demote: Optional[int] = None,
):
    """Rearrange a 2D sparse matrix representing a tensor into a different axis order

    Parameters
    ----------
    matrix : (sparse) matrix-like
        A 2D sparse matrix (typically csc format) that is really storing a sparse tensor
        with the axes raveled into to two collections of axis to fake a 2D sparse
        matrix.
    original_shapes : list of two lists of int
        A list of two lists giving the shapes that ravel to make the rows and columns of
        input matrix.
    axes : list of two lists of int (optional)
        The recipe for the result - which axes from the original form the two sets of
        axes raveled to get the two axes of the fake 2D matrix
    promote : A single axis to move to the front (optional)
        A shortcut to simply move one axis to the front
    demote : A single axis to move to the back (optional)
        A shortcut to simply move one axis to the back

    Returns
    -------
    result : (sparse) matrix-like
        A 2D sparse matrix that's been transposed as requested
    new_shape : list of two lists of int
        The new shape of the result
    undo : list of two lists of ints
        When passed (along with the result) to this same routine, as the new "axes"
        argument provides an "undo" operation.

    """
    # Work out what we've been asked to do
    if len(original_shapes) != 2:
        raise ValueError("Wrong length (not 2) for original_shapes argument")
    n0_in = len(original_shapes[0])
    n1_in = len(original_shapes[1])
    n_in = n0_in + n1_in
    original_shape = original_shapes[0] + original_shapes[1]
    # Work out what the recipe is for the result
    if promote is not None or demote is not None:
        if axes is not None:
            raise ValueError("Cannot set both axes and promote/demote")
        if promote is not None and demote is not None:
            raise ValueError("Cannot set both promote and demote")
        remainder = list(range(n_in))
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
    # Force to int because prod returns float in case where one other term is scalar
    # (i.e., for array of zero length)
    rearranged_shape_2d = [
        np.prod(rearranged_shape[:n0_out], dtype=int),
        np.prod(rearranged_shape[n0_out:], dtype=int),
    ]
    # Convert the matrix to coo form
    matrix_coo = matrix.tocoo()
    # Now get the raveled indices for all the elements
    i_raveled = np.ravel_multi_index((matrix_coo.row, matrix_coo.col), matrix_coo.shape)
    # Now get these all as unravelled indices across all the axes
    i_all_original = np.unravel_index(i_raveled, original_shape)
    # Now rearrange this list of indices according to the prescription supplied.
    i_all_rearranged = [i_all_original[i] for i in all_axes]
    # Now ravel the new row/column indices.  If one or other is a scalar set the raveled
    # index to None
    if n0_out > 0:
        i0_new = np.ravel_multi_index(
            i_all_rearranged[:n0_out], rearranged_shape[:n0_out]
        )
    else:
        i0_new = None
    if n0_out < n_out:
        i1_new = np.ravel_multi_index(
            i_all_rearranged[n0_out:], rearranged_shape[n0_out:]
        )
    else:
        i1_new = None
    # Handle the scalar cases
    if i0_new is None and i0_new is None:
        raise NotImplementedError(
            "The scalar on scalar case has not been tested (perhaps it works fine)"
        )
        # i0_new = 0
        # i1_new = 0
    elif i0_new is None:
        i0_new = i1_new * 0
    elif i1_new is None:
        i1_new = i0_new * 0
    else:
        pass
    # Make the new matrix - this is what does the actual rearrangement
    result = sparse.csc_array(
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


def gather_sparse_rows_to_dense(
    sparse_array: sparse.spmatrix,
) -> tuple[NDArray, NDArray]:
    """Gather all unique non-zero columns from a sparse matrix into a dense matrix.

    This is intended for sparse matrices where each row carries information in much the
    same column as the other rows.  Typically the single column is some ravel of a
    higher dimensional space, where each row traces a similar route through that space.
    Think of the example of a mutli-channel radiative transfer through a two dimensional
    (or higher dimensionality) atmosphere, for a given atmospheric path, the parts of
    the atmosphere encompassed by the path will be the same channel by channel.

    Parameters
    ----------
    sparse_array : sparse.spmatrix
        Sparse matrix

    Returns
    -------
    gathered_array : NDArray
        Dense matrix [<rows>, <unique columns>]
    scatter_structure : NDArray
        Indices of columns in the original sparse matrix
    """
    # Convert the sparse matrix to csc_array form, needed for efficient indexing below.
    # Note that it needs to be a sparse matrix, not a sparse array, as scipy.sparse
    # doesn't support the indexing we need for the latter
    sparse_array = sparse.csc_array(sparse_array)
    # Find all unique column indices with non-zero entries
    nonzero_cols = sparse_array.nonzero()[1]
    unique_cols = np.unique(nonzero_cols)
    # Create a dense matrix [frequency, num_unique_columns]
    gathered_array = np.empty((sparse_array.shape[0], len(unique_cols)))
    # Fill the dense matrix
    for i, col in enumerate(unique_cols):
        gathered_array[:, i] = sparse_array[:, col].toarray().flatten()
    return gathered_array, unique_cols


def scatter_dense_to_sparse(
    gathered_array: NDArray, scatter_structure: NDArray, original_shape: Sequence[int]
) -> sparse.csc_array:
    """Scatter a dense matrix back into a sparse matrix.

    This is intended to undo the gather routine above, or perhaps scatter something
    that's the result of a calculation on something that was gathered.

    Parameters
    ----------
    gathered_array : NDArray
        Dense matrix to scatter
    scatter_structure : NDArray
        Indices of columns in the original sparse matrix
    original_shape : tuple[int]
        Shape of the original sparse matrix to scatter back into

    Returns
    -------
    scattered_array : sparse.csc_array
        Reconstructed sparse matrix.
    """
    # Initialize a sparse matrix with the original shape
    scattered_array = sparse.lil_array(original_shape, dtype=gathered_array.dtype)
    for i, col in enumerate(scatter_structure):
        scattered_array[:, col] = gathered_array[:, i].reshape(-1, 1)
    return scattered_array.tocsc()


# -------------------------------------------------------------------- Classes
#
# OK, now we have some classes the handle specfic tasks. Fundamentallyl, we pull one
# axis of the Jacobian (from within the dependent quantity) to the front and either
# store that as a still-sparse matrix, or store a dense one.  The dense one is optimal
# for cases where each "row" (i.e., along the promoted axix) has a fairly similar set of
# columns populated.


class BaseRearrangedSparseJacobian:
    """An base class for a sparse Jacobian rearranger"""

    @abstractmethod
    def __init__(self, jacobian: "SparseJacobian"):
        """Do some base-level initialization and checking for a rearranger"""
        self.source_dependent_shape = jacobian.dependent_shape
        self.source_dependent_size = jacobian.dependent_size
        self.source_dependent_ndim = jacobian.dependent_ndim
        self.source_independent_shape = jacobian.independent_shape
        self.source_independent_size = jacobian.independent_size
        self.source_independent_ndim = jacobian.independent_ndim
        self.source_shape = jacobian.shape
        self.source_size = jacobian.size
        self.source_ndim = jacobian.ndim
        #
        self.source_dependent_unit = jacobian.dependent_unit
        self.source_independent_unit = jacobian.independent_unit

    @abstractmethod
    def undo(self, array: NDArray | sparse.spmatrix):
        """Apply the inverse of a rearranger to a supplied array"""


class SparselyRearrangedSparseJacobian(BaseRearrangedSparseJacobian):
    """Contains a rearranged form of a SparseJacobian

    A requested axis is moved to the first (rows) axis, and all the others are ravelled
    together as the columns.

    Parameters
    ----------
    jacobian : SparseJacobian
        The Jacobian to develop a rearranging process for
    promoted_axis  : int
        A single axis (in the dependent quantity) to move to the front.
    """

    def __init__(
        self,
        jacobian: "SparseJacobian",
        promoted_axis: int,
    ):
        """Initiailizes the rearrangement"""
        # Store the key information in self
        super().__init__(jacobian)
        # Work out what the recipe is for the result
        self.promoted_axis = promoted_axis
        remainder = list(range(self.source_ndim))
        remainder.pop(promoted_axis)
        axes = [[promoted_axis], remainder]
        self.axes = axes
        # Final setups
        original_shapes = [self.source_dependent_shape, self.source_independent_shape]
        # Invoke the helper routine to do the actual work
        self.matrix, self.rearranged_shape, self.undo_axes = rearrange_2d(
            jacobian.data,
            original_shapes=original_shapes,
            axes=axes,
        )
        self.rearranged_shape_2d = self.matrix.shape

    def undo(
        self,
        array,
        dependent_unit=None,
    ) -> "SparseJacobian":
        """Reverse the effect of our rearrangement on a supplied matrix

        Typically this matrix is not the same as "ours" but is the result of doing some
        manipulation on ours.  This might involve a change in one of the dependent
        dimensions, being the one that was promoted to the front.  Here we apply some
        intelligence to work out what that is.
        """
        # pylint: disable-next=import-outside-toplevel
        from .sparse_jacobians import SparseJacobian

        # Check the consistency
        if array.shape[1] != self.rearranged_shape_2d[1]:
            raise ValueError("Shape mismatch on demoted dimensions")
        # Handle any changes to the number of rows
        if self.rearranged_shape[0] != array.shape[0]:
            rearranged_shape = copy.deepcopy(self.rearranged_shape)
            rearranged_shape[0] = [array.shape[0]]
            dependent_shape = list(copy.deepcopy(self.source_dependent_shape))
            dependent_shape[self.promoted_axis] = array.shape[0]
            dependent_shape = tuple(dependent_shape)
        else:
            rearranged_shape = self.rearranged_shape
            dependent_shape = self.source_dependent_shape
        # Handle any changes to units
        if dependent_unit is None:
            dependent_unit = self.source_dependent_unit
        result, _, _ = rearrange_2d(
            array,
            original_shapes=rearranged_shape,
            axes=self.undo_axes,
        )
        return SparseJacobian(
            result,
            dependent_shape=dependent_shape,
            dependent_unit=dependent_unit,
            independent_shape=self.source_independent_shape,
            independent_unit=self.source_independent_unit,
        )


class DenselyRearrangedSparseJacobian(SparselyRearrangedSparseJacobian):
    """Contains a rearranged form of a SparseJacobian

    A requested axis is moved to the first (rows) axis, and all the others are ravelled
    together as the columns. Thus far, this is as SparselyRearrangedSparseJacobian, as
    above, and indeed, this class invokes that one.  However, in this case, we go a
    further step and identify the subset of the columns that are needed (on the
    assumption that it is the same for each row) and create a dense matrix that is just
    those columns.

    Parameters
    ----------
    jacobian : SparseJacobian
        The Jacobian to develop a rearranging process for
    promoted_axis : int
        A single axis (in the dependent quantity) to move to the front.
    """

    def __init__(
        self,
        jacobian: "SparseJacobian",
        promoted_axis: int,
    ):
        """Initiailizes the rearrangement"""
        # Call the parent __init__ to set up the sparse rearrangement
        super().__init__(jacobian=jacobian, promoted_axis=promoted_axis)
        # Now we identify the populated columns and generate a dense matrix that
        # corresponds to those.
        self.sparse_nnz = self.matrix.nnz
        self.uncompressed_shape = self.matrix.shape
        self.matrix, self.scatter_structure = gather_sparse_rows_to_dense(self.matrix)
        self.bloat = 100.0 * (self.matrix.size - self.sparse_nnz) / self.sparse_nnz

    def undo(self, array: NDArray, dependent_unit=None):
        """Undoes the rearrangement"""
        # Check the consistency
        if array.shape[1] != self.matrix.shape[1]:
            raise ValueError("Shape mismatch on demoted/compacted dimensions")
        # Handle any changes to the number of rows
        if self.uncompressed_shape[0] != array.shape[0]:
            uncompressed_shape = list(copy.deepcopy(self.uncompressed_shape))
            uncompressed_shape[0] = array.shape[0]
            uncompressed_shape = tuple(uncompressed_shape)
        else:
            uncompressed_shape = self.uncompressed_shape
        # Scatter this back into a sparse matrix
        sparse_intermediate = scatter_dense_to_sparse(
            array,
            scatter_structure=self.scatter_structure,
            original_shape=uncompressed_shape,
        )
        # Now call our parent class to rearrange the sparse matrix into the correct
        # form.  Its result is our result
        return super().undo(sparse_intermediate, dependent_unit=dependent_unit)
