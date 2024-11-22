"""Some classes and routines to help deal with sparse Jacobians"""

from typing import Sequence, Optional
import numpy as np
from numpy.typing import NDArray
from scipy import sparse


# First some helper routines
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
    sparse_matrix: sparse.spmatrix,
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
    sparse_matrix : sparse.spmatrix
        Sparse matrix

    Returns
    -------
    gathered_matrix : NDArray
        Dense matrix [<rows>, <unique columns>]
    scatter_structure : NDArray
        Indices of columns in the original sparse matrix
    """
    # Find all unique column indices with non-zero entries
    nonzero_cols = sparse_matrix.nonzero()[1]
    unique_cols = np.unique(nonzero_cols)
    # Create a dense matrix [frequency, num_unique_columns]
    gathered_matrix = np.empty((sparse_matrix.shape[0], len(unique_cols)))
    # Fill the dense matrix
    for i, col in enumerate(unique_cols):
        gathered_matrix[:, i] = sparse_matrix[:, col].toarray().flatten()
    return gathered_matrix, unique_cols


def scatter_dense_to_sparse(
    gathered_matrix: NDArray, scatter_structure: NDArray, original_shape: Sequence[int]
) -> sparse.csc_array:
    """Scatter a dense matrix back into a sparse matrix.

    This is intended to undo the gather routine above, or perhaps scatter something
    that's the result of a calculation on something that was gathered.

    Parameters
    ----------
    gathered_matrix : NDArray
        Dense matrix to scatter
    scatter_structure : NDArray
        Indices of columns in the original sparse matrix
    original_shape : tuple[int]
        Shape of the original sparse matrix to scatter back into

    Returns
    -------
    scattered_matrix : sparse.csc_matrix
        Reconstructed sparse matrix.
    """
    # Initialize a sparse matrix with the original shape
    scattered_matrix = sparse.lil_matrix(original_shape)
    for i, col in enumerate(scatter_structure):
        scattered_matrix[:, col] = gathered_matrix[:, i].reshape(-1, 1)
    return scattered_matrix.tocsc()
