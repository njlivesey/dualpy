"""Class for dense jacobians"""
import numpy as np

from mls_scf_tools.util import linear_interpolation_indices_and_weights

from .jacobian_helpers import (
    _array_to_sparse_diagonal,
    _prepare_jacobians_for_binary_op,
)
from .base_jacobian import BaseJacobian


__all__ = ["DenseJacobian"]


class DenseJacobian(BaseJacobian):
    """A dljacobian that's a full on ndarray"""

    def __init__(self, data=None, template=None, data2d=None, **kwargs):
        from .sparse_jacobians import SparseJacobian
        from .diagonal_jacobians import DiagonalJacobian

        if data is not None and data2d is not None:
            raise ValueError("Cannot supply both data and data2d")
        if isinstance(data, BaseJacobian):
            if template is None:
                template = data
            else:
                raise ValueError(
                    "Cannot supply template with jacobian data simultaneously"
                )
        super().__init__(template=template, **kwargs)
        if isinstance(data, BaseJacobian):
            if isinstance(data, DiagonalJacobian):
                data_ = np.reshape(
                    _array_to_sparse_diagonal(data.data).toarray(), data.shape
                )
            elif isinstance(data, DenseJacobian):
                data_ = data.data
            elif isinstance(data, SparseJacobian):
                data_ = np.reshape(data.data2d.toarray(), template.shape)
            else:
                raise ValueError("Unrecognized type for input jacobian")
        else:
            if data is not None:
                data_ = data
            elif data2d is not None:
                data_ = np.reshape(data2d, self.shape)
            else:
                data_ = np.zeros(shape=self.shape, dtype=self.dtype)
        if data_.shape != self.shape:
            raise ValueError("Attempt to create jacobian_dense with wrong-shaped input")
        self.data = data_
        self.data2d = np.reshape(
            self.data, [self.dependent_size, self.independent_size]
        )
        self.dtype = self.data.dtype

    def __str__(self):
        return (
            super().__str__()
            + f"\ndata is {self.data.shape}\n"
            + f"data2d is {self.data2d.shape}"
        )

    def _check(self, name):
        """Integrity checks on dense Jacobian"""
        self._check_jacobian_fundamentals(name)
        assert (
            self.data.shape == self.shape
        ), f"Array shape mismatch for {name}, {self.data.shape} != {self.shape}"
        assert (
            self.data.size == self.size
        ), f"Array size mismatch for {name}, {self.data.size} != {self.size}"

    def _getjitem(self, new_shape, key):
        """A getitem type method for dense Jacobians"""
        key = self._preprocess_getsetitem_key(key)
        jkey = key + (slice(None),) * self.independent_ndim
        result_ = self.data.__getitem__(jkey)
        new_full_shape = new_shape + self.independent_shape
        if result_.shape != new_full_shape:
            raise NotImplementedError(
                "Looks like we do need to do a reshape after all!"
            )
        # result_ = np.reshape(result_, new_full_shape)
        return DenseJacobian(data=result_, template=self, dependent_shape=new_shape)

    def _setjitem(self, key, value):
        """A getitem type method for dense Jacobians"""
        if value is not None:
            self_, value_, result_type = _prepare_jacobians_for_binary_op(
                self, value
            )
            if result_type != type(self):
                return TypeError(
                    "Jacobian is not of correct type to receive new contents"
                )
        else:
            value_ = 0.0
        key = self._preprocess_getsetitem_key(key)
        self.data.__setitem__(key, value_)

    def broadcast_to(self, shape):
        """Broadcast dense Jacobian to new dependent_shape"""
        # Don't bother doing anything if the shape is already good
        if shape == self.dependent_shape:
            return self
        full_shape = shape + self.independent_shape
        result_ = np.broadcast_to(self.data, full_shape)
        return DenseJacobian(data=result_, template=self, dependent_shape=shape)

    def reshape(self, shape, order, parent_flags):
        """reshape dense Jacobian"""
        # Don't bother doing anything if the shape is already good
        if shape == self.dependent_shape:
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
            full_shape = tuple(shape) + tuple(input_jacobian.independent_shape)
        except TypeError:
            full_shape = (shape,) + tuple(input_jacobian.independent_shape)
        result_ = np.reshape(input_jacobian.data, full_shape, order)
        return DenseJacobian(
            data=result_, template=input_jacobian, dependent_shape=shape
        )

    def premul_diag(self, diag):
        """Diagonal premulitply for dense Jacobian"""
        diag_, dependent_unit, dependent_shape = self._prepare_premul_diag(diag)
        try:
            diag_ = np.reshape(diag_, (diag.shape + self._dummy_independent))
            # This will fail for scalars, but that's OK scalars don't
            # need to be handled specially
        except ValueError:
            pass
        return DenseJacobian(
            diag_ * self.data,
            template=self,
            dependent_unit=dependent_unit,
            dependent_shape=dependent_shape,
        )

    def insert(self, obj, values, axis, dependent_shape):
        """insert method for dense Jacobian"""
        jaxis = self._get_jaxis(axis, none="flatten")
        data = np.insert(self.data, obj, 0.0, jaxis)
        return DenseJacobian(data, template=self, dependent_shape=dependent_shape)

    def sum(self, dependent_shape, axis=None, dtype=None, keepdims=False):
        """Performs sum for the dense Jacobians"""
        # Negative axis requests count backwards from the last index,
        # but the Jacobians have the independent_shape appended to
        # their shape, so we need to correct for that (or not if its positive)
        jaxis = self._get_jaxis(axis, none="all")
        return DenseJacobian(
            data=np.sum(self.data, axis=jaxis, dtype=dtype, keepdims=keepdims),
            template=self,
            dependent_shape=dependent_shape,
        )

    def mean(self, dependent_shape, axis=None, dtype=None, keepdims=False):
        """Performs mean for the dense Jacobians"""
        # Negative axis requests count backwards from the last index,
        # but the Jacobians have the independent_shape appended to
        # their shape, so we need to correct for that (or not if its positive)
        jaxis = self._get_jaxis(axis, none="all")
        return DenseJacobian(
            data=np.mean(self.data, axis=jaxis, dtype=dtype, keepdims=keepdims),
            template=self,
            dependent_shape=dependent_shape,
        )

    def cumsum(self, axis):
        """Perform cumsum for a dense Jacobian"""
        jaxis = self._get_jaxis(axis, none="all")
        return DenseJacobian(template=self, data=np.cumsum(self.data, axis=jaxis))

    def diff(
        self, dependent_shape, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue
    ):
        """diff method for dense jacobian"""
        jaxis = self._get_jaxis(axis)
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
            data=result_, template=self, dependent_shape=dependent_shape
        )

    def transpose(self, axes, result_dependent_shape):
        jaxes = self._get_jaxis(axes, none="transpose")
        jaxes = tuple(jaxes) + tuple(range(self.dependent_ndim, self.ndim))
        return DenseJacobian(
            data=self.data.transpose(jaxes),
            template=self,
            dependent_shape=result_dependent_shape,
        )

    def tensordot(self, other, axes, dependent_unit):
        """Compute self(.)other"""
        import sparse as st

        # Note that axes here must be in the list of two lists form, with no negative
        # numbers.
        n_contractions = len(axes[0])
        # With this order of the tensordot, the annoying thing here is that we actually
        # want our independent dimensions (part of self) at the end, so will have to do
        # a transpose.  Let's do the tensor dot anyway.
        if isinstance(self.data, np.ndarray) and isinstance(other, np.ndarray):
            result_ = np.tensordot(self.data, other, axes)
        else:
            result_ = st.tensordot(self.data, other, axes)
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
        result_ = result_.transpose(new_axis_order)
        result_dependent_shape = result_.shape[: -self.independent_ndim]
        return DenseJacobian(
            data=result_,
            dependent_shape=result_dependent_shape,
            dependent_unit=dependent_unit,
            independent_shape=self.independent_shape,
            independent_unit=self.independent_unit,
        )

    def rtensordot(self, other, axes, dependent_unit):
        """Compute other(.)self"""
        # This one is actually easier than regular tensordot, because the axes end up in
        # the right order
        result_ = np.tensordot(other, self.data, axes)
        result_dependent_shape = result_.shape[: -self.independent_ndim]
        return DenseJacobian(
            data=result_,
            template=self,
            dependent_shape=result_dependent_shape,
            dependent_unit=dependent_unit,
        )

    def extract_diagonal(self):
        """Extract the diagonal from a dense Jacobian"""
        if self.dependent_shape != self.independent_shape:
            raise ValueError("Dense Jacobian is not square")
        result_ = np.reshape(self.data2d.diagonal(), self.dependent_shape)
        return result_ << (self.dependent_unit / self.independent_unit)

    def todensearray(self):
        return self.data << (self.dependent_unit / self.independent_unit)

    def to2ddensearray(self):
        return self.data2d << (self.dependent_unit / self.independent_unit)

    def to2darray(self):
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

    def linear_interpolator(self, x_in, axis=-1):
        """Return an interpolator for a given Jacobian axis"""
        return DenseJacobianLinearInterpolator(self, x_in, axis)

    def spline_interpolator(self, x_in, axis=-1):
        """Return an interpolator for a given Jacobian axis"""
        return DenseJacobianSplineInterpolator(self, x_in, axis)


class DenseJacobianLinearInterpolator(object):
    """Interpolates a DenseJacobian along one dependent axis"""

    def __init__(self, jacobian, x_in, axis=-1):
        """Setup an interpolator for a given DenseJacobian"""
        self.jacobian = jacobian
        self.jaxis = jacobian._get_jaxis(axis, none="first")
        self.x_in = x_in

    def __call__(self, x_out):
        """Inpoterpolate a DenseJacobian to new values along an axis"""
        i_lower, i_upper, w_lower, w_upper = linear_interpolation_indices_and_weights(
            self.x_in, x_out
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
        result = (
            self.jacobian.data[tuple(lower_key)] * w_lower
            + self.jacobian.data[tuple(upper_key)] * w_upper
        )
        # Prepare the result and return
        new_dependent_shape = result.shape[: self.jacobian.dependent_ndim]
        return DenseJacobian(
            template=self.jacobian,
            data=result,
            dependent_shape=new_dependent_shape,
        )


class DenseJacobianSplineInterpolator(object):
    """Interpolates Jacobian along one dependent axis"""

    def __init__(self, jacobian, x_in, axis=-1):
        """Setup an interpolator for a given DenseJacobian"""
        import scipy.interpolate as interpolate

        self.jacobian = jacobian
        self.jaxis = jacobian._get_jaxis(axis, none="first")
        self.interpolator = interpolate.CubicSpline(
            x_in, jacobian.data, axis=self.jaxis
        )

    def __call__(self, x_out):
        """Inpoterpolate a DenseJacobian to new values along an axis"""
        result = self.interpolator(x_out)
        new_dependent_shape = result.shape[: self.jacobian.dependent_ndim]
        return DenseJacobian(
            template=self.jacobian,
            data=result,
            dependent_shape=new_dependent_shape,
        )
