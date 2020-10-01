"""The various Jacobians for duals"""

import warnings
import numpy as np
import scipy.sparse as sparse
import itertools
import inspect
import copy

__all__ = [ "dljacobian_base", "dljacobian_diagonal", "dljacobian_dense", "dljacobian_sparse" ]

# ----------------------------------------------- Helper routines
# First some support routines for the dljacobian class and general
# usage

def _shapes_broadcastable(shp1, shp2):
    # Test if two shapes can be broadcast together
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True

def _broadcasted_shape(shp1, shp2):
    # Return the broadcasted shape of the two arguments
    # Also check's they're legal
    result = []
    for a, b in itertools.zip_longest(shp1[::-1], shp2[::-1], fillvalue=1):
        if a == 1 or b == 1 or a == b:
            result.append(max(a,b))
        else:
            raise ValueError(f"Arrays not broadcastable {shp1} and {shp2}")
    return tuple(result[::-1])

def _array_to_sparse_diagonal(x):
    """Turn an ndarray into a diagonal, stored as csc"""
    x_ = np.array(x).ravel()
    result = sparse.diags(x_, 0, format='csc')
    return result

def _prepare_jacobians_for_binary_op(a, b):
    """Take two Jacobians about to have something binary done to them and
    return their contents in mutually compatible form from a units
    perspective, and as efficiently as possible"""
    # Note that this code does not need to worry about broadcasting,
    # as that is handled elsewhere in this class, and invoked by the
    # methods in dlarray.  Note that the only thing we need to prepare
    # for here is addition, so we don't need to predict the shape of the result.
    scale = 1.0
    if b.dependent_unit != a.dependent_unit:
        scale *= b.dependent_unit._to(a.dependent_unit)
    if b.independent_unit != a.independent_unit:
        scale /= b.independent_unit._to(a.independent_unit)
    # Now go throught the various type combinations
    if type(a) is type(b):
        # If they are both the same type, then things are pretty
        # straight forward.  If they are the same type, then their
        # data attributes are in the same form.
        if type(a) is dljacobian_sparse:
            a_ = a.data2d
            b_ = b.data2d
        else:
            a_ = a.data
            b_ = b.data
        result_type = type(a)
    elif type(a) is dljacobian_diagonal:
        # If a is diagonal (and by implication b is not otherwise the
        # above code would have handled things), then prompte a to
        # sparse and use the 2d view of b
        a_ = _array_to_sparse_diagonal(a.data)
        b_ = b.data2d
        result_type = type(b)
    elif type(b) is dljacobian_diagonal:
        # This is the converse case
        a_ = a.data2d
        b_ = _array_to_sparse_diagonal(b.data)
        result_type = type(a)
    elif type(a) is dljacobian_sparse:
        # OK, so, here a must be sparse, b dense
        a_ = a.data2d
        b_ = b.data2d
        result_type = type(b)
    elif type(b) is dljacobian_sparse:
        # Finally, so it must be that a is dense, b sparse
        a_ = a.data2d
        b_ = b.data2d
        result_type = type(a)
    else:
        raise AssertionError("Failed to understand binary Jacobian operation")
    # If needed, put them in the same units by scaling b to be in a's
    # units
    if scale != 1.0:
        b_ = b_.copy()*scale
    return a_, b_, result_type


# ----------------------------------------------- dljacobian_base
class dljacobian_base(object):

    """This is a container for a jacobian "matrix".  The various child
    classes store the information as either diagonal, a dense array,
    or a sparse array.

    """

    def __init__(self, template=None,
                 dependent_unit=None, independent_unit=None,
                 dependent_shape=None, independent_shape=None):
        """Define a new jacobian"""
        def pick(*args):
            return next((item for item in args if item is not None),None)
        # Set up the core metadata
        if isinstance(template, dljacobian_base):
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
            f"Jacobian of type {type(self)}\n" +
            f"Dependent shape is {self.dependent_shape} <{self.dependent_size}>\n" +
            f"Independent shape is {self.independent_shape} <{self.independent_size}>\n" +
            f"Combined they are {self.shape} <{self.size}>\n" +
            f"Dummies are {self._dummy_dependent} and {self._dummy_independent}\n" +
            f"Units are d<{self.dependent_unit}>/d<{self.independent_unit}> = " +
            f"{(self.dependent_unit/self.independent_unit).decompose()}"
            )

    def __repr__(self):
        return self.__str__()
    
    def __neg__(self):
        return type(self)(-self.data, template=self)

    def __add__(self, other):
        s_, o_, result_type = _prepare_jacobians_for_binary_op(self, other)
        return result_type(data=s_+o_, template=self)

    def __subtract__(self, other):
        s_, o_ = _prepare_jacobians_for_binary_op(self, other)
        return result_type(data=s_-o_, template=self)

    def __lshift__(self, unit):
        result = copy.copy(self) # or should this be deepcopy
        result.dependent_unit = unit

    def real(self):
        return type(self)(np.real(self.data), template=self)

    # This routine is called by the child classes to set up for a
    # premul_diag.  It works out the units issues and sets up for
    # broadcasting.
    def _prepare_premul_diag(self, diag):
        # print (f"Asked for premul_diag on {self}\n.... with {diag.shape}")
        if hasattr(diag, "unit"):
            dependent_unit = diag.unit*self.dependent_unit
            diag_ = diag.value
        else:
            dependent_unit = self.dependent_unit
            diag_ = diag
        dependent_shape = _broadcasted_shape(self.dependent_shape, diag_.shape)
        # print (f"Will return dependent_shape={dependent_shape}")
        return diag_, dependent_unit, dependent_shape

    def flatten(self, order='C'):
        """flatten a jacobian"""
        return self.reshape((self.dependent_size,), order=order)

    def nan_to_num(self, copy=True, nan=0.0, posinf=None, neginf=None):
        return self.__class__(
            template=self, data=np.nan_to_num(self.data, copy=copy, nan=nan, posinf=posinf, neginf=neginf))

    def to(self, unit):
        """Change the dependent_unit for a Jacobian"""
        if unit == self.dependent_unit:
            return self
        scale = self.dependent_unit._to(unit) * (unit/self.dependent_unit)
        # print (f"Scaling from {self.dependent_unit} to {unit}, factor={scale}")
        return self.scalar_multiply(scale)

    def decompose(self):
        """Decompose the dependent_unit for a Jacobian"""
        raise NotImplementedError("Should not be needed")
        print (f"In decompose comes in with {self.dependent_unit}")
        unit = self.dependent_unit.decompose()
        print (f"In decompose, try to give unit {unit}")
        result = self.to(unit)
        print (f"In decompose goes out with {self.dependent_unit}")
        return result

    def scalar_multiply(self, scale):
        """Multiply Jacobian by a scalar"""
        self.dependent_unit *= scale.unit
        self.data *= scale.value
        return self
       
# ----------------------------------------------- dljacobian_diagonal
class dljacobian_diagonal(dljacobian_base):
    """dljacobian that's really a diagonal"""

    def __init__(self, data, template=None, **kwargs):
        if isinstance(data, dljacobian_base):
            if template is None:
                template = data
            else:
                raise ValueError("Cannot supply template with jacobian for data simultaneously")
        super().__init__(template=template, **kwargs)
        if self.dependent_shape != self.independent_shape:
            raise ValueError(
                "Attempt to create a diagonal Jacobian that is not square")
        if isinstance(data, dljacobian_base):
            if isinstance(data, dljacobian_diagonal):
                data_ = data.data
            else:
                raise ValueError("Can only create diagonal Jacobians from other diagonals")
        else:
            data_ = data
        if data_.shape != self.dependent_shape:
            raise ValueError(
                "Attempt to create a jacobian_diagonal using wrong-shaped input")
        self.data = data_

    def __str__(self):
        return super().__str__() + f"\ndata is {self.data.shape}"

    def _getjitem(self, new_shape, key):
        """A getitem type method for diagonal Jacobians"""
        # OK, once we extract items, this will no longer be diagonal,
        # so we convert to sparse before doing the subset.
        self_sparse = dljacobian_sparse(data=self)
        result = self_sparse._getjitem(new_shape, key)
        return result

    def _setjitem(self, key, value):
        """A setitem type method for diagonal Jacobians"""
        # OK, once we insert items, this will no longer be diagonal,
        # so we convert to sparse before doing the subset
        self = dljacobian_sparse(data=self)
        self._setjitem(key, value)

    def broadcast_to(self, shape):
        """Broadcast diagonal jacobian to new dependent shape"""
        # OK, once you broadcast a diagonal, it is not longer,
        # strictly speaking, a diagonal So, convert to sparse and
        # broadcast that.  However, don't bother doing anything if
        # there is no actual broadcast going on.
        if shape == self.dependent_shape:
            return self
        self_sparse = dljacobian_sparse(self)
        return self_sparse.broadcast_to(shape)

    def reshape(self, shape, order='C'):
        # OK, once you reshape a diagonal, it is not longer,
        # strictly speaking, a diagonal So, convert to sparse and
        # reshape that.  However, don't bother doing anything if
        # there is no actual reshape going on.
        if shape == self.dependent_shape:
            return self
        self_sparse = dljacobian_sparse(self)
        return self_sparse.reshape(shape, order)

    def premul_diag(self, diag):
        """Diagonal premulitply for diagonal Jacobian"""
        diag_, dependent_unit, dependent_shape = self._prepare_premul_diag(diag)
        if dependent_shape == self.independent_shape:
            return dljacobian_diagonal(diag_*self.data, template=self,
                                       dependent_unit=dependent_unit)
        else:
            return dljacobian_sparse(self).premul_diag(diag)


    def insert(self, obj, axis, dependent_shape):
        """insert method for diagonal Jacobian"""
        # By construction this is no longer diagonal once inserted to
        # change to sparse and insert there.
        self_sparse = dljacobian_sparse(self)
        return self_sparse.insert(obj, axis, dependent_shape)

    def sum(self, dependent_shape, axis=None, dtype=None, keepdims=False):
        """Performs sum for the diagonal Jacobians"""
        # Once we take the sum, along any or all axes, the jacobian is
        # no longer diagonal by construction, so it needs to be
        # converted to sparse.
        self_sparse = dljacobian_sparse(self)
        return self_sparse.sum(dependent_shape, axis=axis, dtype=dtype,
                               keepdims=keepdims)

    def cumsum(self, axis):
        """Perform cumsum for the diagonal Jacobians"""
        # Once it's been cumsummed then it's no longer diagonal.  For
        # that matter it's not going to be particularly sparse either,
        # so we might as well convert it to dense.
        return dljacobian_dense(self).cumsum(axis)

    def diagonal(self):
        """Return a diagonal Jacobian's contents as just the diagonal (shape=dependent_shape)"""
        return self.data << (self.dependent_unit/self.independent_unit)

    # The reaons we have extract_diagonal and diagonal is that diagonal is
    # only populated for diagonal Jacobians.  extract_diagonal is
    # populated for all.
    def extract_diagonal(self):
        """Extract the diagonal from a diagonal Jacobian"""
        return self.diagonal()

    def todensearray(self):
        self_dense = dljacobian_dense(self)
        return self_dense.todensearray()

    def to2darray(self):
        self_sparse = dljacobian_sparse(self)
        return self_sparse.to2darray()
    
    def to2ddensearray(self):
        self_dense = dljacobian_dense(self)
        return self_dense.to2ddensearray()

# ----------------------------------------------- dljacobian_dense
class dljacobian_dense(dljacobian_base):
    """A dljacobian that's a full on ndarray"""

    def __init__(self, data, template=None, **kwargs):
        if isinstance(data, dljacobian_base):
            if template is None:
                template = data
            else:
                raise ValueError("Cannot supply template with jacobian data simultaneously")
        super().__init__(template=template, **kwargs)
        if isinstance(data, dljacobian_base):
            if isinstance(data, dljacobian_diagonal):
                data_ = np.reshape(_array_to_sparse_diagonal(data.data).toarray(), data.shape)
            elif isinstance(data, dljacobian_dense):
                data_ = data.data
            elif isinstance(data, dljacobian_sparse):
                data_ = np.reshape(data.data2d.toarray(), template.shape)
            else:
                raise ValueError("Unrecognized type for input jacobian")
        else:
            data_ = data
        if data_.shape != self.shape:
            # print (f"\n\nIn atemping to store {data_.shape} into {self.shape}")
            # print (f"Shapes are {self.dependent_shape}, {self.independent_shape}, and {self.shape}")
            raise ValueError(
                "Attempt to create jacobian_dense with wrong-shaped input")
        self.data = data_
        self.data2d = np.reshape(self.data, [self.dependent_size, self.independent_size])

    def __str__(self):
        return (
            super().__str__() +
            f"\ndata is {self.data.shape}\n" +
            f"data2d is {self.data2d.shape}"
            )

    def _getjitem(self, new_shape, key):
        """A getitem type method for dense Jacobians"""
        try:
            jkey = list(key)
        except TypeError:
            jkey = [key]
        extra = [np.s_[:]]*self.independent_ndim
        jkey = tuple(jkey+extra)
        result_ = self.data.__getitem__(jkey)
        new_full_shape = new_shape + self.independent_shape
        # try:
        result_.shape = new_full_shape
        # except AttributeError:
        #    warnings.warn("dljacogian_dense._getjitem had to make a copy")
        #    result_ = np.reshape(result_, new_full_shape)
        return dljacobian_dense(data=result_, template=self,
                                dependent_shape=new_shape)

    def _setjitem(self, key, value):
        """A getitem type method for dense Jacobians"""
        if value is not None:
            self_, value_, result_type = _prepare_jacobians_for_binary_op(self, value)
            if result_type != type(self):
                return TypeError("Jacobian is not of correct time to receive new contents")
        else:
            value_ = 0.0
        try:
            jkey = list(key)
        except TypeError:
            jkey = [key]
        extra = [np.s_[:]]*self.independent_ndim
        jkey = tuple(jkey+extra)
        self.data.__setitem__(jkey, value_)

    def broadcast_to(self, shape):
        """Broadcast dense Jacobian to new dependent_shape"""
        # Don't bother doing anything if the shape is already good
        if shape == self.dependent_shape:
            return self
        full_shape = shape + self.independent_shape
        result_ = np.broadcast_to(self.data, full_shape)
        return dljacobian_dense(data=result_, template=self,
                                dependent_shape=shape)

    def reshape(self, shape, order='C'):
        """reshape dense Jacobian"""
        # Don't bother doing anything if the shape is already good
        if shape == self.dependent_shape:
            return self
        try:
            full_shape = shape + self.independent_shape
        except TypeError:
            full_shape = (shape,) + self.independent_shape
        result_ = np.reshape(self.data, full_shape, order)
        return dljacobian_dense(data=result_, template=self,
                                dependent_shape=shape)
    
    def premul_diag(self, diag):
        """Diagonal premulitply for dense Jacobian"""
        diag_, dependent_unit, dependent_shape = self._prepare_premul_diag(diag)
        try:
            # print (f"OK here, {diag_.shape}, {self.dependent_shape} {self._dummy_independent}")
            diag_ = np.reshape(diag_, (diag.shape + self._dummy_independent))
            # print (f"Gets {diag_.shape}")
            # This will fail for scalars, but that's OK scalars don't
            # need to be handled specially
        except ValueError:
            pass
        return dljacobian_dense(diag_*self.data, template=self,
                                dependent_unit=dependent_unit,
                                dependent_shape=dependent_shape)

    def insert(self, obj, axis, dependent_shape):
        """insert method for dense Jacobian"""
        # print (f"Doing insert on {self}")
        # print (f"obj is {obj}")
        # print (f"axis is {axis}")
        if axis is None:
            jaxis = 0
        else:
            jaxis = axis
        data = np.insert(self.data, obj, 0.0, jaxis)
        # print (f"data comes back as {data.shape}")
        return dljacobian_dense(data, template=self,
                                dependent_shape=dependent_shape)

    def sum(self, dependent_shape, axis=None, dtype=None, keepdims=False):
        """Performs sum for the dense Jacobians"""
        # Negative axis requests count backwards from the last index,
        # but the Jacobians have the independent_shape appended to
        # their shape, so we need to correct for that (or not if its positive)
        if axis is None:
            jaxis = tuple(range(self.dependent_ndim))
        else:
            try:
                jaxis = tuple(a if a>=0 else a-self.independent_ndim for a in axis)
            except TypeError:
                jaxis = axis if axis >= 0 else axis-self.independent_ndim
        return dljacobian_dense(
            data=np.sum(self.data, axis=jaxis, dtype=dtype,
                        keepdims=keepdims),
            template=self, dependent_shape=dependent_shape)

    def cumsum(self, axis):
        """Perform cumsum for a dense Jacobian"""
        return dljacobian_dense(template=self,
                                data=np.cumsum(self.data, axis))

    def extract_diagonal(self):
        """Extract the diagonal from a dense Jacobian"""
        if self.dependent_shape != self.independent_shape:
            raise ValueError("Dense Jacobian is not square")
        result_ = np.reshape(self.data2d.diagonal(), self.dependent_shape)
        return result_ << (self.dependent_unit/self.independent_unit)

    def todensearray(self):
        return self.data << (self.dependent_unit/self.independent_unit)

    def to2ddensearray(self):
        return self.data2d << (self.dependent_unit/self.independent_unit)

    def to2darray(self):
        return self.to2ddensearray()

# ----------------------------------------------- dljacobian_sparse
class dljacobian_sparse(dljacobian_base):
    """A dljacobian that's stored as sparse 2D array under the hood"""

    def __init__(self, data, template=None, **kwargs):
        """Create a new sparse jacobian"""
        # This kind of Jacobian can only be initialized using another
        # sparse jacobian or by a diagonal one.
        if isinstance(data, dljacobian_base):
            if template is None:
                template = data
            else:
                raise ValueError("Cannot supply template with jacobian data simultaneously")
        super().__init__(template=template, **kwargs)
        if isinstance(data, dljacobian_sparse):
            data2d_ = data.data2d
        elif isinstance(data, dljacobian_diagonal):
            data2d_ = _array_to_sparse_diagonal(data.data.ravel())
        elif isinstance(data, dljacobian_dense):
            data2d_ = sparse.csc_matrix(data.data2d)
        elif type(data) is sparse.csc_matrix:
            data2d_ = data
        else:
            raise TypeError("Values supplied to dljacobian_sparse are not suitable")
        if data2d_.shape != self.shape2d:
            # print (data2d_.shape, self.shape2d)
            raise ValueError(
                "Attempt to create jacobian_sparse with wrong-sized input")
        else:
            self.data2d = data2d_
            
    def __str__(self):
        """Provide a string summary of a sparse Jacobian"""
        suffix = ( f"\ndata2d is {self.data2d.shape}" +
                   f" with {self.data2d.nnz} numbers stored")
        return super().__str__() + suffix


    def _getjitem(self, new_shape, key):
        """A getitem type method for sparse Jacobians"""
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
            jSlice = dependent_slice + (np.s_[:],)
        except TypeError:
            jSlice = (dependent_slice,np.s_[:])
        try:
            if len(jSlice) > 2:
                raise TypeError("Dummy raise to fall back to dense")
            result_ =  self.data2d.__getitem__(jSlice)
            return dljacobian_sparse(data=result_, template=self,
                                     dependent_shape=new_shape)
        except TypeError:
            warnings.warn("dljacobian_sparse._getjitem had to fall back to dense")
            self_dense = dljacobian_dense(self)
            return self_dense._getjitem(new_shape, key)

    def _setjitem(self, key, value):
        """A setitem type method for dense Jacobians"""
        raise NotImplementedError("Not (yet) written the setitem capability for sparse Jacobians")

    def broadcast_to(self, shape):
        """Broadcast the dependent vector part of a sparse Jacobian to another shape"""
        # Don't bother doing anything if the shape is already good
        if shape == self.dependent_shape:
            return self
        if not _shapes_broadcastable(self.dependent_shape, shape):
            raise ValueError("Unable to broadcast dljacobian_sparse to new shape")
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
        M = sparse.csc_matrix(sparse.coo_matrix((one,(inew,iold)), shape=(inew.size,self.dependent_size)))
        # Now do a matrix multiply to accomplish what broadcast tries
        # to do.
        result_ = M @ self.data2d
        return dljacobian_sparse(data=result_, template=self, dependent_shape=shape)
    
    def reshape(self, shape, order='C'):
        """Reshape a sparse Jacobian to a new dependent vector"""
        # Don't bother doing anything if the shape is already good
        if shape == self.dependent_shape:
            return self
        # For the sparse jacobians, which never really have a shape
        # anyway, this simply involves updating the shapes of record
        # (assuming it's a valid one).
        if int(np.prod(shape)) != self.dependent_size:
            raise ValueError("Unable to reshape dljacobian_sparse to new shape")
        return dljacobian_sparse(data=self.data2d, template=self, dependent_shape=shape)
    
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
        out_ = sparse.csc_matrix((out_data, self.data2d.indices, self.data2d.indptr), shape=self.data2d.shape)
        return dljacobian_sparse(out_, template=self,
                                 dependent_unit=dependent_unit,
                                 dependent_shape=dependent_shape)
    
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
        self_coo = sparse.coo_matrix((self_coo.data,(row,self_coo.col)),
                                     shape=(dependent_size, self.independent_size))
        return dljacobian_sparse(template=self, data=self_coo.tocsc(),
                                   dependent_shape=dependent_shape)

    
    def sum(self, dependent_shape, axis=None, dtype=None, keepdims=False):
        """Perform sum for the sparse Jacobians"""
        # Two different approaches, depending on whether axis is supplied
        if axis is None:
            # OK, here we want to sum over all the dependent elements, scipy.sparse can do that.
            result_ = np.sum(self.data2d, axis=0, dtype=dtype)
            # Note that result is a dense matrix
            if keepdims:
                result_shape = (1,)*self.dependent_ndim + self.independent_shape
            else:
                result_shape = self.independent_shape
            result_ = np.reshape(np.asarray(result_), result_shape)
            result = dljacobian_dense(template=self, data=result_,
                                      dependent_shape=dependent_shape)
            pass
        else:
            # Here we want to sum over selected indices.  Recall that
            # there is no shame in thinking about things that have the
            # same size as the dependent vector.  To that end, we will
            # formulate this as a sparse-sparse matrix multiply.  In
            # many ways, this operation is the transpose of the
            # broadcast operation, so is constructed in a similar
            # manner.

            # Take the orginal shape and replace the summed-over axes
            # with one.  In the case where keepdims is set, that would
            # be in dependent_shape of course, but we can't rely on
            # that.
            reduced_shape = list(self.dependent_shape)
            # print(f"Start with {reduced_shape}, {type(reduced_shape)}")
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
            M = sparse.csc_matrix(sparse.coo_matrix((one,(ireduced,ioriginal)),
                                                     shape=(ireduced.size,ioriginal.size)))
            result_ = M @ self.data2d
            # Note that by specifying dependent_shape here, supplied
            # by the calling code, we've implicitly taken the value of
            # the keepdims argument into account.
            # print (f"OK, after all that work on {self}\n, summing over {axis}, we got reduced_shape={reduced_shape}")
            # print (f"M is {M.shape}, result_ is {result_.shape} and dependent_shape is {dependent_shape}")
            result = dljacobian_sparse(template=self, data=result_,
                                      dependent_shape=dependent_shape)
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
            self_dense = dljacobian_dense(self)
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
                # Now rearrange this list of indices to pull the target one to the front.
                # Do this by popping the axis in question into a row axis
                row = i.pop(axis)
                # And merging the remainder into a column index
                col = np.ravel_multi_index(i, shape_shuff[1:])
                # Make this a new coo matrix - this is what actually performs the transpose
                new_coo = sparse.coo_matrix((self_coo.data,(row,col)), shape=shape_shuff_2d)
                # Turn this to a csc matrix, this will be what we cumsum over the rows
                new_csc = new_coo.tocsc()
                nrows = shape_shuff[0]
            # We have two choices here, we could do a python loop to build
            # up and store cumulative sums, but I suspect that, while on
            # paper more efficient (reducing the number of additions, it
            # would be slow in reality. Instead, I'll create
            # lower-triangle matrix with ones and zeros and multiply by
            # that.
            lt = sparse.csc_matrix(np.tri(nrows))
            intermediate = lt @ new_csc
            if easy:
                result = dljacobian_sparse(template=self, data=intermediate)
            else:
                # Now we need to transpose this back to the original
                # shape.  Reverse the steps above.
                intermediate_coo = intermediate.tocoo()
                # First ravel the combined row/column index
                i = np.ravel_multi_index((intermediate_coo.row, intermediate_coo.col),
                                         intermediate_coo.shape)
                # Now get these all as unravelled indices across the board
                i = list(np.unravel_index(i, shape_shuff))
                # Now rearrange this to put things back in their proper place
                i.insert(axis, i.pop(0))
                # Now ravel them all into one index again
                i = np.ravel_multi_index(i, self.shape)
                # And now make row and column indices out of them
                row, col = np.unravel_index(i, self.shape2d)
                result_coo = sparse.coo_matrix((intermediate_coo.data,(row,col)), shape=self.shape2d)
                result = dljacobian_sparse(template=self, data=result_coo.tocsc())
        return result
            

    def extract_diagonal(self):
        """Extract the diagonal from a sparse Jacobian"""
        if self.dependent_shape != self.independent_shape:
            raise ValueError("Sparse Jacobian is not square")
        result_ = np.reshape(self.data2d.diagonal(), self.dependent_shape)
        return result_ << (self.dependent_unit/self.independent_unit)

    def todensearray(self):
        self_dense = dljacobian_dense(self)
        return self_dense.todensearray()

    def to2ddensearray(self):
        return self.data2d.toarray() << (self.dependent_unit/self.independent_unit)

    def to2darray(self):
        return self.data2d << (self.dependent_unit/self.independent_unit)

    def nan_to_num(self, copy=True, nan=0.0, posinf=None, neginf=None):
        if copy:
            data2d = self.data2d.copy()
        else:
            data2d = self.data2d
        data2d.data = np.nan_to_num(data2d.data, copy=False, nan=nan, posinf=posinf, neginf=neginf)
        return self.__class__(template=self, data=data2d)
 
    def scalar_multiply(self, scale):
        """Multiply Jacobian by a scalar"""
        self.dependent_unit *= scale.unit
        self.data2d *= scale.value
        return self
      

