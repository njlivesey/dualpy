"""A mixin to provide standard operators for the dlarray class

This is needed because if, instead, one employes the __array_ufunc__ approach, the fact
that dlarray (in its new incarnation) is not an ndarray subclass means that if, for
example, an astropy.units.Quantity is on the left hand side of a binary operation, with
a dlarray (or dlarray_astropy) on the right hand side, then astropy's operator will get
precedence.  Accordingly, instead, we have an agressive approach in this mixin to let
our ufuncs take precedence over all others they're involved with.
"""

import operator

class DualOperatorsMixin:
    """A mixin to provide standard operators for the dlarray class"""
    # Not sure why this is needed or what it adds.
    __slots__ = ()

    def _binary_op(self, other, f, reflexive=False):
        raise NotImplementedError

    def __add__(self, other):
        return self._binary_op(other, self.add)

    def __sub__(self, other):
        return self._binary_op(other, self.sub)

    def __mul__(self, other):
        return self._binary_op(other, self.multiply)

    def __pow__(self, other):
        return self._binary_op(other, self.pow)

    def __truediv__(self, other):
        return self._binary_op(other, self.true_divide)

    # def __floordiv__(self, other):
    #     return self._binary_op(other, self.floordiv)

    # def __mod__(self, other):
    #     return self._binary_op(other, self.mod)

    # def __and__(self, other):
    #     return self._binary_op(other, self.and_)

    # def __xor__(self, other):
    #     return self._binary_op(other, self.xor)

    # def __or__(self, other):
    #     return self._binary_op(other, self.or_)

    def __lt__(self, other):
        return self._binary_op(other, self.lt)

    def __le__(self, other):
        return self._binary_op(other, self.le)

    def __gt__(self, other):
        return self._binary_op(other, self.gt)

    def __ge__(self, other):
        return self._binary_op(other, self.ge)

    def __eq__(self, other):
        return self._binary_op(other, self.eq)

    def __ne__(self, other):
        return self._binary_op(other, self.ne)

    def __radd__(self, other):
        return self._binary_op(other, self.add, reflexive=True)

    def __rsub__(self, other):
        return self._binary_op(other, self.subtract, reflexive=True)

    def __rmul__(self, other):
        return self._binary_op(other, self.multiply, reflexive=True)

    def __rpow__(self, other):
        return self._binary_op(other, self.pow, reflexive=True)

    def __rtruediv__(self, other):
        return self._binary_op(other, self.true_divde, reflexive=True)

    # def __rfloordiv__(self, other):
    #     return self._binary_op(other, self.floordiv, reflexive=True)

    # def __rmod__(self, other):
    #     return self._binary_op(other, self.mod, reflexive=True)

    # def __rand__(self, other):
    #     return self._binary_op(other, self.and_, reflexive=True)

    # def __rxor__(self, other):
    #     return self._binary_op(other, self.xor, reflexive=True)

    # def __ror__(self, other):
    #     return self._binary_op(other, self.or_, reflexive=True)

    def _inplace_binary_op(self, other, f):
        raise NotImplementedError

    def __iadd__(self, other):
        return self._inplace_binary_op(other, self.iadd)

    def __isub__(self, other):
        return self._inplace_binary_op(other, self.isub)

    def __imul__(self, other):
        return self._inplace_binary_op(other, self.imul)

    def __ipow__(self, other):
        return self._inplace_binary_op(other, self.ipow)

    def __itruediv__(self, other):
        return self._inplace_binary_op(other, self.itruediv)

    # def __ifloordiv__(self, other):
    #     return self._inplace_binary_op(other, self.ifloordiv)

    # def __imod__(self, other):
    #     return self._inplace_binary_op(other, self.imod)

    # def __iand__(self, other):
    #     return self._inplace_binary_op(other, self.iand)

    # def __ixor__(self, other):
    #     return self._inplace_binary_op(other, self.ixor)

    # def __ior__(self, other):
    #     return self._inplace_binary_op(other, self.ior)

    def _unary_op(self, f, *args, **kwargs):
        raise NotImplementedError

    def __neg__(self):
        return self._unary_op(self.neg)

    def __pos__(self):
        return self._unary_op(self.pos)

    def __abs__(self):
        return self._unary_op(self.abs)

    # def __invert__(self):
    #     return self._unary_op(self.invert)

    # def round(self, *args, **kwargs):
    #     return self._unary_op(ops.round_, *args, **kwargs)

    # def argsort(self, *args, **kwargs):
    #     return self._unary_op(ops.argsort, *args, **kwargs)

    # def conj(self, *args, **kwargs):
    #     return self._unary_op(ops.conj, *args, **kwargs)

    # def conjugate(self, *args, **kwargs):
    #     return self._unary_op(ops.conjugate, *args, **kwargs)

    __add__.__doc__ = operator.add.__doc__
    __sub__.__doc__ = operator.sub.__doc__
    __mul__.__doc__ = operator.mul.__doc__
    __pow__.__doc__ = operator.pow.__doc__
    __truediv__.__doc__ = operator.truediv.__doc__
    # __floordiv__.__doc__ = operator.floordiv.__doc__
    # __mod__.__doc__ = operator.mod.__doc__
    # __and__.__doc__ = operator.and_.__doc__
    # __xor__.__doc__ = operator.xor.__doc__
    # __or__.__doc__ = operator.or_.__doc__
    __lt__.__doc__ = operator.lt.__doc__
    __le__.__doc__ = operator.le.__doc__
    __gt__.__doc__ = operator.gt.__doc__
    __ge__.__doc__ = operator.ge.__doc__
    __eq__.__doc__ = operator.eq.__doc__
    __ne__.__doc__ = operator.ne.__doc__
    __radd__.__doc__ = operator.add.__doc__
    __rsub__.__doc__ = operator.sub.__doc__
    __rmul__.__doc__ = operator.mul.__doc__
    __rpow__.__doc__ = operator.pow.__doc__
    __rtruediv__.__doc__ = operator.truediv.__doc__
    # __rfloordiv__.__doc__ = operator.floordiv.__doc__
    # __rmod__.__doc__ = operator.mod.__doc__
    # __rand__.__doc__ = operator.and_.__doc__
    # __rxor__.__doc__ = operator.xor.__doc__
    # __ror__.__doc__ = operator.or_.__doc__
    __iadd__.__doc__ = operator.iadd.__doc__
    __isub__.__doc__ = operator.isub.__doc__
    __imul__.__doc__ = operator.imul.__doc__
    __ipow__.__doc__ = operator.ipow.__doc__
    __itruediv__.__doc__ = operator.itruediv.__doc__
    # __ifloordiv__.__doc__ = operator.ifloordiv.__doc__
    # __imod__.__doc__ = operator.imod.__doc__
    # __iand__.__doc__ = operator.iand.__doc__
    # __ixor__.__doc__ = operator.ixor.__doc__
    # __ior__.__doc__ = operator.ior.__doc__
    __neg__.__doc__ = operator.neg.__doc__
    __pos__.__doc__ = operator.pos.__doc__
    __abs__.__doc__ = operator.abs.__doc__
    # __invert__.__doc__ = operator.invert.__doc__
    # round.__doc__ = ops.round_.__doc__
    # argsort.__doc__ = ops.argsort.__doc__
    # conj.__doc__ = ops.conj.__doc__
    # conjugate.__doc__ = ops.conjugate.__doc__


