"""Support routines for testing dualpy"""

from typing import Any

import numpy as np
import pint
from scipy import sparse

import dualpy as dp


def custom_json_serializer(obj: Any):
    """Serialize anything prepatory to storing as json etc.

    Note, this is for testing/gold-brick purposes only, thre is no need to meaningfully
    deserialize later, so we can be cavelier about ignoreing metadata, hints, etc.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pint.Unit):
        return str(obj)
    if isinstance(obj, pint.Quantity):
        return {"magnitude": obj.magnitude, "units": obj.units}
    if isinstance(obj, dp.dlarray):
        return {"variable": obj.variable, "jacobians": obj.jacobians}
    if isinstance(obj, dp.BaseJacobian):
        return {
            "dependent_unit": obj.dependent_unit,
            "independent_unit": obj.independent_unit,
            "dependent_shape": obj.dependent_shape,
            "independent_shape": obj.independent_shape,
            "dependent_size": obj.dependent_size,
            "independent_size": obj.independent_size,
            "dependent_ndim": obj.dependent_ndim,
            "independent_ndim": obj.independent_ndim,
            "shape": obj.shape,
            "ndim": obj.ndim,
            "shape_2d": obj.shape_2d,
            "data": obj.data,
        }
    # pylint: disable-next=protected-access
    if isinstance(obj, sparse._csc.csc_array):
        return {
            "indicies": obj.indices,
            "indptr": obj.indptr,
            "data": obj.data,
        }
    raise TypeError(f"Do not know how to serialize objects of type {type(obj)}")
