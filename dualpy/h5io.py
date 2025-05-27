"""A library for reading/writing duals (or arrays/Quantities) into HDF

Can be used standa lone, but is more typically used as part of larger HDF I/O libraries
for more complex classes.
"""

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional

import pint
import h5py
import numpy as np
from numpy.typing import NDArray

# Some type hints for what follows
GroupOrFile = h5py.Group | h5py.File
H5Location = str | Path | h5py.Group | h5py.File


@contextmanager
def open_h5_group(
    location: H5Location,
    mode: Optional[str] = None,
    group: Optional[str] = None,
    create_group: Optional[bool] = False,
):  # -> ContextManager[h5py .Group]
    # (pylance doesn't like the type hint on the output and ChatGPT says it's got a
    # point so I'm omitting it)
    """Opens (creating if needed/requested) a group in an HDF5-file

    Intended to be used as a context manager

    Parameters
    ----------
    location : str | Path | h5py.File
        The HDF5 file/group to access, can be opened already, or supplied as a
        filename/Path.  Note, if it's supplied as a name, then it cannot include the
        group name.
    mode : str
        Passed to h5py.File, see documentation for that ("r", "r+", "w", "w-", "x", "a")
    group : Optional[str], optional
        The group to access (creating if needed, see create_group, below).  Slashes
        denote subgroups.
    create_group : Optional[bool], optional
        If set, create any parent groups

    Returns
    -------
    ContextManager[h5py.Group]
        The opened and ready-to-go group
    """
    # Work out whether the file is open or not and get a context that is the open file.
    file_is_open = isinstance(location, (h5py.File, h5py.Group))
    if file_is_open:
        file_context = nullcontext(location)
    else:
        file_context = h5py.File(location, mode)
    # OK, open the file if we need to and start thinking about the group
    with file_context as file_object:
        current = file_object
        if group is not None:
            for part in group.strip("/").split("/"):
                if create_group:
                    current = current.require_group(part)
                else:
                    current = current[part]
        yield current


def save_ndarray(
    value: NDArray,
    destination: H5Location,
    name: Optional[str] = None,
    preferred_method: Optional[str] = None,
    **kwargs,
):
    """Save an NDarray to an HDF5 file

    Arguments
    ---------
    value : NDarray
        Array to save
    destination : H5Location (this filename, Path, h5py.File, or h5pt.Group)
        The HDF5 file/group to save this in
    name : Optional[str]
        The name to save this quantity under.  It could end up being an attribute name,
        a dataset name or a group name
    preferred_method : Optional[str]
        How to (try to) save the quantity, options are (thus far): "attribute", "dataset" or "group",
        this is only examined when needed, and will be ignored when context dictates
    **kwargs : Various
        Optional arguments to give HDF5 instructions on compression etc.
    """
    with open_h5_group(destination) as h5location:
        if preferred_method == "attribute":
            h5location.attrs[name] = value
        elif preferred_method == "dataset":
            h5location.create_dataset(name=name, data=value)
        else:
            raise ValueError(f"Unrecognized preferred_method {preferred_method}")


def load_ndarray(
    source: H5Location,
    name: str,
    method: Optional[str] = None,
):
    """Read numpy array from HDF 5 file (dataset or attribute)

    Arguments
    ---------
    source : H5Location (thus filename, Path, h5py.File, or h5pt.Group)
        The HDF5 file/group to save this in
    name : str
        The name of the dataset or attribute to read it from
    method : Optional[str]
        How is it saved "dataset" (default) or "attribute"
    """
    with open_h5_group(source) as h5_location:
        if method == "dataset":
            return np.array(h5_location[name])
        elif method == "attribute":
            return np.array(h5_location.attrs[name])
        else:
            raise ValueError(f"Unrecognized method: {method}")


def save_pint_quantity(
    value: NDArray,
    destination: h5py.Group | h5py.File,
    name: Optional[str] = None,
):
    """Save a Pint quantity to an HDF5 file

    Arguments
    ---------
    value : NDarray
        Array to save
    destination : H5Location (this filename, Path, h5py.File, or h5pt.Group)
        The HDF5 file/group to save this in
    name : Optional[str]
        The name to save this quantity under.  It could end up being an attribute name,
        a dataset name or a group name
    **kwargs : Various
        Optional arguments to give HDF5 instructions on compression etc.
    """
    with open_h5_group(destination) as h5_location:
        dataset = h5_location.create_dataset(name=name, data=value)
        dataset.attrs["units"] = str(value.units)


def load_pint_quantity(
    source: H5Location,
    name: str,
):
    """Read pint quantity from HDF 5 file (dataset or attribute)

    Arguments
    ---------
    source : H5Location (thus filename, Path, h5py.File, or h5pt.Group)
        The HDF5 file/group to save this in
    name : str
        The name of the dataset or attribute to read it from
    """
    app_ureg = pint.get_application_registry()
    with open_h5_group(source) as h5_location:
        magnitude = load_ndarray(source=source, name=name, method="dataset")
        units_str = h5_location["name"].attrs["units"]
        units = app_ureg[units_str]
        return magnitude * units
