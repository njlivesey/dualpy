"""A module for noting dualpy configuration information"""

import contextlib
import copy
from dataclasses import dataclass, fields
from typing import Any, Optional

import numpy as np

_CONFIG_DEFAULTS = {
    # This item provides the default class for creating new Jacobian objects (at least
    # for dense and diagonal, sparse is it's own beast).  This is intended so that users
    # can invoke alternatives such as dask or cupy arrays (not supported as yet).
    "default_zero_array_type": np.zeros,
    # This next one defines the potential strategy for doing FFTs on sparse Jacobians.
    # Options are "dense", "matrix-multiply", and "gather", each of which may prove
    # optimal under various circumstances.  See the documentation for the fft routines.
    # Note that, if needed, it can be a dict giving the strategy on a
    # Jacobian-by-Jacobian basis (with an empty string key for the default case).
    "sparse_jacobian_fft_strategy": "gather",
    # This one defines the strategy for cumsum, again it can be on a Jacobian-by-
    # Jacobian basis.  Options are "dense", "matrix-multiply", and "gather"
    "sparse_jacobian_cumsum_strategy": "matrix-multiply",
    # This one turns on frequent checking of the jacobians
    "check_jacobians": False,
    # This one is used when reading from HDF (or possible similar), when a "units"
    # attribute is found, what library should be used, pint (default) or astropy.
    "default_units_library": "pint",
}


@dataclass
class DualpyConfig:
    """A class for holding the dualpy configuration"""

    default_zero_array_type: type
    sparse_jacobian_fft_strategy: str | dict[str]
    sparse_jacobian_cumsum_strategy: str | dict[str]
    check_jacobians: bool
    default_units_library: str


_config = DualpyConfig(**_CONFIG_DEFAULTS)


def get_config(key: str) -> Any:
    """Return an item from the dualpy config

    Parameters
    ----------
    key : str
        The entry from the config to return

    Returns
    -------
    DualpyConfig | Any
        The corresponding entry from the config
    """
    return getattr(_config, key)


def set_config(**kwargs):
    """Set items of the dualpy configuration"""
    for key, value in kwargs.items():
        if key not in _CONFIG_DEFAULTS:
            raise NameError(f"No such dualy configuration argument {key}")
        setattr(_config, key, value)


def get_full_config() -> DualpyConfig:
    """Return the current full configuration"""
    return copy.deepcopy(_config)


def set_full_config(config: DualpyConfig):
    """Set the configuration to the supplied values

    Also check that there are no extra keys and that all are supplied"""
    for field in fields(config):
        # This raises an error of an inappropriate key is supplied
        set_config(**{field.name: getattr(config, field.name)})
    # Check we got all our keys
    for field in fields(DualpyConfig):
        if not hasattr(config, field.name):
            raise AttributeError(f"Supplied config is missing field {field.name}")


def get_jacobian_specific_config(config_key: str, jacobian_key: str):
    """For a given term in the config return the value for a given Jacobian

    If the config element is a str, we just return that. If it is a dict, then we return
    the entry corresponding to jacobian_key.  If there isn't one, we return the entry
    with the key "", if there is one.
    """
    config_info = get_config(config_key)
    # It can be a string or a jacobian-dependent dict of strings (with "" as the
    # default). If it's a string, just return that.
    if isinstance(config_info, str):
        return config_info
    # Otherwise, it's a dict, try to get it, with "" as the default
    if jacobian_key in config_info:
        return config_info[jacobian_key]
    return config_info[""]


@contextlib.contextmanager
def dualpy_context(**kwargs):
    """Provides a context manager for temporarily setting dualpy configuration options

    Parameters
    ----------
    **argv : dict
        Various potential configuration options
    """
    original_config = get_full_config()
    set_config(**kwargs)
    try:
        yield original_config
    finally:
        set_full_config(original_config)


def reset_config(just_check: Optional[bool] = False):
    """Reset the dualpy configuration to its defaults

    Also does some error checking to enure the DualPyConfig and _CONFIG_DEFAULTS match

    Parameters
    ----------
    just_check : bool, optional, default False
        If set, don't copy, just do the checking described above (used on first
        initialization)
    """
    field_names = [field.name for field in fields(DualpyConfig)]
    for key, item in _CONFIG_DEFAULTS.items():
        if key not in field_names:
            raise KeyError(f"Unexpected DualpyConfig key: {key}")
        if not just_check:
            setattr(_config, key, item)
    # Check that every key in DualpyConfig got filled
    for key in field_names:
        if key not in _CONFIG_DEFAULTS:
            raise KeyError(f"Unfilled key in DualpyConfig: {key}")


# Setup the configuration (it was already initialized above, just do the consistency
# checks that DualpyConfig and _CONFIG_DEFAULTS match)
reset_config(just_check=True)
