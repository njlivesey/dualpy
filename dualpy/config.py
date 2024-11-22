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
}


@dataclass
class DualpyConfig:
    """A class for holding the dualpy configuration"""

    default_zero_array_type: type
    sparse_jacobian_fft_strategy: str | dict[str]


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
    for key in fields(config):
        # This raises an error of an inappropriate key is supplied
        set_config(key=getattr(config, key))
    # Check we got all our keys
    for key in fields(DualpyConfig):
        if not hasattr(config, key):
            raise AttributeError(f"Supplied config is missing key {key}")


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
        yield
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
    for key, item in _CONFIG_DEFAULTS.items():
        if key not in fields(DualpyConfig):
            raise KeyError(f"Unexpected DualpyConfig key: {key}")
        if not just_check:
            setattr(_config, key, item)
    # Check that every key in DualpyConfig got filled
    for key in fields(DualpyConfig):
        if key not in _CONFIG_DEFAULTS:
            raise KeyError(f"Unfilled key in DualpyConfig: {key}")


# Setup the configuration (it was already initialized above, just do the consistency
# chekcs that DualpyConfig and _CONFIG_DEFAULTS match)
reset_config(just_check=True)
