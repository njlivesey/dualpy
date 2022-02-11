"""A module for noting dualpy configuration information"""
import copy
from dataclasses import dataclass


__all__ = ["get_config", "set_config", "reset_config"]


@dataclass
class DualpyConfig:
    """A class for holding the dualpy configuration"""

    dask: set[str]


_config_defaults = {
    "dask": {
        "CubicSplineLinearJacobians",
        "irfft",
        "rfft",
        "tensordot",
    },
}

_config = DualpyConfig(**_config_defaults)


def get_config():
    """Return (a copy of the) the dualpy configuration"""
    return copy.deepcopy(_config)


def set_config(**kwargs):
    """Set items of the dualpy configuration"""
    for key, value in kwargs.items():
        if key not in _config_defaults:
            raise NameError(f"No such dualy configuration argument {key}")
        setattr(_config, key, value)

def reset_config():
    """Reset the dualpy configuration to its defaults"""
    global _config
    _config = DualpyConfig(**_config_defaults)
    
    
