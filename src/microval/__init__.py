"""
microval: A high-performance Python package for statistical validation and randomization of weighted networks.
"""

from .validation import compute_pval, validate_network
from .samplers.west import WEST, pre_compute_west
from .samplers.biwest import BiWEST, pre_compute_biwest
from .samplers.wes import WES, pre_compute_wes
from .samplers.biwes import BiWES, pre_compute_biwes
from . import samplers

__version__ = "0.1.0"

__all__ = [
    # Top-level statistical validation functions
    "compute_pval",
    "validate_network",
    # Sampler functions
    "WEST",
    "pre_compute_west",
    "BiWEST",
    "pre_compute_biwest",
    "WES",
    "pre_compute_wes",
    "BiWES",
    "pre_compute_biwes",
    # Submodules
    "samplers",
]
