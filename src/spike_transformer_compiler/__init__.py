"""Spike-Transformer-Compiler: Neuromorphic compilation for SpikeFormers.

A TVM-style compiler that converts PyTorch SpikeFormer models into optimized 
binaries for Intel Loihi 3 neuromorphic hardware.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.ai"

from .compiler import SpikeCompiler
from .optimization import OptimizationPass
from .backend import ResourceAllocator

__all__ = [
    "SpikeCompiler",
    "OptimizationPass", 
    "ResourceAllocator",
    "__version__",
]