"""Spike Intermediate Representation (IR) module."""

from .spike_graph import SpikeGraph, SpikeNode, SpikeEdge
from .builder import SpikeIRBuilder
from .passes import IRPass, DeadCodeElimination, SpikeFusion
from .types import SpikeTensor, MembraneState, SynapticWeights

__all__ = [
    "SpikeGraph",
    "SpikeNode", 
    "SpikeEdge",
    "SpikeIRBuilder",
    "IRPass",
    "DeadCodeElimination",
    "SpikeFusion", 
    "SpikeTensor",
    "MembraneState",
    "SynapticWeights",
]