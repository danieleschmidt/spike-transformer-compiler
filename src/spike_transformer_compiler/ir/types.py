"""Data types for Spike IR."""

from typing import Tuple, Optional
from enum import Enum
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    
    # Mock numpy for basic functionality
    class np:
        @staticmethod
        def prod(shape):
            result = 1
            for dim in shape:
                result *= dim
            return result


class SpikeType(Enum):
    """Spike data types."""
    BINARY = "binary"        # 0/1 spikes
    GRADED = "graded"       # Continuous spike values  
    TEMPORAL = "temporal"    # Time-encoded spikes
    PHASE = "phase"         # Phase-encoded spikes


class SpikeTensor:
    """Tensor representation for spike trains."""
    
    def __init__(
        self, 
        shape: Tuple[int, ...],
        spike_type: SpikeType = SpikeType.BINARY,
        temporal_dim: int = -1,
        dtype: str = "float32"
    ):
        self.shape = shape
        self.spike_type = spike_type
        self.temporal_dim = temporal_dim
        self.dtype = dtype
        self.sparsity = 0.0  # Will be computed during analysis
        
    def __repr__(self) -> str:
        return f"SpikeTensor(shape={self.shape}, type={self.spike_type.value})"
        
    def get_temporal_shape(self) -> Tuple[int, ...]:
        """Get shape with temporal dimension explicit."""
        if self.temporal_dim == -1:
            return self.shape
        shape_list = list(self.shape)
        return tuple(shape_list[:self.temporal_dim] + [1] + shape_list[self.temporal_dim:])
    
    def estimate_memory(self) -> int:
        """Estimate memory usage in bytes."""
        total_elements = np.prod(self.shape)
        if self.spike_type == SpikeType.BINARY:
            return int(total_elements / 8)  # Packed bits
        elif self.dtype == "float32":
            return total_elements * 4
        elif self.dtype == "float16":
            return total_elements * 2
        else:
            return total_elements * 4  # Default


class MembraneState:
    """Neuron membrane state representation."""
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: str = "float32",
        persistent: bool = True
    ):
        self.shape = shape
        self.dtype = dtype
        self.persistent = persistent  # Whether state persists across time steps
        
    def __repr__(self) -> str:
        return f"MembraneState(shape={self.shape}, persistent={self.persistent})"
        
    def estimate_memory(self) -> int:
        """Estimate memory usage for membrane states."""
        total_elements = np.prod(self.shape)
        if self.dtype == "float32":
            return total_elements * 4
        elif self.dtype == "float16":
            return total_elements * 2
        else:
            return total_elements * 4


class SynapticWeights:
    """Synaptic weight representation with quantization and sparsity."""
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: str = "float32",
        quantized: bool = False,
        sparsity: float = 0.0,
        sparsity_pattern: Optional[str] = None
    ):
        self.shape = shape
        self.dtype = dtype
        self.quantized = quantized
        self.sparsity = sparsity
        self.sparsity_pattern = sparsity_pattern  # "structured", "unstructured", etc.
        
    def __repr__(self) -> str:
        return f"SynapticWeights(shape={self.shape}, sparsity={self.sparsity:.2f})"
        
    def estimate_memory(self) -> int:
        """Estimate memory usage considering sparsity and quantization."""
        total_elements = np.prod(self.shape)
        effective_elements = int(total_elements * (1 - self.sparsity))
        
        if self.quantized:
            bytes_per_element = 1  # 8-bit quantized
        elif self.dtype == "float32":
            bytes_per_element = 4
        elif self.dtype == "float16":
            bytes_per_element = 2
        else:
            bytes_per_element = 4
            
        # Add overhead for sparse representation
        if self.sparsity > 0.5:
            index_overhead = effective_elements * 2  # 2 bytes per index
            return effective_elements * bytes_per_element + index_overhead
        else:
            return total_elements * bytes_per_element
    
    def get_compression_ratio(self) -> float:
        """Get effective compression ratio."""
        if self.sparsity == 0:
            return 2.0 if self.quantized else 1.0
        
        dense_memory = np.prod(self.shape) * 4  # float32 baseline
        sparse_memory = self.estimate_memory()
        return dense_memory / sparse_memory