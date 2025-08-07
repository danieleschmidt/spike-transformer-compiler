"""Spike-based convolution kernels for neuromorphic processing."""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from ..ir.types import SpikeTensor, SpikeType


class SpikeConv2d:
    """Spike-based 2D convolution for neuromorphic hardware."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        threshold: float = 1.0,
        tau_mem: float = 10.0,
        tau_syn: float = 5.0
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.threshold = threshold
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        
        # Calculate kernel parameters
        self.weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.num_params = np.prod(self.weight_shape)
        if bias:
            self.num_params += out_channels
            
    def get_resource_requirements(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Calculate hardware resource requirements."""
        batch_size, in_channels, in_height, in_width = input_shape
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Resource calculations
        total_neurons = batch_size * self.out_channels * out_height * out_width
        total_synapses = total_neurons * self.kernel_size * self.kernel_size * self.in_channels
        
        # Memory requirements
        weight_memory = self.num_params * 4  # 4 bytes per float32
        activation_memory = total_neurons * 4
        intermediate_memory = in_height * in_width * self.in_channels * 4
        
        return {
            "neurons": total_neurons,
            "synapses": total_synapses,
            "memory_bytes": weight_memory + activation_memory + intermediate_memory,
            "output_shape": (batch_size, self.out_channels, out_height, out_width),
            "operations_per_timestep": total_synapses,
            "kernel_ops": out_height * out_width * self.kernel_size * self.kernel_size * self.in_channels * self.out_channels
        }
        
    def compile_for_loihi3(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Compile convolution kernel for Loihi3."""
        resources = self.get_resource_requirements(input_shape)
        
        config = {
            "kernel_type": "spike_conv2d",
            "parameters": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "bias": self.bias,
                "threshold": self.threshold,
                "tau_mem": self.tau_mem,
                "tau_syn": self.tau_syn
            },
            "resources": resources,
            "loihi3_config": {
                "neuron_model": "LIF",
                "compartments": 1,
                "dendrite_accumulation": True,
                "spike_input": True,
                "spike_output": True
            },
            "core_mapping": self._generate_conv_core_mapping(resources),
            "weight_mapping": self._generate_weight_mapping(resources)
        }
        
        return config
        
    def _generate_conv_core_mapping(self, resources: Dict[str, Any]) -> Dict[str, Any]:
        """Generate core mapping optimized for convolution."""
        neurons_per_core = 1024
        cores_needed = (resources["neurons"] + neurons_per_core - 1) // neurons_per_core
        
        # Optimize for spatial locality
        output_shape = resources["output_shape"]
        batch_size, out_channels, out_height, out_width = output_shape
        
        # Try to keep spatial neighborhoods on the same core
        spatial_neurons_per_core = min(neurons_per_core, out_height * out_width)
        channels_per_core = neurons_per_core // spatial_neurons_per_core
        
        return {
            "cores_needed": cores_needed,
            "neurons_per_core": neurons_per_core,
            "spatial_neurons_per_core": spatial_neurons_per_core,
            "channels_per_core": channels_per_core,
            "spatial_locality_optimization": True,
            "channel_parallelism": min(out_channels, cores_needed)
        }
        
    def _generate_weight_mapping(self, resources: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized weight storage mapping."""
        return {
            "weight_layout": "channel_major",
            "weight_compression": "sparse_csr",
            "weight_quantization": "int8",
            "shared_weights": True,
            "weight_reuse_pattern": "spatial_sharing"
        }
        
    def simulate_conv(
        self,
        input_spikes: np.ndarray,
        weights: np.ndarray,
        bias: Optional[np.ndarray] = None,
        time_steps: int = 4
    ) -> np.ndarray:
        """Simulate spike-based convolution."""
        batch_size, in_channels, in_height, in_width = input_spikes.shape
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize membrane potentials and output spikes
        membrane_potential = np.zeros((batch_size, self.out_channels, out_height, out_width))
        output_spikes = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Pad input if necessary
        if self.padding > 0:
            input_spikes = np.pad(
                input_spikes,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant', constant_values=0
            )
        
        # Simulate over time steps
        for t in range(time_steps):
            # Convolution operation
            for out_c in range(self.out_channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        # Extract receptive field
                        h_start = out_h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = out_w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        receptive_field = input_spikes[:, :, h_start:h_end, w_start:w_end]
                        
                        # Compute weighted sum
                        conv_output = np.sum(
                            receptive_field * weights[out_c, :, :, :],
                            axis=(1, 2, 3)
                        )
                        
                        # Add bias if present
                        if bias is not None:
                            conv_output += bias[out_c]
                            
                        # Update membrane potential with LIF dynamics
                        membrane_potential[:, out_c, out_h, out_w] = (
                            membrane_potential[:, out_c, out_h, out_w] * 
                            np.exp(-1.0 / self.tau_mem) + conv_output
                        )
                        
                        # Generate spikes
                        spike_mask = membrane_potential[:, out_c, out_h, out_w] > self.threshold
                        output_spikes[:, out_c, out_h, out_w] = spike_mask.astype(np.float32)
                        
                        # Reset membrane potential after spiking
                        membrane_potential[:, out_c, out_h, out_w] *= (1 - spike_mask)
                        
        return output_spikes
        
    def estimate_energy(self, input_shape: Tuple[int, ...], activity_rate: float = 0.05) -> float:
        """Estimate energy consumption for spike convolution."""
        resources = self.get_resource_requirements(input_shape)
        
        # Loihi3 energy model
        energy_per_spike = 23.6e-12  # pJ
        energy_per_synapse = 50e-15  # fJ
        energy_per_neuron_static = 1e-15  # fJ
        
        # Activity-dependent calculations
        active_neurons = resources["neurons"] * activity_rate
        active_synapses = resources["synapses"] * activity_rate
        
        spike_energy = active_neurons * energy_per_spike
        synapse_energy = active_synapses * energy_per_synapse
        static_energy = resources["neurons"] * energy_per_neuron_static
        
        total_energy = spike_energy + synapse_energy + static_energy
        return total_energy * 1e9  # Convert to nJ


class SpikeConv1d:
    """Spike-based 1D convolution for sequence processing."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        threshold: float = 1.0
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.threshold = threshold
        
        # Delegate to 2D convolution with height=1
        self.conv2d = SpikeConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            threshold=threshold
        )
        
    def get_resource_requirements(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Calculate resource requirements for 1D convolution."""
        batch_size, in_channels, sequence_length = input_shape
        
        # Convert to 2D shape for calculation
        input_2d_shape = (batch_size, in_channels, 1, sequence_length)
        resources = self.conv2d.get_resource_requirements(input_2d_shape)
        
        # Adjust output shape back to 1D
        out_length = (sequence_length + 2 * self.padding - self.kernel_size) // self.stride + 1
        resources["output_shape"] = (batch_size, self.out_channels, out_length)
        
        return resources
        
    def simulate_conv1d(
        self,
        input_spikes: np.ndarray,
        weights: np.ndarray,
        bias: Optional[np.ndarray] = None,
        time_steps: int = 4
    ) -> np.ndarray:
        """Simulate 1D spike convolution."""
        batch_size, in_channels, sequence_length = input_spikes.shape
        
        # Reshape to 2D for processing
        input_2d = input_spikes.reshape(batch_size, in_channels, 1, sequence_length)
        weights_2d = weights.reshape(self.out_channels, in_channels, 1, self.kernel_size)
        
        # Use 2D convolution
        output_2d = self.conv2d.simulate_conv(input_2d, weights_2d, bias, time_steps)
        
        # Reshape back to 1D
        output_1d = output_2d.squeeze(axis=2)
        
        return output_1d