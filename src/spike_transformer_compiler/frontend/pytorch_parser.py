"""PyTorch model parser for converting to Spike IR."""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from ..ir.builder import SpikeIRBuilder
from ..ir.spike_graph import SpikeGraph
from ..ir.types import SpikeType


class PyTorchParser:
    """Parser for converting PyTorch models to Spike IR."""
    
    def __init__(self):
        self.supported_layers = {
            nn.Linear: self._parse_linear,
            nn.Conv2d: self._parse_conv2d,
            nn.ReLU: self._parse_activation,
            nn.Sigmoid: self._parse_activation,
            nn.Tanh: self._parse_activation,
            nn.Dropout: self._parse_dropout,
            nn.BatchNorm2d: self._parse_batchnorm,
            nn.AdaptiveAvgPool2d: self._parse_pooling,
            nn.MaxPool2d: self._parse_pooling,
        }
        
    def parse_model(self, model: nn.Module, input_shape: Tuple[int, ...], time_steps: int = 4) -> SpikeGraph:
        """Parse PyTorch model into Spike IR graph."""
        builder = SpikeIRBuilder(f"parsed_{model.__class__.__name__}")
        builder.set_graph_metadata(
            time_steps=time_steps,
            input_shape=input_shape,
            model_class=model.__class__.__name__
        )
        
        # Add input node
        input_id = builder.add_input("model_input", input_shape, SpikeType.BINARY)
        
        # Add spike encoding
        encoded_id = builder.add_spike_encoding(input_id, "rate", time_steps)
        
        # Parse model layers
        current_id = encoded_id
        layer_counter = 0
        
        for name, layer in model.named_modules():
            if name == "":  # Skip the root module
                continue
                
            layer_type = type(layer)
            if layer_type in self.supported_layers:
                current_id = self.supported_layers[layer_type](
                    builder, layer, current_id, f"layer_{layer_counter}_{name}"
                )
                layer_counter += 1
            else:
                print(f"Warning: Unsupported layer type {layer_type.__name__} ({name})")
                
        # Add output node
        builder.add_output(current_id, "model_output")
        
        return builder.build()
        
    def _parse_linear(self, builder: SpikeIRBuilder, layer: nn.Linear, input_id: str, layer_name: str) -> str:
        """Parse Linear layer."""
        linear_id = builder.add_spike_linear(
            input_id,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            node_id=f"{layer_name}_linear"
        )
        
        # Add spiking neuron after linear layer
        neuron_id = builder.add_spike_neuron(
            linear_id,
            neuron_model="LIF",
            threshold=1.0,
            node_id=f"{layer_name}_neuron"
        )
        
        return neuron_id
        
    def _parse_conv2d(self, builder: SpikeIRBuilder, layer: nn.Conv2d, input_id: str, layer_name: str) -> str:
        """Parse Conv2d layer."""
        conv_id = builder.add_spike_conv2d(
            input_id,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size[0],
            stride=layer.stride[0],
            padding=layer.padding[0],
            bias=layer.bias is not None,
            node_id=f"{layer_name}_conv"
        )
        
        # Add spiking neuron after convolution
        neuron_id = builder.add_spike_neuron(
            conv_id,
            neuron_model="LIF",
            threshold=1.0,
            node_id=f"{layer_name}_neuron"
        )
        
        return neuron_id
        
    def _parse_activation(self, builder: SpikeIRBuilder, layer: nn.Module, input_id: str, layer_name: str) -> str:
        """Parse activation layer - convert to spiking neuron."""
        activation_type = type(layer).__name__
        
        # Map activation to neuron parameters
        if activation_type == "ReLU":
            threshold = 0.5
        elif activation_type == "Sigmoid":
            threshold = 0.7
        elif activation_type == "Tanh":
            threshold = 0.8
        else:
            threshold = 1.0
            
        neuron_id = builder.add_spike_neuron(
            input_id,
            neuron_model="LIF",
            threshold=threshold,
            node_id=f"{layer_name}_{activation_type.lower()}"
        )
        
        return neuron_id
        
    def _parse_dropout(self, builder: SpikeIRBuilder, layer: nn.Dropout, input_id: str, layer_name: str) -> str:
        """Parse Dropout layer - pass through in spiking context."""
        # In spiking networks, dropout can be implemented as probabilistic spike dropping
        # For now, we'll pass through the input unchanged
        return input_id
        
    def _parse_batchnorm(self, builder: SpikeIRBuilder, layer: nn.BatchNorm2d, input_id: str, layer_name: str) -> str:
        """Parse BatchNorm layer - convert to adaptive threshold."""
        # BatchNorm in spiking networks can be implemented as adaptive thresholding
        neuron_id = builder.add_spike_neuron(
            input_id,
            neuron_model="AdaptiveLIF",
            threshold=1.0,
            tau_mem=10.0,
            tau_syn=5.0,
            node_id=f"{layer_name}_adaptive_neuron"
        )
        
        return neuron_id
        
    def _parse_pooling(self, builder: SpikeIRBuilder, layer: nn.Module, input_id: str, layer_name: str) -> str:
        """Parse pooling layer - convert to temporal pooling."""
        if isinstance(layer, nn.AdaptiveAvgPool2d):
            method = "avg"
            window_size = 2
        elif isinstance(layer, nn.MaxPool2d):
            method = "max"
            window_size = layer.kernel_size
        else:
            method = "sum"
            window_size = 2
            
        pool_id = builder.add_temporal_pooling(
            input_id,
            window_size=window_size,
            method=method,
            node_id=f"{layer_name}_temporal_pool"
        )
        
        return pool_id
        
    def extract_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """Extract model parameters for analysis."""
        params = {}
        total_params = 0
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            params[name] = {
                "shape": list(param.shape),
                "requires_grad": param.requires_grad,
                "param_count": param_count
            }
            total_params += param_count
            
        params["total_parameters"] = total_params
        return params
        
    def estimate_model_complexity(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Estimate computational complexity of the model."""
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape[1:])
        
        # Count operations using hooks
        operations = {"conv": 0, "linear": 0, "activation": 0}
        
        def conv_hook(module, input, output):
            operations["conv"] += output.numel() * module.weight.numel()
            
        def linear_hook(module, input, output):
            operations["linear"] += output.numel() * module.weight.numel()
            
        def activation_hook(module, input, output):
            operations["activation"] += output.numel()
            
        # Register hooks
        handles = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
                handles.append(module.register_forward_hook(activation_hook))
                
        # Forward pass
        try:
            with torch.no_grad():
                model.eval()
                _ = model(dummy_input)
        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()
                
        return {
            "conv_operations": operations["conv"],
            "linear_operations": operations["linear"], 
            "activation_operations": operations["activation"],
            "total_operations": sum(operations.values())
        }