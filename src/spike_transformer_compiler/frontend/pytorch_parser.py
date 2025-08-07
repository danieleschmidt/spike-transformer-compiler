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
            nn.Conv1d: self._parse_conv1d,
            nn.ReLU: self._parse_activation,
            nn.GELU: self._parse_activation,
            nn.Sigmoid: self._parse_activation,
            nn.Tanh: self._parse_activation,
            nn.Dropout: self._parse_dropout,
            nn.BatchNorm1d: self._parse_batchnorm,
            nn.BatchNorm2d: self._parse_batchnorm,
            nn.AdaptiveAvgPool2d: self._parse_pooling,
            nn.MaxPool2d: self._parse_pooling,
            nn.AvgPool2d: self._parse_pooling,
            nn.Flatten: self._parse_flatten,
            nn.MultiheadAttention: self._parse_multihead_attention,
            nn.LayerNorm: self._parse_layer_norm,
        }
        self.layer_counter = 0
        self.residual_stack = []
        
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
        elif activation_type == "GELU":
            threshold = 0.6  # GELU has smoother activation curve
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
        
    def _parse_conv1d(self, builder: SpikeIRBuilder, layer: nn.Conv1d, input_id: str, layer_name: str) -> str:
        """Parse Conv1d layer."""
        # Convert 1D conv to 2D equivalent for spike processing
        conv_id = builder.add_spike_conv2d(
            input_id,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size[0],
            stride=layer.stride[0],
            padding=layer.padding[0],
            bias=layer.bias is not None,
            node_id=f"{layer_name}_conv1d"
        )
        
        # Add spiking neuron after convolution
        neuron_id = builder.add_spike_neuron(
            conv_id,
            neuron_model="LIF",
            threshold=1.0,
            node_id=f"{layer_name}_neuron"
        )
        
        return neuron_id
        
    def _parse_flatten(self, builder: SpikeIRBuilder, layer: nn.Flatten, input_id: str, layer_name: str) -> str:
        """Parse Flatten layer - reshape operation for spike data."""
        # Flatten is handled at the IR level, pass through the input
        # The actual reshaping will be done during backend compilation
        return input_id
        
    def _parse_dropout(self, builder: SpikeIRBuilder, layer: nn.Dropout, input_id: str, layer_name: str) -> str:
        """Parse Dropout layer - pass through in spiking context."""
        # In spiking networks, dropout can be implemented as probabilistic spike dropping
        # For now, we'll pass through the input unchanged
        return input_id
        
    def _parse_batchnorm(self, builder: SpikeIRBuilder, layer: nn.Module, input_id: str, layer_name: str) -> str:
        """Parse BatchNorm layer - convert to adaptive threshold."""
        # BatchNorm in spiking networks can be implemented as adaptive thresholding
        # Use different time constants based on BatchNorm type
        if isinstance(layer, nn.BatchNorm1d):
            tau_mem = 8.0
        else:
            tau_mem = 10.0
            
        neuron_id = builder.add_spike_neuron(
            input_id,
            neuron_model="AdaptiveLIF",
            threshold=1.0,
            tau_mem=tau_mem,
            tau_syn=5.0,
            adaptation_strength=0.05,
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
            window_size = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
        elif isinstance(layer, nn.AvgPool2d):
            method = "avg"
            window_size = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
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
        
    def _parse_multihead_attention(self, builder: SpikeIRBuilder, layer: nn.MultiheadAttention, input_id: str, layer_name: str) -> str:
        """Parse MultiheadAttention layer - convert to spike-based attention."""
        attention_id = builder.add_spike_attention(
            input_id,
            embed_dim=layer.embed_dim,
            num_heads=layer.num_heads,
            dropout=layer.dropout,
            spike_mode="binary",
            window_size=4,
            sparse_ratio=0.1,
            node_id=f"{layer_name}_spike_attention"
        )
        
        # Add output projection neuron
        neuron_id = builder.add_spike_neuron(
            attention_id,
            neuron_model="LIF", 
            threshold=0.8,
            tau_mem=10.0,
            node_id=f"{layer_name}_attention_neuron"
        )
        
        return neuron_id
        
    def _parse_layer_norm(self, builder: SpikeIRBuilder, layer: nn.LayerNorm, input_id: str, layer_name: str) -> str:
        """Parse LayerNorm layer - convert to adaptive threshold mechanism."""
        # LayerNorm in spiking networks implemented as adaptive thresholding
        neuron_id = builder.add_spike_neuron(
            input_id,
            neuron_model="AdaptiveLIF",
            threshold=1.0,
            tau_mem=15.0,  # Longer time constant for normalization
            tau_syn=5.0,
            adaptation_strength=0.1,
            node_id=f"{layer_name}_layer_norm"
        )
        
        return neuron_id
        
    def detect_transformer_blocks(self, model: nn.Module) -> list:
        """Detect transformer blocks in the model architecture."""
        transformer_blocks = []
        
        for name, module in model.named_modules():
            # Look for common transformer block patterns
            if any(pattern in name.lower() for pattern in ['transformer', 'block', 'layer']):
                has_attention = any(isinstance(child, nn.MultiheadAttention) 
                                  for child in module.children())
                has_feedforward = any(isinstance(child, nn.Linear) 
                                    for child in module.children())
                has_norm = any(isinstance(child, nn.LayerNorm) 
                              for child in module.children())
                
                if has_attention or (has_feedforward and has_norm):
                    transformer_blocks.append({
                        'name': name,
                        'module': module,
                        'has_attention': has_attention,
                        'has_feedforward': has_feedforward,
                        'has_norm': has_norm
                    })
                    
        return transformer_blocks
        
    def parse_transformer_block(self, builder: SpikeIRBuilder, block_module: nn.Module, input_id: str, block_name: str) -> str:
        """Parse a complete transformer block with residual connections."""
        # Store input for residual connection
        residual_input = input_id
        current_id = input_id
        
        # Parse each component in the block
        for name, layer in block_module.named_children():
            layer_type = type(layer)
            if layer_type in self.supported_layers:
                current_id = self.supported_layers[layer_type](
                    builder, layer, current_id, f"{block_name}_{name}"
                )
                
        # Add residual connection
        if current_id != residual_input:
            residual_id = builder.add_residual_connection(
                residual_input,
                current_id,
                node_id=f"{block_name}_residual"
            )
            return residual_id
        else:
            return current_id
        
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