"""Kernel builder for custom neuromorphic operations."""

from typing import Dict, Any, Callable, Optional, List
import inspect
from ..ir.spike_graph import SpikeNode, NodeType


class KernelBuilder:
    """Builder for creating custom neuromorphic kernels."""
    
    _registered_kernels: Dict[str, Callable] = {}
    
    def __init__(self, kernel_name: str):
        self.kernel_name = kernel_name
        self.neuron_groups = []
        self.synapse_groups = []
        self.connections = []
        self.parameters = {}
        
    @classmethod
    def register(cls, kernel_name: str):
        """Decorator to register custom kernels."""
        def decorator(func: Callable) -> Callable:
            cls._registered_kernels[kernel_name] = func
            return func
        return decorator
        
    @classmethod
    def get_registered_kernels(cls) -> List[str]:
        """Get list of registered kernel names."""
        return list(cls._registered_kernels.keys())
        
    @classmethod
    def create_kernel(cls, kernel_name: str, params: Dict[str, Any]) -> 'CompiledKernel':
        """Create kernel instance from registered kernels."""
        if kernel_name not in cls._registered_kernels:
            raise ValueError(f"Kernel '{kernel_name}' not registered")
            
        kernel_func = cls._registered_kernels[kernel_name]
        builder = cls(kernel_name)
        
        # Call the kernel function with builder and parameters
        result = kernel_func(builder, params)
        return result if isinstance(result, CompiledKernel) else builder.build()
        
    def add_neuron_group(
        self,
        name: str,
        size: int,
        neuron_model: str = "LIF",
        threshold: float = 1.0,
        tau_mem: float = 10.0,
        tau_syn: float = 5.0,
        reset_mode: str = "zero",
        **kwargs
    ) -> str:
        """Add a group of neurons to the kernel."""
        neuron_group = {
            "name": name,
            "size": size,
            "neuron_model": neuron_model,
            "threshold": threshold,
            "tau_mem": tau_mem,
            "tau_syn": tau_syn,
            "reset_mode": reset_mode,
            "extra_params": kwargs
        }
        
        self.neuron_groups.append(neuron_group)
        return name
        
    def add_synapse_group(
        self,
        pre: str,
        post: str,
        connectivity: str = "dense",
        weight_init: str = "uniform",
        weight_scale: float = 1.0,
        delay: int = 1,
        **kwargs
    ) -> str:
        """Add synaptic connections between neuron groups."""
        synapse_group = {
            "pre": pre,
            "post": post,
            "connectivity": connectivity,
            "weight_init": weight_init,
            "weight_scale": weight_scale,
            "delay": delay,
            "extra_params": kwargs
        }
        
        self.synapse_groups.append(synapse_group)
        connection_name = f"{pre}_to_{post}"
        return connection_name
        
    def add_input_connection(
        self,
        input_name: str,
        target_group: str,
        connection_type: str = "one_to_one"
    ) -> None:
        """Add connection from external input to neuron group."""
        connection = {
            "type": "input",
            "source": input_name,
            "target": target_group,
            "connection_type": connection_type
        }
        self.connections.append(connection)
        
    def add_output_connection(
        self,
        source_group: str,
        output_name: str,
        connection_type: str = "one_to_one"
    ) -> None:
        """Add connection from neuron group to external output."""
        connection = {
            "type": "output",
            "source": source_group,
            "target": output_name,
            "connection_type": connection_type
        }
        self.connections.append(connection)
        
    def set_parameter(self, name: str, value: Any) -> None:
        """Set kernel parameter."""
        self.parameters[name] = value
        
    def build(self) -> 'CompiledKernel':
        """Build the kernel into a compiled form."""
        kernel_spec = {
            "name": self.kernel_name,
            "neuron_groups": self.neuron_groups,
            "synapse_groups": self.synapse_groups,
            "connections": self.connections,
            "parameters": self.parameters
        }
        
        return CompiledKernel(kernel_spec)
        
    def validate(self) -> bool:
        """Validate kernel configuration."""
        # Check that all referenced groups exist
        group_names = {group["name"] for group in self.neuron_groups}
        
        for synapse in self.synapse_groups:
            if synapse["pre"] != "input" and synapse["pre"] not in group_names:
                raise ValueError(f"Pre-synaptic group '{synapse['pre']}' not found")
            if synapse["post"] != "output" and synapse["post"] not in group_names:
                raise ValueError(f"Post-synaptic group '{synapse['post']}' not found")
                
        for connection in self.connections:
            if connection["type"] == "input":
                if connection["target"] not in group_names:
                    raise ValueError(f"Target group '{connection['target']}' not found")
            elif connection["type"] == "output":
                if connection["source"] not in group_names:
                    raise ValueError(f"Source group '{connection['source']}' not found")
                    
        return True


class CompiledKernel:
    """Represents a compiled neuromorphic kernel."""
    
    def __init__(self, kernel_spec: Dict[str, Any]):
        self.kernel_spec = kernel_spec
        self.name = kernel_spec["name"]
        self.resources = self._calculate_resources()
        
    def _calculate_resources(self) -> Dict[str, Any]:
        """Calculate resource requirements for the kernel."""
        total_neurons = sum(group["size"] for group in self.kernel_spec["neuron_groups"])
        
        total_synapses = 0
        for synapse in self.kernel_spec["synapse_groups"]:
            # Estimate synapse count based on connectivity
            pre_group = self._find_group(synapse["pre"])
            post_group = self._find_group(synapse["post"])
            
            if pre_group and post_group:
                if synapse["connectivity"] == "dense":
                    total_synapses += pre_group["size"] * post_group["size"]
                elif synapse["connectivity"] == "sparse":
                    sparsity = synapse.get("extra_params", {}).get("sparsity", 0.1)
                    total_synapses += int(pre_group["size"] * post_group["size"] * sparsity)
                elif synapse["connectivity"] == "conv2d":
                    kernel_size = synapse.get("extra_params", {}).get("kernel_size", 3)
                    total_synapses += post_group["size"] * kernel_size * kernel_size
                else:
                    total_synapses += min(pre_group["size"], post_group["size"])
                    
        return {
            "neurons": total_neurons,
            "synapses": total_synapses,
            "memory_bytes": total_neurons * 16 + total_synapses * 4,  # Estimated
            "neuron_groups": len(self.kernel_spec["neuron_groups"]),
            "synapse_groups": len(self.kernel_spec["synapse_groups"])
        }
        
    def _find_group(self, group_name: str) -> Optional[Dict[str, Any]]:
        """Find neuron group by name."""
        for group in self.kernel_spec["neuron_groups"]:
            if group["name"] == group_name:
                return group
        return None
        
    def get_loihi3_config(self) -> Dict[str, Any]:
        """Generate Loihi3 hardware configuration."""
        config = {
            "kernel_name": self.name,
            "neuron_configs": {},
            "synapse_configs": {},
            "core_mapping": {},
            "resources": self.resources
        }
        
        # Generate neuron configurations
        for group in self.kernel_spec["neuron_groups"]:
            config["neuron_configs"][group["name"]] = {
                "neuron_model": group["neuron_model"],
                "threshold": int(group["threshold"] * 1024),  # Scale for Loihi3
                "decay_u": int(1024 * (1 - 1/group["tau_mem"])),
                "decay_v": int(1024 * (1 - 1/group["tau_syn"])),
                "reset_mode": 1 if group["reset_mode"] == "zero" else 0,
                "size": group["size"]
            }
            
        # Generate synapse configurations
        for synapse in self.kernel_spec["synapse_groups"]:
            connection_name = f"{synapse['pre']}_to_{synapse['post']}"
            config["synapse_configs"][connection_name] = {
                "weight_scale": int(synapse["weight_scale"] * 256),  # Scale for Loihi3
                "delay": synapse["delay"],
                "connectivity": synapse["connectivity"],
                "sign_mode": 1  # Mixed excitatory/inhibitory
            }
            
        return config
        
    def estimate_energy(self, activity_rate: float = 0.05) -> float:
        """Estimate energy consumption for the kernel."""
        # Loihi3 energy model
        energy_per_spike = 23.6e-12  # pJ
        energy_per_synapse = 50e-15  # fJ
        
        active_neurons = self.resources["neurons"] * activity_rate
        active_synapses = self.resources["synapses"] * activity_rate
        
        total_energy = (active_neurons * energy_per_spike + 
                       active_synapses * energy_per_synapse)
        
        return total_energy * 1e9  # Convert to nJ
        
    def profile(self, num_inferences: int = 1000, measure: List[str] = None) -> Dict[str, Any]:
        """Profile kernel performance."""
        if measure is None:
            measure = ["energy", "latency", "spike_count"]
            
        profile_data = {}
        
        if "energy" in measure:
            profile_data["energy_per_inference"] = self.estimate_energy()
            profile_data["total_energy_mJ"] = profile_data["energy_per_inference"] * num_inferences / 1e6
            
        if "latency" in measure:
            # Estimate latency based on neuron count and operations
            base_latency = 1.0  # ms
            complexity_factor = np.log10(max(1, self.resources["neurons"]))
            profile_data["latency_ms"] = base_latency * complexity_factor
            
        if "spike_count" in measure:
            # Estimate spike activity
            profile_data["avg_spikes_per_inference"] = self.resources["neurons"] * 0.05  # 5% activity
            
        profile_data["resources"] = self.resources
        profile_data["num_inferences"] = num_inferences
        
        return profile_data


# Pre-built kernel definitions
@KernelBuilder.register("spike_conv2d")
def create_spike_conv2d_kernel(builder: KernelBuilder, params: Dict[str, Any]) -> CompiledKernel:
    """Create a spike-based 2D convolution kernel."""
    in_channels = params["in_channels"]
    out_channels = params["out_channels"]
    kernel_size = params.get("kernel_size", 3)
    output_height = params.get("output_height", 1)
    output_width = params.get("output_width", 1)
    
    # Add convolution neurons
    conv_neurons = builder.add_neuron_group(
        name="conv_neurons",
        size=out_channels * output_height * output_width,
        neuron_model="LIF",
        threshold=params.get("threshold", 1.0)
    )
    
    # Add input-to-conv synapses
    builder.add_synapse_group(
        pre="input",
        post="conv_neurons",
        connectivity="conv2d",
        kernel_size=kernel_size,
        stride=params.get("stride", 1),
        weight_init="kaiming"
    )
    
    # Add connections
    builder.add_input_connection("input", "conv_neurons")
    builder.add_output_connection("conv_neurons", "output")
    
    return builder.build()


@KernelBuilder.register("spike_attention") 
def create_spike_attention_kernel(builder: KernelBuilder, params: Dict[str, Any]) -> CompiledKernel:
    """Create a spike-based attention kernel."""
    embed_dim = params["embed_dim"]
    num_heads = params["num_heads"]
    sequence_length = params.get("sequence_length", 1)
    
    # Query, Key, Value projection neurons
    qkv_size = embed_dim * 3
    builder.add_neuron_group(
        name="qkv_neurons",
        size=qkv_size,
        neuron_model="LIF",
        threshold=0.8
    )
    
    # Attention computation neurons
    attention_size = sequence_length * num_heads
    builder.add_neuron_group(
        name="attention_neurons", 
        size=attention_size,
        neuron_model="LIF",
        threshold=1.0
    )
    
    # Output projection neurons
    builder.add_neuron_group(
        name="output_neurons",
        size=embed_dim,
        neuron_model="LIF", 
        threshold=0.9
    )
    
    # Add connections
    builder.add_synapse_group("input", "qkv_neurons", "dense")
    builder.add_synapse_group("qkv_neurons", "attention_neurons", "sparse", sparsity=0.1)
    builder.add_synapse_group("attention_neurons", "output_neurons", "dense")
    
    builder.add_input_connection("input", "qkv_neurons")
    builder.add_output_connection("output_neurons", "output")
    
    return builder.build()