"""Backend code generation for neuromorphic hardware."""

from typing import Dict, Any


class ResourceAllocator:
    """Resource allocation strategy for multi-chip deployments."""
    
    def __init__(
        self,
        num_chips: int = 1,
        cores_per_chip: int = 128,
        synapses_per_core: int = 1024,
    ):
        self.num_chips = num_chips
        self.cores_per_chip = cores_per_chip
        self.synapses_per_core = synapses_per_core
        
    def allocate(self, model: Any) -> Dict[str, Any]:
        """Allocate hardware resources for model deployment."""
        from .ir.spike_graph import SpikeGraph
        
        if not isinstance(model, SpikeGraph):
            return {
                "allocation": "unsupported_model_type",
                "estimated_utilization": 0.0,
            }
        
        # Analyze resource requirements
        resources = model.analyze_resources()
        
        total_cores = self.num_chips * self.cores_per_chip
        total_synapses = total_cores * self.synapses_per_core
        
        # Calculate utilization
        neuron_utilization = min(1.0, resources["neuron_count"] / total_cores)
        synapse_utilization = min(1.0, resources["synapse_count"] / total_synapses)
        overall_utilization = max(neuron_utilization, synapse_utilization)
        
        # Generate allocation plan
        allocation_plan = {
            "total_chips": self.num_chips,
            "cores_per_chip": self.cores_per_chip,
            "required_neurons": resources["neuron_count"],
            "required_synapses": resources["synapse_count"],
            "neuron_utilization": neuron_utilization,
            "synapse_utilization": synapse_utilization,
            "memory_requirement": resources["total_memory_bytes"],
            "computational_load": resources["total_computation_ops"]
        }
        
        # Determine if allocation is feasible
        if overall_utilization > 1.0:
            allocation_plan["status"] = "insufficient_resources"
            allocation_plan["recommendation"] = f"Increase to {int(overall_utilization * self.num_chips) + 1} chips"
        else:
            allocation_plan["status"] = "feasible"
            allocation_plan["recommendation"] = "allocation_within_limits"
        
        return {
            "allocation": allocation_plan,
            "estimated_utilization": overall_utilization,
        }
        
    def optimize_placement(self, model: Any) -> Dict[str, Any]:
        """Optimize placement of model components across chips."""
        from .ir.spike_graph import SpikeGraph
        
        if not isinstance(model, SpikeGraph):
            return {"placement": "unsupported"}
            
        # Simple placement strategy: distribute nodes across chips
        nodes = list(model.nodes.keys())
        nodes_per_chip = len(nodes) // self.num_chips
        
        placement = {}
        for i, node_id in enumerate(nodes):
            chip_id = min(i // max(1, nodes_per_chip), self.num_chips - 1)
            placement[node_id] = f"chip_{chip_id}"
            
        return {
            "placement": placement,
            "communication_cost": self._estimate_communication_cost(model, placement),
            "load_balance": self._calculate_load_balance(placement)
        }
        
    def _estimate_communication_cost(self, model: SpikeGraph, placement: Dict[str, str]) -> float:
        """Estimate inter-chip communication cost."""
        comm_cost = 0.0
        
        for edge in model.edges:
            source_chip = placement.get(edge.source, "chip_0")
            target_chip = placement.get(edge.target, "chip_0") 
            
            if source_chip != target_chip:
                # Add cost for inter-chip communication
                bandwidth = edge.get_bandwidth_requirement()
                comm_cost += bandwidth * 1.5  # 1.5x penalty for inter-chip
                
        return comm_cost
        
    def _calculate_load_balance(self, placement: Dict[str, str]) -> float:
        """Calculate load balance across chips."""
        chip_loads = {}
        
        for node_id, chip_id in placement.items():
            if chip_id not in chip_loads:
                chip_loads[chip_id] = 0
            chip_loads[chip_id] += 1
            
        if not chip_loads:
            return 1.0
            
        loads = list(chip_loads.values())
        avg_load = sum(loads) / len(loads)
        
        if avg_load == 0:
            return 1.0
            
        # Calculate coefficient of variation
        variance = sum((load - avg_load) ** 2 for load in loads) / len(loads)
        std_dev = variance ** 0.5
        cv = std_dev / avg_load
        
        # Return balance score (1.0 = perfect balance, 0.0 = worst)
        return max(0.0, 1.0 - cv)