"""HDLGenerator — generates Verilog-like pseudocode for spiking neural networks."""

from __future__ import annotations

from typing import Dict

from .graph import Edge, Node, SpikingGraph


class HDLGenerator:
    """Generates human-readable HDL pseudocode for a mapped SNN."""

    def generate_module(self, node: Node, core_id: int) -> str:
        """
        Generate a Verilog-like module declaration for a single node.

        Example output::

            module spiking_layer_input #(N_NEURONS=16) (
              input clk, rst, spike_in,
              output spike_out
            );
            // layer_type=input | core=0 | n_neurons=16
            // ... neuron logic
            endmodule
        """
        return (
            f"module spiking_layer_{node.id} #(N_NEURONS={node.n_neurons}) (\n"
            f"  input clk, rst, spike_in,\n"
            f"  output spike_out\n"
            f");\n"
            f"// layer_type={node.layer_type} | core={core_id} | n_neurons={node.n_neurons}\n"
            f"// ... neuron logic\n"
            f"endmodule"
        )

    def generate_synapse(self, edge: Edge, src_core: int, dst_core: int) -> str:
        """
        Generate a wire/connection declaration between two modules.

        Example output::

            wire spike_wire_input_to_attn0;  // core0 -> core1
            assign spiking_layer_attn0.spike_in = spiking_layer_input.spike_out;
        """
        wire_name = f"spike_wire_{edge.src_id}_to_{edge.dst_id}"
        return (
            f"wire {wire_name};  // core{src_core} -> core{dst_core}\n"
            f"assign spiking_layer_{edge.dst_id}.spike_in = "
            f"spiking_layer_{edge.src_id}.spike_out;"
        )

    def generate_network(self, graph: SpikingGraph, mapping: Dict[str, int]) -> str:
        """
        Generate full HDL pseudocode for the entire network.

        Includes all module declarations and synapse wires.
        """
        lines = ["// ============================================================",
                 "// spike-transformer-compiler — generated HDL pseudocode",
                 "// ============================================================",
                 ""]

        # Module declarations in topological order where possible
        try:
            order = graph.topological_sort()
        except ValueError:
            order = list(graph.nodes.keys())

        for node_id in order:
            node = graph.nodes[node_id]
            core_id = mapping.get(node_id, 0)
            lines.append(self.generate_module(node, core_id))
            lines.append("")

        # Synapse wires
        if graph.edges:
            lines.append("// --- Synapse connections ---")
            for edge in graph.edges:
                src_core = mapping.get(edge.src_id, 0)
                dst_core = mapping.get(edge.dst_id, 0)
                lines.append(self.generate_synapse(edge, src_core, dst_core))
            lines.append("")

        return "\n".join(lines)
