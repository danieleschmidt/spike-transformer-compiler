"""NeuronMapper — maps SpikingGraph nodes to logical Loihi-style neuromorphic cores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .graph import SpikingGraph


@dataclass
class Core:
    """A logical neuromorphic core with a fixed neuron capacity."""

    core_id: int
    capacity: int = 1024
    neurons: List[str] = field(default_factory=list)

    @property
    def used(self) -> int:
        """Return total neurons currently assigned to this core."""
        return sum(0 for _ in self.neurons)  # placeholder — will be replaced

    def neuron_count(self, graph: SpikingGraph) -> int:
        """Return total neuron count for nodes assigned to this core."""
        return sum(graph.nodes[nid].n_neurons for nid in self.neurons if nid in graph.nodes)

    def available(self, graph: SpikingGraph) -> int:
        return self.capacity - self.neuron_count(graph)


class NeuronMapper:
    """Greedy neuron mapper that packs graph nodes into fixed-capacity cores."""

    def __init__(self, max_neurons_per_core: int = 1024) -> None:
        self.max_neurons_per_core = max_neurons_per_core
        self.cores: List[Core] = []
        self._mapping: Dict[str, int] = {}

    def _get_or_create_core(self, n_neurons: int, graph: SpikingGraph) -> Core:
        """Return an existing core with enough space, or create a new one."""
        for core in self.cores:
            if core.available(graph) >= n_neurons:
                return core
        new_core = Core(core_id=len(self.cores), capacity=self.max_neurons_per_core)
        self.cores.append(new_core)
        return new_core

    def map_graph(self, graph: SpikingGraph) -> Dict[str, int]:
        """
        Assign each node in the graph to a core using greedy first-fit packing.

        Processes nodes in topological order to maintain data-flow order.
        Returns a mapping of {node_id: core_id}.
        """
        self.cores = []
        self._mapping = {}

        try:
            order = graph.topological_sort()
        except ValueError:
            order = list(graph.nodes.keys())

        for node_id in order:
            node = graph.nodes[node_id]
            n = node.n_neurons

            if n > self.max_neurons_per_core:
                raise ValueError(
                    f"Node '{node_id}' requires {n} neurons, which exceeds "
                    f"max_neurons_per_core={self.max_neurons_per_core}."
                )

            core = self._get_or_create_core(n, graph)
            core.neurons.append(node_id)
            self._mapping[node_id] = core.core_id

        return self._mapping

    def get_mapping(self) -> Dict[str, int]:
        """Return the current {node_id: core_id} mapping."""
        return dict(self._mapping)
