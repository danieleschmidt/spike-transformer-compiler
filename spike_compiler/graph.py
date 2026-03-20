"""SpikingGraph — parses SpikeFormer-style model specs into a graph representation."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Node:
    """A single layer node in a spiking neural network graph."""

    id: str
    layer_type: str  # 'spiking_attn' | 'spiking_ffn' | 'input' | 'output'
    n_neurons: int
    params: Dict = field(default_factory=dict)


@dataclass
class Edge:
    """A directed connection between two nodes, optionally carrying a weight matrix."""

    src_id: str
    dst_id: str
    weight_matrix: Optional[np.ndarray] = None


class SpikingGraph:
    """Directed acyclic graph representing a spiking neural network."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """Add a directed edge to the graph."""
        self.edges.append(edge)

    @classmethod
    def from_dict(cls, spec: dict) -> "SpikingGraph":
        """
        Parse a SpikeFormer-style spec dict into a SpikingGraph.

        Expected spec format::

            {
                "layers": [
                    {"id": "input", "layer_type": "input", "n_neurons": 16},
                    {"id": "attn0", "layer_type": "spiking_attn", "n_neurons": 32, "params": {...}},
                    ...
                ],
                "connections": [
                    ["input", "attn0"],
                    ["attn0", "ffn0", <optional_weight_list>],
                    ...
                ]
            }
        """
        graph = cls()

        for layer_spec in spec.get("layers", []):
            node = Node(
                id=layer_spec["id"],
                layer_type=layer_spec.get("layer_type", "spiking_ffn"),
                n_neurons=layer_spec.get("n_neurons", 64),
                params=layer_spec.get("params", {}),
            )
            graph.add_node(node)

        for conn in spec.get("connections", []):
            src_id = conn[0]
            dst_id = conn[1]
            weight_matrix = None
            if len(conn) > 2 and conn[2] is not None:
                w = conn[2]
                weight_matrix = np.array(w) if not isinstance(w, np.ndarray) else w
            edge = Edge(src_id=src_id, dst_id=dst_id, weight_matrix=weight_matrix)
            graph.add_edge(edge)

        return graph

    @classmethod
    def from_json(cls, json_str: str) -> "SpikingGraph":
        """Parse a JSON string spec."""
        return cls.from_dict(json.loads(json_str))

    def topological_sort(self) -> List[str]:
        """Return node ids in topological order (Kahn's algorithm)."""
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        adjacency: Dict[str, List[str]] = defaultdict(list)

        for edge in self.edges:
            adjacency[edge.src_id].append(edge.dst_id)
            if edge.dst_id in in_degree:
                in_degree[edge.dst_id] += 1
            else:
                in_degree[edge.dst_id] = 1

        queue: deque = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: List[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for neighbor in adjacency[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle — topological sort not possible.")

        return order
