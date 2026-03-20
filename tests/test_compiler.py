"""Comprehensive test suite for spike_compiler."""

from __future__ import annotations

import sys
import os

# Ensure the repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from spike_compiler.graph import Edge, Node, SpikingGraph
from spike_compiler.mapper import Core, NeuronMapper
from spike_compiler.encoder import SynapseEncoder
from spike_compiler.hdl_gen import HDLGenerator
from spike_compiler.compiler import compile as snn_compile
from spike_compiler.demo import compile_3layer_snn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_spec():
    """Return a minimal valid spec for testing."""
    return {
        "layers": [
            {"id": "in",  "layer_type": "input",       "n_neurons": 4},
            {"id": "mid", "layer_type": "spiking_attn","n_neurons": 8},
            {"id": "out", "layer_type": "output",      "n_neurons": 2},
        ],
        "connections": [
            ["in",  "mid"],
            ["mid", "out"],
        ],
    }


# ---------------------------------------------------------------------------
# Graph tests
# ---------------------------------------------------------------------------

def test_graph_from_dict_nodes():
    """SpikingGraph.from_dict should parse all layers into nodes."""
    graph = SpikingGraph.from_dict(_simple_spec())
    assert len(graph.nodes) == 3
    assert "in" in graph.nodes
    assert "mid" in graph.nodes
    assert "out" in graph.nodes
    assert graph.nodes["mid"].n_neurons == 8
    assert graph.nodes["mid"].layer_type == "spiking_attn"


def test_graph_from_dict_edges():
    """SpikingGraph.from_dict should parse all connections into edges."""
    graph = SpikingGraph.from_dict(_simple_spec())
    assert len(graph.edges) == 2
    src_ids = [e.src_id for e in graph.edges]
    dst_ids = [e.dst_id for e in graph.edges]
    assert "in" in src_ids
    assert "out" in dst_ids


def test_graph_from_dict_with_weights():
    """Edges with weight lists should be stored as numpy arrays."""
    W = [[0.1, 0.2], [0.3, 0.4]]
    spec = {
        "layers": [
            {"id": "a", "layer_type": "input",  "n_neurons": 2},
            {"id": "b", "layer_type": "output", "n_neurons": 2},
        ],
        "connections": [["a", "b", W]],
    }
    graph = SpikingGraph.from_dict(spec)
    assert graph.edges[0].weight_matrix is not None
    assert isinstance(graph.edges[0].weight_matrix, np.ndarray)
    assert graph.edges[0].weight_matrix.shape == (2, 2)


def test_graph_topological_sort():
    """topological_sort should return nodes in valid topological order."""
    graph = SpikingGraph.from_dict(_simple_spec())
    order = graph.topological_sort()
    assert order.index("in") < order.index("mid")
    assert order.index("mid") < order.index("out")


def test_graph_topological_sort_all_nodes():
    """topological_sort should include every node."""
    graph = SpikingGraph.from_dict(_simple_spec())
    order = graph.topological_sort()
    assert set(order) == set(graph.nodes.keys())


# ---------------------------------------------------------------------------
# NeuronMapper tests
# ---------------------------------------------------------------------------

def test_neuron_mapper_assigns_cores():
    """Every node should receive a core assignment."""
    graph = SpikingGraph.from_dict(_simple_spec())
    mapper = NeuronMapper()
    mapping = mapper.map_graph(graph)
    assert set(mapping.keys()) == set(graph.nodes.keys())
    for core_id in mapping.values():
        assert isinstance(core_id, int)
        assert core_id >= 0


def test_neuron_mapper_respects_capacity():
    """Mapper should not exceed max_neurons_per_core per core."""
    spec = {
        "layers": [
            {"id": f"n{i}", "layer_type": "spiking_ffn", "n_neurons": 100}
            for i in range(20)
        ],
        "connections": [],
    }
    graph = SpikingGraph.from_dict(spec)
    mapper = NeuronMapper(max_neurons_per_core=1024)
    mapping = mapper.map_graph(graph)

    # Verify capacity per core
    core_neuron_counts: dict = {}
    for node_id, core_id in mapping.items():
        core_neuron_counts.setdefault(core_id, 0)
        core_neuron_counts[core_id] += graph.nodes[node_id].n_neurons

    for core_id, total in core_neuron_counts.items():
        assert total <= 1024, f"Core {core_id} exceeded capacity: {total}"


def test_neuron_mapper_get_mapping():
    """get_mapping should return the same mapping after map_graph."""
    graph = SpikingGraph.from_dict(_simple_spec())
    mapper = NeuronMapper()
    mapping = mapper.map_graph(graph)
    assert mapper.get_mapping() == mapping


# ---------------------------------------------------------------------------
# SynapseEncoder tests
# ---------------------------------------------------------------------------

def test_synapse_encoder_sparse_format():
    """encode_weights should produce CSR-like sparse format."""
    W = np.array([[1.0, 0.0], [0.0, 2.0]])
    encoder = SynapseEncoder()
    result = encoder.encode_weights(W)
    assert "non_zero_count" in result
    assert "compressed_rows" in result
    assert "compressed_cols" in result
    assert "compressed_vals" in result
    assert result["non_zero_count"] == 2
    assert len(result["compressed_vals"]) == 2


def test_synapse_encoder_zero_matrix():
    """A fully-zero weight matrix should produce zero non-zero count."""
    W = np.zeros((4, 4))
    encoder = SynapseEncoder()
    result = encoder.encode_weights(W)
    assert result["non_zero_count"] == 0
    assert result["compressed_vals"] == []


def test_synapse_encoder_spike_event_keys():
    """spike_event should return a dict with required keys."""
    encoder = SynapseEncoder()
    event = encoder.spike_event(pre_neuron=3, post_neuron=7, weight=0.5, timestamp=100)
    assert event["pre_neuron"] == 3
    assert event["post_neuron"] == 7
    assert event["weight"] == pytest.approx(0.5)
    assert event["timestamp"] == 100


def test_synapse_encoder_encode_layer():
    """encode_layer should produce one spike event per non-zero weight."""
    W = np.array([[0.5, 0.0], [0.0, 0.3]])
    edge = Edge(src_id="a", dst_id="b", weight_matrix=W)
    encoder = SynapseEncoder()
    events = encoder.encode_layer(edge, timestamp_base=0)
    assert len(events) == 2
    for ev in events:
        assert "pre_neuron" in ev
        assert "post_neuron" in ev
        assert "weight" in ev
        assert "timestamp" in ev


def test_synapse_encoder_encode_layer_none_weights():
    """encode_layer with no weight matrix should return empty list."""
    edge = Edge(src_id="a", dst_id="b", weight_matrix=None)
    encoder = SynapseEncoder()
    events = encoder.encode_layer(edge, timestamp_base=0)
    assert events == []


# ---------------------------------------------------------------------------
# HDLGenerator tests
# ---------------------------------------------------------------------------

def test_hdl_generator_module_string():
    """generate_module should produce a valid module string."""
    node = Node(id="attn0", layer_type="spiking_attn", n_neurons=64)
    gen = HDLGenerator()
    output = gen.generate_module(node, core_id=1)
    assert "spiking_layer_attn0" in output
    assert "N_NEURONS=64" in output
    assert "endmodule" in output
    assert "input clk" in output
    assert "output spike_out" in output


def test_hdl_generator_synapse_string():
    """generate_synapse should produce wire and assign statements."""
    edge = Edge(src_id="in", dst_id="out")
    gen = HDLGenerator()
    output = gen.generate_synapse(edge, src_core=0, dst_core=1)
    assert "wire" in output
    assert "assign" in output
    assert "spike_wire_in_to_out" in output


def test_hdl_generator_network_contains_modules():
    """generate_network should contain a module for every node."""
    graph = SpikingGraph.from_dict(_simple_spec())
    mapper = NeuronMapper()
    mapping = mapper.map_graph(graph)
    gen = HDLGenerator()
    hdl = gen.generate_network(graph, mapping)
    for node_id in graph.nodes:
        assert f"spiking_layer_{node_id}" in hdl


# ---------------------------------------------------------------------------
# End-to-end compile tests
# ---------------------------------------------------------------------------

def test_compile_returns_keys():
    """compile() should return a dict with all required keys."""
    result = snn_compile(_simple_spec())
    assert "graph" in result
    assert "mapping" in result
    assert "hdl_code" in result
    assert "synapse_events" in result


def test_compile_graph_type():
    """The 'graph' key should be a SpikingGraph."""
    result = snn_compile(_simple_spec())
    assert isinstance(result["graph"], SpikingGraph)


def test_compile_mapping_covers_all_nodes():
    """The mapping should cover all nodes in the graph."""
    result = snn_compile(_simple_spec())
    assert set(result["mapping"].keys()) == set(result["graph"].nodes.keys())


def test_compile_hdl_is_string():
    """hdl_code should be a non-empty string."""
    result = snn_compile(_simple_spec())
    assert isinstance(result["hdl_code"], str)
    assert len(result["hdl_code"]) > 0


# ---------------------------------------------------------------------------
# Demo test
# ---------------------------------------------------------------------------

def test_demo_3layer_runs():
    """compile_3layer_snn should run end-to-end without errors."""
    artifact = compile_3layer_snn()
    assert artifact is not None
    assert "graph" in artifact
    assert "mapping" in artifact
    assert "hdl_code" in artifact
    assert "synapse_events" in artifact

    graph = artifact["graph"]
    assert len(graph.nodes) == 4
    assert "input" in graph.nodes
    assert "attn0" in graph.nodes
    assert "ffn0" in graph.nodes
    assert "output" in graph.nodes

    # Should have synapse events (sparse weights, non-zero expected)
    assert isinstance(artifact["synapse_events"], list)

    # HDL should mention all layers
    hdl = artifact["hdl_code"]
    for node_id in ["input", "attn0", "ffn0", "output"]:
        assert node_id in hdl
