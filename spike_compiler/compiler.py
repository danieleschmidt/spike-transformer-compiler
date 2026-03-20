"""End-to-end SNN compiler pipeline."""

from __future__ import annotations

from typing import Dict, List

from .encoder import SynapseEncoder
from .graph import SpikingGraph
from .hdl_gen import HDLGenerator
from .mapper import NeuronMapper


def compile(spec: dict) -> dict:
    """
    Compile a SpikeFormer-style model spec into a full artifact.

    Pipeline: parse → map → encode synapses → generate HDL

    Parameters
    ----------
    spec : dict
        Model specification with 'layers' and 'connections' keys.

    Returns
    -------
    dict with keys:
        - graph       : SpikingGraph instance
        - mapping     : dict[str, int] — node_id to core_id
        - hdl_code    : str — Verilog-like pseudocode
        - synapse_events : list[dict] — spike events for all edges
    """
    # 1. Parse
    graph = SpikingGraph.from_dict(spec)

    # 2. Map neurons to cores
    mapper = NeuronMapper()
    mapping = mapper.map_graph(graph)

    # 3. Encode synaptic weights into spike events
    encoder = SynapseEncoder()
    synapse_events: List[dict] = []
    for i, edge in enumerate(graph.edges):
        timestamp_base = i * 1000  # stagger layers by 1000 ticks
        events = encoder.encode_layer(edge, timestamp_base=timestamp_base)
        synapse_events.extend(events)

    # 4. Generate HDL pseudocode
    hdl_gen = HDLGenerator()
    hdl_code = hdl_gen.generate_network(graph, mapping)

    return {
        "graph": graph,
        "mapping": mapping,
        "hdl_code": hdl_code,
        "synapse_events": synapse_events,
    }
