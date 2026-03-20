"""Demo: compile a 3-layer SNN using spike_compiler."""

from __future__ import annotations

import numpy as np

from .compiler import compile


def compile_3layer_snn() -> dict:
    """
    Build a 3-layer SNN spec and compile it end-to-end.

    Topology: input(16) → spiking_attn(32) → spiking_ffn(32) → output(4)

    Random weight matrices are generated for each connection.

    Returns the compiled artifact dict from :func:`spike_compiler.compiler.compile`.
    """
    rng = np.random.default_rng(42)

    # Random sparse weights (70% zeros to mimic biological sparsity)
    def sparse_weights(rows: int, cols: int) -> list:
        W = rng.uniform(-1.0, 1.0, (rows, cols))
        mask = rng.random((rows, cols)) < 0.70
        W[mask] = 0.0
        return W.tolist()

    spec = {
        "layers": [
            {"id": "input",   "layer_type": "input",       "n_neurons": 16},
            {"id": "attn0",   "layer_type": "spiking_attn","n_neurons": 32,
             "params": {"heads": 4, "head_dim": 8}},
            {"id": "ffn0",    "layer_type": "spiking_ffn", "n_neurons": 32,
             "params": {"hidden_dim": 64}},
            {"id": "output",  "layer_type": "output",      "n_neurons": 4},
        ],
        "connections": [
            ["input",  "attn0",  sparse_weights(16, 32)],
            ["attn0",  "ffn0",   sparse_weights(32, 32)],
            ["ffn0",   "output", sparse_weights(32, 4)],
        ],
    }

    return compile(spec)


if __name__ == "__main__":
    artifact = compile_3layer_snn()
    print("=== Mapping ===")
    for node_id, core_id in artifact["mapping"].items():
        print(f"  {node_id} → core {core_id}")
    print("\n=== HDL Code ===")
    print(artifact["hdl_code"])
    print(f"\n=== Synapse Events: {len(artifact['synapse_events'])} total ===")
    if artifact["synapse_events"]:
        print("  First event:", artifact["synapse_events"][0])
