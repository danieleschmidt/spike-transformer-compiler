# spike-transformer-compiler

Compiler toolchain for Spiking Neural Network (SNN) models targeting neuromorphic hardware. Parses SpikeFormer-style model specs, maps neurons to Loihi-style cores, encodes event-driven synaptic weights, and generates Verilog-like HDL pseudocode.

## Install

```bash
pip install numpy
```

## Usage

```python
from spike_compiler.demo import compile_3layer_snn

artifact = compile_3layer_snn()
print(artifact['hdl_code'])
```

## Architecture

| Module | Class | Responsibility |
|--------|-------|---------------|
| `graph.py` | `SpikingGraph` | Parse SpikeFormer spec → DAG of Nodes/Edges |
| `mapper.py` | `NeuronMapper` | Greedy packing of nodes onto Loihi-style cores |
| `encoder.py` | `SynapseEncoder` | Sparse CSR weight encoding + spike events |
| `hdl_gen.py` | `HDLGenerator` | Verilog-like HDL pseudocode generation |
| `compiler.py` | `compile()` | End-to-end pipeline |
| `demo.py` | `compile_3layer_snn()` | 3-layer demo: input→attn→ffn→output |

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

MIT
