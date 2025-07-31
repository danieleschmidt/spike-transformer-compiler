# Development Guide

## Development Environment Setup

### Prerequisites

- Python 3.9+ 
- Git
- Virtual environment manager (venv, conda, etc.)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/yourusername/spike-transformer-compiler.git
cd spike-transformer-compiler

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
spike-compile --help
```

### Development Dependencies

```bash
# Core development tools
pip install -e ".[dev]"

# Optional: Loihi 3 SDK (requires Intel access)
pip install -e ".[loihi3]"

# Optional: Visualization tools
pip install -e ".[viz]"
```

## Project Structure

```
spike-transformer-compiler/
├── src/spike_transformer_compiler/  # Main package
│   ├── __init__.py
│   ├── cli.py                      # Command-line interface
│   ├── frontend/                   # Model parsing
│   ├── ir/                         # Intermediate representation
│   ├── optimizations/              # Optimization passes
│   ├── backend/                    # Code generation
│   ├── kernels/                    # Optimized kernels
│   └── runtime/                    # Runtime system
├── tests/                          # Test suite
├── docs/                           # Documentation
├── examples/                       # Usage examples
├── benchmarks/                     # Performance benchmarks
└── scripts/                        # Development scripts
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=spike_transformer_compiler

# Run specific test file
pytest tests/test_compiler.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8

# Type checking
mypy src/

# Run all checks
pre-commit run --all-files
```

### Building Documentation

```bash
cd docs/
make html
# Open docs/_build/html/index.html
```

## Adding New Features

### 1. Frontend Support
- Add model parsers in `src/spike_transformer_compiler/frontend/`
- Extend IR generation for new operations
- Add comprehensive tests

### 2. Optimization Passes
- Implement passes in `src/spike_transformer_compiler/optimizations/`
- Follow the pass interface pattern
- Include performance benchmarks

### 3. Backend Targets
- Add code generators in `src/spike_transformer_compiler/backend/`
- Implement hardware-specific optimizations
- Provide simulation capabilities

### 4. Kernels
- Add optimized kernels in `src/spike_transformer_compiler/kernels/`
- Include both CPU and neuromorphic implementations
- Benchmark against baselines

## Debugging

### Compiler Pipeline Debugging

```python
from spike_transformer_compiler import SpikeCompiler

compiler = SpikeCompiler(debug=True, verbose=True)
# Enable IR dumping
compiler.set_debug_options(dump_ir=True, dump_passes=True)
```

### Hardware Simulation

```python
# Use simulation backend for debugging
compiler = SpikeCompiler(target="simulation")
result = compiler.compile(model)
result.debug_trace()  # Show execution trace
```

## Performance Profiling

```bash
# Profile compilation time
python -m cProfile -o profile.stats scripts/profile_compilation.py

# Profile memory usage
python -m memory_profiler examples/compile_model.py

# Benchmark against baselines
python benchmarks/run_benchmarks.py
```

## Hardware Testing

### Loihi 3 Setup (Requires Intel Access)

```bash
# Install Loihi SDK
pip install nxsdk

# Configure hardware access
export LOIHI_GEN=3
export PYTHONPATH=$PYTHONPATH:/path/to/nxsdk

# Test hardware connection
python tests/test_loihi_connection.py
```

### Simulation Testing

```bash
# Run simulation tests
pytest tests/test_simulation.py

# Compare simulation vs hardware results
python scripts/validate_simulation.py
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite: `pytest`
4. Build package: `python -m build`
5. Test package: `pip install dist/*.whl`
6. Create release PR
7. Tag release: `git tag v0.1.0`
8. Push tags: `git push --tags`

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure package is installed with `-e` flag
2. **Test failures**: Check Python version compatibility
3. **Linting errors**: Run `pre-commit run --all-files`
4. **Type errors**: Update type stubs with `pip install types-all`

### Getting Help

- Check existing issues on GitHub
- Read the documentation
- Ask questions in discussions
- Contact maintainers for urgent issues

## Contributing Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.