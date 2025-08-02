#!/bin/bash

# Spike-Transformer-Compiler Development Container Post-Create Script
# This script runs once after the container is created

set -e

echo "🚀 Setting up Spike-Transformer-Compiler development environment..."

# Ensure we're in the correct directory
cd /workspaces/spike-transformer-compiler

# Install project in development mode
echo "📦 Installing project in development mode..."
pip install -e ".[dev]"

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p {logs,debug,models/cache,.cache/compilation}

# Set up Git configuration for better development experience
echo "⚙️  Configuring Git for development..."
git config --global --add safe.directory /workspaces/spike-transformer-compiler

# Initialize Git LFS if .gitattributes exists
if [ -f .gitattributes ]; then
    echo "🔄 Initializing Git LFS..."
    git lfs install
fi

# Download sample models for testing (if configured)
if [ -f scripts/download-sample-models.sh ]; then
    echo "📥 Downloading sample models..."
    bash scripts/download-sample-models.sh
fi

# Run initial tests to verify setup
echo "🧪 Running initial tests to verify setup..."
python -m pytest tests/ -x --tb=short -q || {
    echo "⚠️  Some tests failed - this is normal for a fresh setup"
    echo "   Run 'spike-test' to see detailed test results"
}

# Set up development database/cache if needed
echo "💾 Initializing development cache..."
python -c "
import os
from pathlib import Path

cache_dir = Path('.cache/compilation')
cache_dir.mkdir(parents=True, exist_ok=True)

# Create cache metadata
cache_metadata = cache_dir / 'metadata.json'
if not cache_metadata.exists():
    import json
    metadata = {
        'version': '1.0',
        'created': '$(date -Iseconds)',
        'max_size_gb': 10
    }
    with open(cache_metadata, 'w') as f:
        json.dump(metadata, f, indent=2)

print('✅ Development cache initialized')
"

# Check for required environment variables
echo "🔍 Checking environment configuration..."
python -c "
import os
import sys

required_vars = ['SPIKE_COMPILER_ENV']
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print(f'⚠️  Missing environment variables: {missing_vars}')
    print('   Copy .env.example to .env and configure your settings')
else:
    print('✅ Environment configuration looks good')
"

# Generate development documentation
echo "📚 Building development documentation..."
if [ -f docs/conf.py ]; then
    cd docs && make html && cd ..
    echo "✅ Documentation built successfully"
else
    echo "📝 Sphinx documentation not configured yet"
fi

# Set up IDE configuration
echo "⚙️  Configuring development tools..."
python -c "
import json
from pathlib import Path

# VS Code Python settings
vscode_dir = Path('.vscode')
vscode_dir.mkdir(exist_ok=True)

python_settings = {
    'python.defaultInterpreterPath': '/opt/venv/bin/python',
    'python.formatting.provider': 'black',
    'python.linting.enabled': True,
    'python.linting.pylintEnabled': True,
    'python.testing.pytestEnabled': True,
    'python.testing.pytestArgs': ['tests/'],
    'files.exclude': {
        '**/.cache': True,
        '**/__pycache__': True,
        '**/*.pyc': True
    }
}

settings_file = vscode_dir / 'settings.json'
if not settings_file.exists():
    with open(settings_file, 'w') as f:
        json.dump(python_settings, f, indent=2)
    print('✅ VS Code settings configured')
"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and configure your settings"
echo "  2. Run 'spike-test' to run the test suite"
echo "  3. Run 'spike-lint' to check code quality"
echo "  4. Run 'spike-format' to format code"
echo "  5. Start developing! 🚀"
echo ""
echo "Useful commands:"
echo "  spike-test      - Run test suite"
echo "  spike-lint      - Run linting"
echo "  spike-format    - Format code"
echo "  spike-typecheck - Run type checking"
echo ""

# Display system information
echo "System Information:"
echo "  Python: $(python --version)"
echo "  Pip: $(pip --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  Working Directory: $(pwd)"
echo "  Git Branch: $(git branch --show-current 2>/dev/null || echo 'No Git repository')"
echo ""