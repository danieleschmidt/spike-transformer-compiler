#!/bin/bash
# Development environment setup script for Spike-Transformer-Compiler

set -e

echo "🚀 Setting up Spike-Transformer-Compiler development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(echo "$python_version >= $required_version" | bc -l)" -eq 0 ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "📥 Installing package dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs
mkdir -p data/models
mkdir -p data/benchmarks
mkdir -p .security/reports

# Download sample models (placeholder)
echo "📊 Setting up sample data..."
echo "# Sample models will be downloaded here" > data/models/README.md
echo "# Benchmark data will be stored here" > data/benchmarks/README.md

# Set up monitoring directories
mkdir -p monitoring/logs
echo "# Monitoring logs" > monitoring/logs/README.md

# Create .env template
if [ ! -f ".env" ]; then
    echo "⚙️  Creating environment configuration..."
    cat > .env << 'EOL'
# Development environment configuration
PYTHONPATH=src/
LOG_LEVEL=DEBUG
MONITORING_ENABLED=true
METRICS_PORT=8000

# Hardware configuration (optional)
# LOIHI_SDK_PATH=/path/to/nxsdk
# LOIHI_GEN=3

# Security settings
SECURITY_SCANNING_ENABLED=true
VULNERABILITY_THRESHOLD=medium

# Docker settings
COMPOSE_PROJECT_NAME=spike-compiler
EOL
fi

# Run initial tests to verify setup
echo "🧪 Running initial tests..."
python -m pytest tests/test_compiler.py -v

# Run code quality checks
echo "🔍 Running code quality checks..."
make format-check
make lint
make type-check

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "Available commands:"
echo "  make help          - Show all available commands"
echo "  make test          - Run test suite"
echo "  make quality       - Run all quality checks"
echo "  make docs          - Build documentation"
echo ""
echo "VS Code users: Install recommended extensions from .vscode/extensions.json"
echo "Docker users: Run 'docker-compose up spike-compiler-dev' for containerized development"