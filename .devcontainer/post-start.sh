#!/bin/bash

# Spike-Transformer-Compiler Development Container Post-Start Script
# This script runs every time the container starts

set -e

echo "🔄 Starting Spike-Transformer-Compiler development session..."

# Ensure we're in the correct directory
cd /workspaces/spike-transformer-compiler

# Activate virtual environment
source /opt/venv/bin/activate

# Check if .env file exists and load it
if [ -f .env ]; then
    echo "📄 Loading environment configuration from .env"
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️  No .env file found. Using default configuration."
    echo "   Copy .env.example to .env to customize settings"
fi

# Update development dependencies if needed
if [ "${SPIKE_COMPILER_ENV}" = "development" ]; then
    echo "🔧 Checking for dependency updates..."
    pip install --upgrade pip > /dev/null 2>&1
fi

# Clean up any temporary files from previous sessions
echo "🧹 Cleaning up temporary files..."
find . -name "*.pyc" -delete > /dev/null 2>&1 || true
find . -name "__pycache__" -type d -exec rm -rf {} + > /dev/null 2>&1 || true

# Check disk usage for cache directories
echo "💾 Checking cache usage..."
python -c "
import os
from pathlib import Path

def get_dir_size(path):
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except (OSError, FileNotFoundError):
        pass
    return total

cache_dir = Path('.cache')
if cache_dir.exists():
    size_mb = get_dir_size(cache_dir) / 1024 / 1024
    print(f'Cache size: {size_mb:.1f} MB')
    if size_mb > 1000:  # 1GB
        print('⚠️  Cache is large, consider cleaning with: rm -rf .cache/*')
else:
    print('No cache directory found')
"

# Check for any running background processes that might conflict
echo "🔍 Checking for conflicting processes..."
if pgrep -f "spike.*compiler" > /dev/null; then
    echo "⚠️  Found existing spike-compiler processes"
fi

# Display git status if in a git repository
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "📊 Git status:"
    git status --porcelain | head -5
    if [ $(git status --porcelain | wc -l) -gt 5 ]; then
        echo "... and $(git status --porcelain | wc -l) more files"
    fi
    
    # Show current branch
    current_branch=$(git branch --show-current)
    echo "🌿 Current branch: ${current_branch}"
    
    # Check if there are uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        echo "📝 You have uncommitted changes"
    fi
fi

# Start background services if configured
if [ "${ENERGY_PROFILING_ENABLED}" = "true" ]; then
    echo "⚡ Energy profiling is enabled"
fi

if [ "${PERFORMANCE_MONITORING_ENABLED}" = "true" ]; then
    echo "📈 Performance monitoring is enabled"
fi

# Check for any important updates or notifications
if [ -f .devcontainer/UPDATES.md ]; then
    echo "📢 Development updates available:"
    head -5 .devcontainer/UPDATES.md
fi

# Display quick help
echo ""
echo "🎯 Quick Start:"
echo "  spike-test      - Run tests"
echo "  spike-lint      - Check code quality" 
echo "  spike-format    - Format code"
echo "  make build      - Build project"
echo "  make help       - Show all available commands"
echo ""

# Set up shell prompt with project context
if [ "$SHELL" = "/bin/bash" ] || [ "$SHELL" = "/usr/bin/bash" ]; then
    export PS1="[spike-compiler] \u@\h:\w\$ "
elif [ "$SHELL" = "/bin/zsh" ] || [ "$SHELL" = "/usr/bin/zsh" ]; then
    export PS1="[spike-compiler] %n@%m:%~$ "
fi

echo "✅ Development session ready!"