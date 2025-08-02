# Multi-stage build for Spike-Transformer-Compiler
# Optimized for neuromorphic computing workloads

# ============================================================================
# Base Stage - Common dependencies and setup
# ============================================================================
FROM python:3.11-slim as base

# Set environment variables for Python and security
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_TIMEOUT=100 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for neuromorphic computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    # Scientific computing libraries
    libblas-dev \
    liblapack-dev \
    libfftw3-dev \
    # System utilities
    git \
    curl \
    wget \
    # Security and monitoring
    ca-certificates \
    # Cleanup
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Set working directory
WORKDIR /app

# ============================================================================
# Development Stage - Full development environment
# ============================================================================
FROM base as development

# Install development dependencies
COPY requirements-dev.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code and configuration
COPY . .

# Install package in development mode with all extras
RUN pip install -e ".[dev,test,docs]"

# Create directories for development
RUN mkdir -p logs debug artifacts models/cache .cache

# Development health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import spike_transformer_compiler; print('OK')" || exit 1

# Default development command
CMD ["python", "-m", "spike_transformer_compiler.cli", "--help"]

# ============================================================================
# Production Stage - Minimal production image
# ============================================================================
FROM base as production

# Create non-root user for security
RUN groupadd --gid 1000 compiler && \
    useradd --uid 1000 --gid compiler --shell /bin/bash --create-home compiler

# Switch to non-root user
USER compiler
WORKDIR /home/compiler/app

# Copy only necessary production files
COPY --chown=compiler:compiler requirements.txt ./
COPY --chown=compiler:compiler src/ src/
COPY --chown=compiler:compiler pyproject.toml ./
COPY --chown=compiler:compiler README.md LICENSE ./

# Install production dependencies
RUN pip install --user --no-cache-dir -r requirements.txt && \
    pip install --user --no-cache-dir -e .

# Add user's local bin to PATH
ENV PATH="/home/compiler/.local/bin:${PATH}"

# Create necessary directories
RUN mkdir -p logs artifacts models/cache

# Production health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import spike_transformer_compiler; print('OK')" || exit 1

# Set resource limits and security
ENV PYTHONMALLOC=malloc \
    MALLOC_ARENA_MAX=2

# Default production command
CMD ["python", "-m", "spike_transformer_compiler.cli", "--help"]

# ============================================================================
# Loihi3 Stage - Hardware deployment with Loihi 3 SDK
# ============================================================================
FROM production as loihi3

# Switch back to root for system package installation
USER root

# Install Loihi 3 SDK system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Hardware interface dependencies
    libudev-dev \
    libusb-1.0-0-dev \
    # Networking for multi-chip communication
    libzmq3-dev \
    # Performance libraries
    libnuma-dev \
    # Cleanup
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directory for Loihi SDK (placeholder)
RUN mkdir -p /opt/intel/loihi3-sdk && \
    echo "# Loihi 3 SDK would be installed here" > /opt/intel/loihi3-sdk/README.txt && \
    chown -R compiler:compiler /opt/intel

# Switch back to non-root user
USER compiler

# Install Loihi 3 specific Python dependencies
RUN pip install --user --no-cache-dir \
    # Placeholder for Loihi 3 Python packages
    networkx>=3.0 \
    graphviz>=0.20 \
    # Hardware monitoring
    psutil>=5.9.0 \
    # Communication protocols
    pyzmq>=25.0.0 || echo "Some Loihi3 dependencies not available"

# Set Loihi 3 environment variables
ENV LOIHI3_SDK_PATH="/opt/intel/loihi3-sdk" \
    PYTHONPATH="/opt/intel/loihi3-sdk/python:${PYTHONPATH}" \
    LD_LIBRARY_PATH="/opt/intel/loihi3-sdk/lib:${LD_LIBRARY_PATH}"

# Loihi3 health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import spike_transformer_compiler.backend.loihi3; print('Loihi3 OK')" || exit 1

# ============================================================================
# GPU Stage - GPU-enabled for baseline comparisons
# ============================================================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as gpu

# Install Python and basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Set Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN groupadd --gid 1000 compiler && \
    useradd --uid 1000 --gid compiler --shell /bin/bash --create-home compiler

USER compiler
WORKDIR /home/compiler/app

# Copy requirements and source
COPY --chown=compiler:compiler requirements.txt ./
COPY --chown=compiler:compiler src/ src/
COPY --chown=compiler:compiler pyproject.toml ./

# Install GPU-enabled dependencies
RUN pip install --user --no-cache-dir \
    torch>=2.0.0+cu121 \
    torchvision>=0.15.0+cu121 \
    -f https://download.pytorch.org/whl/cu121/torch_stable.html && \
    pip install --user --no-cache-dir -e .

# GPU health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || exit 1

# ============================================================================
# Documentation Stage - For building documentation
# ============================================================================
FROM base as docs

# Install documentation dependencies
RUN pip install --no-cache-dir \
    sphinx>=6.0.0 \
    sphinx-rtd-theme>=1.2.0 \
    myst-parser>=1.0.0 \
    sphinx-autodoc-typehints>=1.20.0

# Copy source and docs
COPY . .

# Build documentation
RUN cd docs && make html

# Serve documentation
EXPOSE 8000
CMD ["python", "-m", "http.server", "8000", "--directory", "docs/_build/html"]

# ============================================================================
# Security scanning stage
# ============================================================================
FROM base as security

# Install security scanning tools
RUN pip install --no-cache-dir \
    bandit[toml]>=1.7.0 \
    safety>=2.3.0 \
    semgrep>=1.0.0

# Copy source code
COPY . .

# Run security scans
RUN bandit -r src/ -f json -o security-report.json || true
RUN safety check --json --output safety-report.json || true

# Security scan results
CMD ["cat", "security-report.json"]