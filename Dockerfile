# Multi-stage build for Spike-Transformer-Compiler
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e ".[dev]"

# Production stage  
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash compiler
USER compiler
WORKDIR /home/compiler/app

# Copy only necessary files
COPY --chown=compiler:compiler requirements.txt .
COPY --chown=compiler:compiler src/ src/
COPY --chown=compiler:compiler pyproject.toml .
COPY --chown=compiler:compiler README.md .
COPY --chown=compiler:compiler LICENSE .

# Install production dependencies
RUN pip install --user -e .

# Add user's local bin to PATH
ENV PATH="/home/compiler/.local/bin:${PATH}"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD spike-compile --help || exit 1

# Default command
CMD ["spike-compile", "--help"]

# Loihi3 stage (for hardware deployment)
FROM production as loihi3

USER root

# Install Loihi 3 SDK dependencies (placeholder - requires Intel access)
RUN echo "# Loihi 3 SDK installation would go here" > /etc/loihi3-placeholder

USER compiler

# Install optional Loihi 3 dependencies
RUN pip install --user -e ".[loihi3]" || echo "Loihi3 SDK not available in container"