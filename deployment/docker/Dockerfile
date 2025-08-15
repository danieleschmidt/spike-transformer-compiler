# Production-ready Dockerfile for Spike Transformer Compiler
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./

# Install the package in development mode
RUN pip install -e .

# Create directories for runtime
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 9090

# Default command
CMD ["python", "-m", "spike_transformer_compiler.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]


# Development stage
FROM base as development

USER root

# Install development dependencies
RUN pip install -r requirements-dev.txt

# Install debugging and development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    strace \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Override command for development
CMD ["python", "-m", "spike_transformer_compiler.cli", "serve", "--host", "0.0.0.0", "--port", "8080", "--debug"]


# Production stage
FROM base as production

# Copy configuration files
COPY deployment/config/ ./config/

# Set production environment
ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    METRICS_ENABLED=true

# Run security scan and cleanup (if available)
USER root
RUN apt-get update && apt-get upgrade -y && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER appuser

# Production command with optimizations
CMD ["python", "-O", "-m", "spike_transformer_compiler.cli", "serve", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "4", "--production"]