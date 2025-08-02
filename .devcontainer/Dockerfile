# Spike-Transformer-Compiler Development Container
FROM mcr.microsoft.com/vscode/devcontainers/python:3.11-bullseye

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        # Build tools
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        # Scientific computing libraries
        libblas-dev \
        liblapack-dev \
        libfftw3-dev \
        # Visualization and GUI
        graphviz \
        xvfb \
        # Networking and debugging
        curl \
        wget \
        htop \
        tree \
        jq \
        # Additional utilities
        zip \
        unzip \
        git-lfs \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install base packages
RUN pip install --upgrade pip setuptools wheel

# Install core Python dependencies
RUN pip install \
    # Core scientific stack
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    # Deep learning frameworks
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    # Data processing
    pandas>=2.0.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    # Development tools
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    pytest-xdist>=3.0.0 \
    black>=23.0.0 \
    flake8>=6.0.0 \
    mypy>=1.0.0 \
    isort>=5.12.0 \
    # Documentation
    sphinx>=6.0.0 \
    sphinx-rtd-theme>=1.2.0 \
    myst-parser>=1.0.0 \
    # Jupyter and notebooks
    jupyter>=1.0.0 \
    jupyterlab>=4.0.0 \
    ipywidgets>=8.0.0 \
    # Profiling and monitoring
    py-spy>=0.3.0 \
    memory-profiler>=0.61.0 \
    line-profiler>=4.0.0 \
    # Utilities
    tqdm>=4.65.0 \
    rich>=13.0.0 \
    click>=8.1.0

# Install development tools
RUN pip install \
    # Pre-commit hooks
    pre-commit>=3.0.0 \
    # Code quality
    bandit>=1.7.0 \
    safety>=2.3.0 \
    # Build tools
    build>=0.10.0 \
    twine>=4.0.0

# Set up workspace
WORKDIR /workspaces/spike-transformer-compiler

# Create directories for development
RUN mkdir -p \
    /workspaces/spike-transformer-compiler/.cache \
    /workspaces/spike-transformer-compiler/logs \
    /workspaces/spike-transformer-compiler/models \
    /workspaces/spike-transformer-compiler/debug

# Configure Git (will be overridden by user's .gitconfig mount)
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false \
    && git config --global core.autocrlf input

# Set up shell environment
RUN echo 'export PATH="/workspaces/spike-transformer-compiler/scripts:$PATH"' >> /home/vscode/.bashrc \
    && echo 'export PYTHONPATH="/workspaces/spike-transformer-compiler/src:$PYTHONPATH"' >> /home/vscode/.bashrc \
    && echo 'source /opt/venv/bin/activate' >> /home/vscode/.bashrc

# Install Oh My Zsh for better terminal experience
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && echo 'export PATH="/workspaces/spike-transformer-compiler/scripts:$PATH"' >> /home/vscode/.zshrc \
    && echo 'export PYTHONPATH="/workspaces/spike-transformer-compiler/src:$PYTHONPATH"' >> /home/vscode/.zshrc \
    && echo 'source /opt/venv/bin/activate' >> /home/vscode/.zshrc

# Set proper ownership
RUN chown -R vscode:vscode /opt/venv /workspaces

# Switch to vscode user
USER vscode

# Set up development aliases
RUN echo 'alias ll="ls -la"' >> ~/.bashrc \
    && echo 'alias la="ls -la"' >> ~/.bashrc \
    && echo 'alias spike-test="python -m pytest"' >> ~/.bashrc \
    && echo 'alias spike-lint="flake8 src/ tests/"' >> ~/.bashrc \
    && echo 'alias spike-format="black src/ tests/ && isort src/ tests/"' >> ~/.bashrc \
    && echo 'alias spike-typecheck="mypy src/"' >> ~/.bashrc

# Copy same aliases to zsh
RUN echo 'alias ll="ls -la"' >> ~/.zshrc \
    && echo 'alias la="ls -la"' >> ~/.zshrc \
    && echo 'alias spike-test="python -m pytest"' >> ~/.zshrc \
    && echo 'alias spike-lint="flake8 src/ tests/"' >> ~/.zshrc \
    && echo 'alias spike-format="black src/ tests/ && isort src/ tests/"' >> ~/.zshrc \
    && echo 'alias spike-typecheck="mypy src/"' >> ~/.zshrc

# Set default working directory
WORKDIR /workspaces/spike-transformer-compiler

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; print(f'Python {sys.version}'); import torch; print(f'PyTorch {torch.__version__}')" || exit 1