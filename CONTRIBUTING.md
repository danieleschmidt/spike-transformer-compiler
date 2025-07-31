# Contributing to Spike-Transformer-Compiler

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/spike-transformer-compiler.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install development dependencies: `pip install -e ".[dev]"`
5. Install pre-commit hooks: `pre-commit install`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest`
4. Run linting: `black . && isort . && flake8`
5. Run type checking: `mypy src/`
6. Commit your changes: `git commit -m "Description of changes"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting
- Use isort for import sorting
- Add type hints for all functions
- Write docstrings for public APIs
- Keep line length to 88 characters

## Testing

- Write unit tests for new functionality
- Maintain test coverage above 80%
- Test both success and failure scenarios
- Include integration tests for compiler pipeline

## Documentation

- Update README.md if adding user-facing features
- Add docstrings to all public functions and classes
- Update API documentation for changes
- Include examples for new functionality

## Reporting Issues

- Use the GitHub issue tracker
- Provide clear reproduction steps
- Include relevant system information
- Add appropriate labels

## Questions?

Feel free to open an issue for questions or reach out to maintainers.