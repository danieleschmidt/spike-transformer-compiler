# Makefile for Spike-Transformer-Compiler

.PHONY: help install install-dev test lint format type-check clean docs build

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=spike_transformer_compiler --cov-report=html --cov-report=term-missing

lint:  ## Run linting
	flake8 src/ tests/

format:  ## Format code
	black src/ tests/
	isort src/ tests/

format-check:  ## Check code formatting
	black --check src/ tests/
	isort --check-only src/ tests/

type-check:  ## Run type checking
	mypy src/

quality:  ## Run all quality checks
	$(MAKE) format-check
	$(MAKE) lint  
	$(MAKE) type-check
	$(MAKE) test

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docs:  ## Build documentation
	cd docs && make html

docs-serve:  ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

build:  ## Build package
	python -m build

release-check:  ## Check if ready for release
	$(MAKE) quality
	$(MAKE) build
	twine check dist/*

# Development helpers
dev-setup:  ## Complete development setup
	python -m venv venv
	. venv/bin/activate && pip install -e ".[dev]"
	. venv/bin/activate && pre-commit install
	@echo "Development environment ready! Activate with: source venv/bin/activate"

benchmark:  ## Run performance benchmarks
	python benchmarks/run_benchmarks.py

profile:  ## Profile compilation performance
	python -m cProfile -o profile.stats scripts/profile_compilation.py