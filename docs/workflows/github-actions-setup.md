# GitHub Actions Setup Guide

This document describes the required GitHub Actions workflows for the Spike-Transformer-Compiler project.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)

**Location**: `.github/workflows/ci.yml`

**Triggers**: 
- Push to main branch
- Pull requests to main branch

**Jobs**:
```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Lint with flake8
      run: flake8 src/ tests/
    
    - name: Check formatting with black
      run: black --check src/ tests/
    
    - name: Type check with mypy
      run: mypy src/
    
    - name: Test with pytest
      run: pytest --cov=spike_transformer_compiler --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
```

### 2. Security Scanning (`security.yml`)

**Location**: `.github/workflows/security.yml`

**Triggers**: 
- Schedule: weekly
- Push to main branch

**Features**:
- Dependency vulnerability scanning with Safety
- Code security scanning with Bandit
- SBOM generation
- License compliance checking

### 3. Documentation (`docs.yml`)

**Location**: `.github/workflows/docs.yml`

**Triggers**:
- Push to main branch
- Release tags

**Features**:
- Build Sphinx documentation
- Deploy to GitHub Pages
- Link checking
- API documentation generation

### 4. Release (`release.yml`)

**Location**: `.github/workflows/release.yml`

**Triggers**:
- GitHub release creation

**Features**:
- Build Python packages
- Run comprehensive test suite
- Upload to PyPI
- Create release artifacts

## Setup Instructions

1. **Create workflow files** in `.github/workflows/` directory
2. **Configure secrets** in repository settings:
   - `PYPI_API_TOKEN`: For PyPI uploads
   - `CODECOV_TOKEN`: For coverage reporting
3. **Enable GitHub Pages** for documentation deployment
4. **Configure branch protection** for main branch:
   - Require status checks to pass
   - Require up-to-date branches
   - Restrict pushes to main

## Additional Recommendations

### Pre-commit Integration
- Add pre-commit workflow validation
- Sync pre-commit config with CI checks

### Performance Monitoring
- Add benchmark regression detection
- Performance comparison against baselines
- Energy efficiency tracking

### Hardware Testing
- Add simulation-based hardware tests
- Integration with Loihi 3 SDK (when available)
- Cross-platform compatibility testing

## Repository Settings

### Branch Protection Rules
```yaml
main:
  required_status_checks:
    - ci (3.9)
    - ci (3.10) 
    - ci (3.11)
    - ci (3.12)
  enforce_admins: true
  required_pull_request_reviews:
    required_approving_review_count: 1
  restrictions: null
```

### Issue and PR Templates
- Bug report template: `.github/ISSUE_TEMPLATE/bug_report.md` ✅
- Feature request template: `.github/ISSUE_TEMPLATE/feature_request.md` ✅  
- Pull request template: `.github/pull_request_template.md` ✅

### Labels
Recommended labels for issue and PR management:
- `bug`, `enhancement`, `documentation`
- `good first issue`, `help wanted`
- `priority-high`, `priority-medium`, `priority-low`
- `neuromorphic`, `compilation`, `optimization`
- `loihi3`, `simulation`, `performance`