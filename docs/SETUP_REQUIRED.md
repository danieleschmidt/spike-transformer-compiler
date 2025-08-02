# Manual Setup Required

This document outlines the manual setup steps required to complete the SDLC implementation due to GitHub App permission limitations.

## GitHub Workflow Creation

### Required Workflows

The following workflow files need to be manually created in `.github/workflows/` directory:

1. **CI Workflow** - Copy from `docs/workflows/examples/ci.yml` to `.github/workflows/ci.yml`
2. **Security Scanning** - Copy from `docs/workflows/examples/security.yml` to `.github/workflows/security.yml`
3. **Continuous Deployment** - Copy from `docs/workflows/examples/cd.yml` to `.github/workflows/cd.yml`
4. **Dependency Updates** - Copy from `docs/workflows/examples/dependency-update.yml` to `.github/workflows/dependency-update.yml`

### Manual Creation Commands

```bash
# Create .github/workflows directory
mkdir -p .github/workflows

# Copy workflow files
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/security.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

## Repository Settings Configuration

### Branch Protection Rules

Configure branch protection for the `main` branch:

1. Go to Settings → Branches
2. Add rule for `main` branch
3. Configure:
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Require pull request reviews before merging (1 reviewer)
   - ✅ Dismiss stale PR approvals when new commits are pushed
   - ✅ Restrict pushes that create files larger than 100MB
   - ✅ Include administrators in restrictions

### Required Secrets

Add the following secrets in Settings → Secrets and variables → Actions:

#### CI/CD Secrets
- `CODECOV_TOKEN` - For code coverage reporting
- `PYPI_API_TOKEN` - For publishing to PyPI
- `DEPENDENCY_UPDATE_TOKEN` - Personal access token for dependency updates

#### Deployment Secrets (if applicable)
- `DEPLOY_SSH_KEY` - SSH key for deployment servers
- `DEPLOY_HOST` - Deployment server hostname
- `DOCKER_REGISTRY_TOKEN` - Container registry authentication

### Repository Labels

Create the following labels for issue and PR management:

```bash
# Install GitHub CLI if not already installed
gh auth login

# Create labels
gh label create "bug" --color "d73a4a" --description "Something isn't working"
gh label create "enhancement" --color "a2eeef" --description "New feature or request"
gh label create "documentation" --color "0075ca" --description "Improvements or additions to documentation"
gh label create "good first issue" --color "7057ff" --description "Good for newcomers"
gh label create "help wanted" --color "008672" --description "Extra attention is needed"
gh label create "priority-high" --color "ff0000" --description "High priority issue"
gh label create "priority-medium" --color "ff9500" --description "Medium priority issue"
gh label create "priority-low" --color "00ff00" --description "Low priority issue"
gh label create "neuromorphic" --color "6f42c1" --description "Related to neuromorphic computing"
gh label create "compilation" --color "b60205" --description "Related to model compilation"
gh label create "optimization" --color "fbca04" --description "Performance optimization"
gh label create "loihi3" --color "0e8a16" --description "Intel Loihi 3 specific"
gh label create "simulation" --color "1d76db" --description "Software simulation"
gh label create "automated" --color "ededed" --description "Automated PR or issue"
gh label create "dependencies" --color "0366d6" --description "Pull requests that update a dependency file"
```

## GitHub Pages Setup

### Enable GitHub Pages

1. Go to Settings → Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages` (will be created by docs workflow)
4. Folder: `/ (root)`

### Documentation Build

The documentation workflow will automatically:
- Build documentation using Sphinx/MkDocs
- Deploy to GitHub Pages
- Update on every push to main branch

## Issue and PR Templates

### Create Template Directory

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

### Bug Report Template

Create `.github/ISSUE_TEMPLATE/bug_report.md`:

```markdown
---
name: Bug report
about: Create a report to help us improve
title: ''
labels: 'bug'
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment:**
 - OS: [e.g. Ubuntu 20.04]
 - Python Version: [e.g. 3.11]
 - Package Version: [e.g. 0.1.0]
 - Hardware: [e.g. Intel Loihi 3, Simulation]

**Additional context**
Add any other context about the problem here.
```

### Feature Request Template

Create `.github/ISSUE_TEMPLATE/feature_request.md`:

```markdown
---
name: Feature request
about: Suggest an idea for this project
title: ''
labels: 'enhancement'
assignees: ''
---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

### Pull Request Template

Create `.github/pull_request_template.md`:

```markdown
## Description

Brief description of changes.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance benchmarks run (if applicable)

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] CHANGELOG.md updated (if applicable)

## Related Issues

Closes #(issue_number)
```

## CODEOWNERS File

Create `.github/CODEOWNERS`:

```
# Global owners
* @maintenance-team

# Core compilation engine
/src/spike_transformer_compiler/compiler.py @compiler-experts
/src/spike_transformer_compiler/optimization.py @optimization-team

# Backend implementations
/src/spike_transformer_compiler/backend/ @backend-team

# Documentation
/docs/ @docs-team
*.md @docs-team

# CI/CD and workflows
/.github/ @devops-team
/docker-compose.yml @devops-team
/Dockerfile @devops-team

# Security-related files
/SECURITY.md @security-team
```

## Monitoring and Alerting Setup

### External Monitoring (Optional)

If using external monitoring services:

1. **Sentry** - Error tracking and performance monitoring
2. **DataDog** - Infrastructure and application monitoring
3. **PagerDuty** - Incident response and alerting

### Webhook Configuration

For automated notifications, configure webhooks in repository settings:

- Slack notifications for PR reviews
- Discord notifications for releases
- Email notifications for security alerts

## Verification Checklist

After completing manual setup:

- [ ] All workflows are created and active
- [ ] Branch protection rules are configured
- [ ] Required secrets are added
- [ ] Labels are created
- [ ] GitHub Pages is enabled
- [ ] Issue/PR templates are working
- [ ] CODEOWNERS file is active
- [ ] First CI run is successful
- [ ] Security scanning is working
- [ ] Documentation builds successfully

## Support

If you encounter issues during setup:

1. Check the [troubleshooting guide](docs/troubleshooting.md)
2. Review workflow logs in the Actions tab
3. Verify all required secrets are properly configured
4. Ensure repository permissions are correctly set

For additional help, create an issue with the "help wanted" label.
