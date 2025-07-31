# Security Compliance Guide

This document outlines the security compliance framework for Spike-Transformer-Compiler.

## SLSA Compliance

The project targets **SLSA Level 2** compliance for supply chain security.

### SLSA Level 2 Requirements

✅ **Build Requirements**
- Scripted build process (Makefile, pyproject.toml)
- Version controlled source (Git)
- Build isolation (Docker containers)
- Authenticated builds (GitHub Actions with OIDC)

✅ **Provenance Requirements**  
- Build provenance generation
- In-toto attestation format
- GitHub attestations storage
- Artifact signing with sigstore

### Implementation Status

| Requirement | Status | Implementation |
|-------------|---------|----------------|
| Scripted Build | ✅ | Makefile + pyproject.toml |
| Version Control | ✅ | Git with signed commits |
| Build Isolation | ✅ | Docker multi-stage builds |
| Provenance Generation | 📋 | GitHub Actions workflow |
| Artifact Signing | 📋 | Sigstore integration |

## SBOM Generation

Software Bill of Materials (SBOM) generation is configured for transparency.

### SBOM Configuration

- **Format**: SPDX 2.3
- **Output**: `sbom.spdx.json`
- **Scope**: Production dependencies only
- **Generation**: Automated in CI/CD

### SBOM Contents

```json
{
  "spdxVersion": "SPDX-2.3",
  "creationInfo": {
    "created": "2025-07-31T00:00:00Z",
    "creators": ["Tool: pip-tools", "Organization: Terragon Labs"]
  },
  "packages": [
    "Dependencies automatically discovered"
  ]
}
```

## Vulnerability Management

### Dependency Scanning

**Tools**:
- `safety`: Python dependency vulnerability scanning
- `pip-audit`: OSS vulnerability database integration  
- `bandit`: Static security analysis for Python

**Configuration**:
```bash
# Run vulnerability scans
make security-scan

# Update vulnerability database
safety --db update

# Generate security report
bandit -r src/ -f json -o security-report.json
```

### Container Security

**Base Image Security**:
- Minimal Python slim images
- Regular base image updates
- Multi-stage builds for reduced attack surface

**Runtime Security**:
- Non-root user execution
- Read-only filesystem where possible
- Capability dropping

## Security Workflows

### Pre-commit Security Hooks

```yaml
# .pre-commit-config.yaml additions
- repo: https://github.com/PyCQA/bandit
  rev: 1.7.5
  hooks:
    - id: bandit
      args: ['-r', 'src/']

- repo: https://github.com/gitguardian/ggshield
  rev: v1.18.0
  hooks:
    - id: ggshield
      language: python
      stages: [commit]
```

### CI/CD Security Pipeline

**GitHub Actions Security Steps**:
1. Dependency vulnerability scanning
2. SAST (Static Application Security Testing)
3. Container security scanning
4. SBOM generation and attestation
5. Artifact signing with sigstore

## Incident Response

### Vulnerability Disclosure

Follow responsible disclosure process outlined in [SECURITY.md](../SECURITY.md):

1. **Report**: Email security@terragonlabs.ai
2. **Acknowledge**: 48-hour response commitment
3. **Assess**: Security team evaluation
4. **Fix**: Coordinated patch development
5. **Disclose**: Public disclosure after fix

### Security Monitoring

**Automated Monitoring**:
- Dependabot for dependency updates
- CodeQL for code security analysis
- GitHub Advisory Database integration

**Manual Reviews**:
- Quarterly security architecture reviews
- Annual third-party security audits
- Penetration testing for production deployments

## Compliance Frameworks

### NIST Secure Software Development Framework (SSDF)

Implementation mapping:
- **Prepare the Organization (PO)**: Security training, secure development practices
- **Protect the Software (PS)**: Code review, security testing, SAST/DAST
- **Produce Well-Secured Software (PW)**: Secure coding, vulnerability management
- **Respond to Vulnerabilities (RV)**: Incident response, patch management

### Industry Standards

- **ISO 27001**: Information security management
- **SOC 2 Type II**: Security and availability controls
- **NIST Cybersecurity Framework**: Risk management approach

## Security Metrics

### Key Performance Indicators

```json
{
  "vulnerability_metrics": {
    "mean_time_to_detection": "< 24 hours",
    "mean_time_to_remediation": "< 7 days",
    "critical_vulnerability_sla": "< 72 hours"
  },
  "build_security": {
    "signed_commits_percentage": "> 95%",
    "automated_security_scanning": "100%",
    "slsa_compliance_level": "Level 2"
  },
  "dependency_management": {
    "outdated_dependencies": "< 5%",
    "known_vulnerabilities": "0 critical/high",
    "license_compliance": "100%"
  }
}
```

## Getting Started

### Enable Security Features

```bash
# Install security tools
pip install safety bandit pip-audit

# Run security scan
make security-scan

# Generate SBOM
make generate-sbom

# Verify SLSA compliance
make verify-slsa
```

### Security Review Checklist

- [ ] All dependencies scanned for vulnerabilities
- [ ] SAST scan passed without critical issues
- [ ] SBOM generated and verified
- [ ] Container security scan passed
- [ ] Secrets detection scan passed
- [ ] License compliance verified

For questions about security compliance, contact the security team at security@terragonlabs.ai.