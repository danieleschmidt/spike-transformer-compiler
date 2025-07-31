# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please follow these steps:

1. **Do not** create a public GitHub issue for the vulnerability
2. Send an email to security@terragonlabs.ai with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes

## Security Considerations

### Model Security
- Compiled models may contain sensitive information
- Ensure proper access controls for deployed models
- Validate all input data before processing

### Hardware Security
- Loihi 3 deployments should use secure communication channels
- Implement proper authentication for hardware access
- Monitor for unusual activity patterns

### Data Protection
- Training data should be handled according to privacy regulations
- Implement proper data sanitization for logs
- Use encrypted storage for sensitive model parameters

## Vulnerability Response

- Initial response within 48 hours
- Security patches released as soon as possible
- Public disclosure after fixes are available
- Credit given to responsible reporters

## Security Best Practices

1. Keep dependencies updated
2. Use virtual environments
3. Validate all inputs
4. Follow principle of least privilege
5. Regular security audits

For more information about neuromorphic security considerations, see our [Security Documentation](docs/security.md).