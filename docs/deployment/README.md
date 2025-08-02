# Deployment Documentation

This directory contains deployment guides and configurations for the Spike-Transformer-Compiler.

## Deployment Options

### Container Deployment
- [Docker Deployment](docker-deployment.md) - Single container deployment
- [Docker Compose](docker-compose-guide.md) - Multi-service development stack
- [Kubernetes Deployment](kubernetes-deployment.md) - Production Kubernetes setup

### Cloud Deployment
- [AWS Deployment](aws-deployment.md) - Amazon Web Services deployment
- [Google Cloud Deployment](gcp-deployment.md) - Google Cloud Platform setup
- [Azure Deployment](azure-deployment.md) - Microsoft Azure configuration

### Edge Deployment
- [Edge Device Setup](edge-deployment.md) - Deployment on edge devices
- [Embedded Systems](embedded-deployment.md) - Embedded system integration

### Hardware-Specific Deployment
- [Loihi 3 Hardware Setup](loihi3-deployment.md) - Intel Loihi 3 configuration
- [GPU Deployment](gpu-deployment.md) - NVIDIA GPU setup for baselines

## Quick Start

### Development Environment
```bash
# Start development environment
docker-compose up spike-compiler-dev

# Run tests
docker-compose --profile testing up spike-compiler-test

# Start monitoring stack
docker-compose --profile monitoring up prometheus grafana
```

### Production Deployment
```bash
# Build production image
docker build --target production -t spike-compiler:latest .

# Run production container
docker run -d --name spike-compiler spike-compiler:latest
```

### Hardware Deployment
```bash
# Loihi 3 deployment
docker-compose --profile hardware up loihi3-compiler

# GPU baseline deployment
docker-compose --profile gpu up gpu-baseline
```

## Configuration

### Environment Variables
See [Environment Configuration](../configuration/environment.md) for complete variable reference.

### Volume Mounts
- `compilation-cache`: Compilation artifacts cache
- `models-cache`: Pre-trained model cache
- `artifacts`: Deployment artifacts

### Network Configuration
- Default subnet: `172.20.0.0/16`
- Service discovery via Docker network

## Security Considerations

### Container Security
- Non-root user execution
- Read-only root filesystem where possible
- Resource limits configured
- Security scanning integrated

### Network Security
- Internal network isolation
- Minimal port exposure
- TLS encryption for external communication

### Data Security
- Secrets management via environment variables
- Encrypted storage for sensitive data
- Audit logging enabled

## Monitoring and Observability

### Health Checks
All services include comprehensive health checks:
- Application startup verification
- Dependency connectivity checks
- Resource utilization monitoring

### Metrics Collection
- Prometheus metrics scraping
- Grafana dashboards for visualization
- Custom metrics for neuromorphic workloads

### Logging
- Structured JSON logging
- Centralized log aggregation
- Log retention policies

## Troubleshooting

### Common Issues
1. **Container Build Failures**: Check Docker daemon and build context
2. **Port Conflicts**: Verify port availability before starting services
3. **Volume Mount Issues**: Ensure proper permissions and paths
4. **Hardware Access**: Verify device permissions for hardware services

### Debug Mode
Enable debug mode for development:
```bash
# Set debug environment
export DEBUG_MODE=true
export SPIKE_COMPILER_LOG_LEVEL=DEBUG

# Run with debug output
docker-compose up spike-compiler-dev
```

### Resource Requirements
- **Development**: 4GB RAM, 2 CPU cores
- **Testing**: 8GB RAM, 4 CPU cores  
- **Production**: 16GB RAM, 8 CPU cores
- **Hardware**: Additional requirements for Loihi 3 SDK

## Performance Optimization

### Build Optimization
- Multi-stage builds to minimize image size
- BuildKit caching for faster builds
- Layer optimization for better caching

### Runtime Optimization
- Resource limits to prevent resource exhaustion
- Volume caching for dependencies
- Health check tuning for faster recovery

### Scaling Considerations
- Horizontal scaling via container orchestration
- Load balancing for high availability
- Resource allocation based on workload patterns