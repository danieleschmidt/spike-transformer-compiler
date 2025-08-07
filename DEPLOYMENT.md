# Production Deployment Guide

This guide covers deploying the Spike-Transformer-Compiler in production environments.

## 🚀 Deployment Overview

The Spike-Transformer-Compiler is designed for production deployment across multiple environments:

- **Development**: Local development and testing
- **Staging**: Pre-production validation
- **Production**: High-availability production deployment
- **Edge**: Resource-constrained edge devices

## 📋 Pre-Deployment Checklist

### System Requirements

**Minimum Requirements:**
- Python 3.9+
- 4GB RAM
- 10GB disk space
- Linux/macOS/Windows

**Recommended Production:**
- Python 3.11+
- 16GB RAM
- 100GB disk space (for caching)
- Linux (Ubuntu 20.04+ or RHEL 8+)

**Hardware-Specific Requirements:**
- **Intel Loihi3**: NX SDK 1.0+, appropriate drivers
- **NVIDIA**: CUDA 11.8+, cuDNN 8.6+
- **Edge Deployment**: ARM64 support, minimum 2GB RAM

### Security Configuration

1. **Environment Variables**
```bash
# Security settings
export SPIKE_SECURITY_MODE=true
export SPIKE_ALLOW_PICKLE=false
export SPIKE_REQUIRE_HW_VERIFICATION=true

# Resource limits
export SPIKE_MAX_MODEL_SIZE_MB=2000
export SPIKE_MAX_COMPILATION_TIME=7200
export SPIKE_MAX_MEMORY_GB=12

# Allowed paths (restrict model loading)
export SPIKE_ALLOWED_MODEL_PATHS="/opt/models,/data/trusted_models"
export SPIKE_ALLOWED_TARGETS="simulation,loihi3"
```

2. **File Permissions**
```bash
# Secure installation directory
chmod 755 /opt/spike-compiler
chown -R spike-user:spike-group /opt/spike-compiler

# Secure cache directory
mkdir -p /var/cache/spike-compiler
chmod 750 /var/cache/spike-compiler
chown spike-user:spike-group /var/cache/spike-compiler
```

## 🐳 Container Deployment

### Docker

**Dockerfile Example:**
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ cmake \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 spike-user

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/

# Install the package
RUN pip install -e .

# Create cache directory
RUN mkdir -p /app/cache && chown spike-user:spike-user /app/cache

# Switch to non-root user
USER spike-user

# Set environment variables
ENV SPIKE_CACHE_DIR=/app/cache
ENV SPIKE_LOG_LEVEL=INFO
ENV SPIKE_SECURITY_MODE=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD spike-compile info || exit 1

# Default command
CMD ["spike-compile", "--help"]
```

**Build and Run:**
```bash
# Build image
docker build -t spike-compiler:latest .

# Run container
docker run -it --rm \
    -v /path/to/models:/app/models:ro \
    -v /path/to/cache:/app/cache \
    -e SPIKE_LOG_LEVEL=INFO \
    spike-compiler:latest \
    spike-compile compile /app/models/model.pth --input-shape 1,3,224,224
```

### Docker Compose

```yaml
version: '3.8'

services:
  spike-compiler:
    build: .
    environment:
      - SPIKE_LOG_LEVEL=INFO
      - SPIKE_SECURITY_MODE=true
      - SPIKE_CACHE_DIR=/app/cache
      - SPIKE_MAX_MEMORY_GB=8
    volumes:
      - ./models:/app/models:ro
      - spike_cache:/app/cache
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          memory: 12G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
    restart: unless-stopped

  # Optional: Redis for distributed caching
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  spike_cache:
  redis_data:
```

## ☸️ Kubernetes Deployment

### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: spike-compiler-config
data:
  SPIKE_LOG_LEVEL: "INFO"
  SPIKE_SECURITY_MODE: "true"
  SPIKE_CACHE_ENABLED: "true"
  SPIKE_MAX_MEMORY_GB: "8"
```

### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spike-compiler
  labels:
    app: spike-compiler
spec:
  replicas: 3
  selector:
    matchLabels:
      app: spike-compiler
  template:
    metadata:
      labels:
        app: spike-compiler
    spec:
      containers:
      - name: spike-compiler
        image: spike-compiler:latest
        ports:
        - containerPort: 8080
        envFrom:
        - configMapRef:
            name: spike-compiler-config
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
          limits:
            memory: "12Gi"
            cpu: "4"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
        - name: cache-storage
          mountPath: /app/cache
        livenessProbe:
          exec:
            command:
            - spike-compile
            - info
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - spike-compile
            - info
          initialDelaySeconds: 10
          periodSeconds: 10
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
      - name: cache-storage
        persistentVolumeClaim:
          claimName: cache-storage-pvc
```

## 📊 Monitoring and Observability

### Metrics Collection

**Prometheus Configuration:**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'spike-compiler'
    static_configs:
      - targets: ['spike-compiler:8080']
    metrics_path: /metrics
    scrape_interval: 30s
```

**Key Metrics to Monitor:**
- Compilation success rate
- Average compilation time
- Memory usage patterns
- Cache hit rates
- Error rates by target
- Queue depth (if using queuing system)

### Logging

**Structured Logging Configuration:**
```python
# logging_config.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/app/logs/spike-compiler.log',
            'maxBytes': 100 * 1024 * 1024,  # 100MB
            'backupCount': 10,
            'formatter': 'json',
            'level': 'INFO',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file'],
    },
}
```

### Alerting Rules

**Example Alerts:**
```yaml
# alerts.yml
groups:
- name: spike-compiler
  rules:
  - alert: HighCompilationFailureRate
    expr: (spike_compiler_compilation_failures / spike_compiler_compilation_total) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High compilation failure rate detected"
      
  - alert: LongCompilationTime
    expr: spike_compiler_compilation_duration_seconds > 1800
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Compilation taking too long"
      
  - alert: HighMemoryUsage
    expr: spike_compiler_memory_usage_bytes > 10737418240  # 10GB
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
```

## 🔧 Configuration Management

### Production Configuration Template

```yaml
# spike_compiler_production.yaml
compiler:
  default_target: "loihi3"
  default_optimization_level: 2
  default_time_steps: 4
  log_level: "INFO"
  metrics_enabled: true
  security_mode: true
  max_memory_usage_gb: 10.0
  max_compilation_time_seconds: 3600
  compilation_cache_enabled: true
  cache_directory: "/var/cache/spike-compiler"
  
  optimization_passes:
    dead_code_elimination: true
    common_subexpression_elimination: true
    spike_fusion: true
    memory_optimization: true
    temporal_fusion: true

targets:
  simulation:
    enabled: true
    priority: 1
    max_neurons: 1000000
    
  loihi3:
    enabled: true
    priority: 2
    max_neurons: 1024000
    cores_per_chip: 128
    energy_model: "loihi3"
```

## 🔒 Security Hardening

### Network Security
- Use TLS 1.3 for all communications
- Implement API rate limiting
- Use mutual TLS for service-to-service communication
- Network segmentation for hardware access

### Access Control
```bash
# Create dedicated service account
useradd -r -s /bin/false spike-compiler

# Set up sudo rules for hardware access (if needed)
echo "spike-compiler ALL=(root) NOPASSWD: /usr/local/bin/nxsdk-init" >> /etc/sudoers.d/spike-compiler
```

### File System Security
```bash
# Secure model storage
mkdir -p /opt/models
chmod 755 /opt/models
chown root:spike-compiler /opt/models

# Secure temporary directories
mkdir -p /tmp/spike-compiler
chmod 1777 /tmp/spike-compiler
chown spike-compiler:spike-compiler /tmp/spike-compiler
```

## 🚦 Load Balancing

### HAProxy Configuration
```
# haproxy.cfg
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5s
    timeout client 30s
    timeout server 30s

backend spike-compiler
    balance roundrobin
    server compiler1 spike-compiler-1:8080 check
    server compiler2 spike-compiler-2:8080 check
    server compiler3 spike-compiler-3:8080 check

frontend spike-compiler-frontend
    bind *:80
    default_backend spike-compiler
```

### NGINX Configuration
```nginx
upstream spike_compiler {
    server spike-compiler-1:8080;
    server spike-compiler-2:8080;
    server spike-compiler-3:8080;
}

server {
    listen 80;
    server_name spike-compiler.example.com;
    
    location / {
        proxy_pass http://spike_compiler;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_timeout 1800s;
    }
}
```

## 📈 Scaling Strategies

### Horizontal Scaling
- Deploy multiple compiler instances behind load balancer
- Use message queues (Redis/RabbitMQ) for work distribution
- Implement circuit breakers for fault tolerance

### Vertical Scaling
- Optimize memory allocation for large models
- Use GPU acceleration where available
- Implement model sharding for very large models

### Auto-scaling
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: spike-compiler-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: spike-compiler
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Run tests
      run: |
        pip install -e ".[dev]"
        pytest tests/ --cov=spike_transformer_compiler --cov-report=xml
    - name: Security scan
      run: |
        pip install bandit safety
        bandit -r src/
        safety check
        
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build Docker image
      run: |
        docker build -t spike-compiler:${{ github.sha }} .
        docker tag spike-compiler:${{ github.sha }} spike-compiler:latest
    - name: Push to registry
      run: |
        echo ${{ secrets.REGISTRY_PASSWORD }} | docker login -u ${{ secrets.REGISTRY_USERNAME }} --password-stdin
        docker push spike-compiler:${{ github.sha }}
        docker push spike-compiler:latest
        
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/')
    steps:
    - name: Deploy to production
      run: |
        kubectl set image deployment/spike-compiler spike-compiler=spike-compiler:${{ github.sha }}
        kubectl rollout status deployment/spike-compiler
```

## 🆘 Disaster Recovery

### Backup Strategy
- **Configuration**: Version controlled in Git
- **Models**: Backup to S3/Azure Blob Storage
- **Cache**: Can be regenerated, but backup for performance
- **Logs**: Centralized logging with retention policy

### Recovery Procedures
1. **Service Recovery**: Automated through orchestrator (K8s/Docker Swarm)
2. **Data Recovery**: Restore from backups with RTO < 4 hours
3. **Configuration Recovery**: Deploy from Git with infrastructure as code

## 📞 Support and Maintenance

### Health Checks
```python
# health_check.py
def health_check():
    checks = {
        'compiler': check_compiler_status(),
        'cache': check_cache_status(),
        'memory': check_memory_usage(),
        'disk': check_disk_space(),
        'hardware': check_hardware_availability(),
    }
    
    all_healthy = all(checks.values())
    return {'status': 'healthy' if all_healthy else 'unhealthy', 'details': checks}
```

### Maintenance Windows
- **Weekly**: Log rotation, cache cleanup
- **Monthly**: Security updates, dependency updates  
- **Quarterly**: Full system review, performance tuning

### Emergency Contacts
- **Primary**: DevOps team (ops@company.com)
- **Secondary**: ML Engineering team (ml-eng@company.com)
- **Escalation**: CTO (cto@company.com)

## 📋 Deployment Checklist

**Pre-Deployment:**
- [ ] Security scan passed
- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] Configuration reviewed
- [ ] Backup strategy verified
- [ ] Monitoring configured
- [ ] Alerting rules tested

**Deployment:**
- [ ] Blue-green deployment executed
- [ ] Health checks passing
- [ ] Smoke tests completed
- [ ] Performance monitoring active
- [ ] Rollback plan ready

**Post-Deployment:**
- [ ] Service stability confirmed
- [ ] Error rates within SLA
- [ ] Performance metrics normal
- [ ] User acceptance testing
- [ ] Documentation updated

---

For questions or issues, contact the development team or refer to the troubleshooting guide in the main documentation.