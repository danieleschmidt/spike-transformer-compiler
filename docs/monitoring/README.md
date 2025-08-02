# Monitoring and Observability

## Overview

This directory contains monitoring and observability configuration for the Spike-Transformer-Compiler. The system uses Prometheus for metrics collection, Grafana for visualization, and structured logging for debugging.

## Components

### Prometheus Configuration
- `../monitoring/prometheus.yml`: Metrics collection configuration
- `../monitoring/alerts.yml`: Alerting rules for critical conditions
- Health checks for compilation processes and hardware utilization

### Grafana Dashboards
- `../monitoring/grafana/dashboards/spike-compiler-dashboard.json`: Main monitoring dashboard
- Visualization of compilation metrics, energy consumption, and performance

### Health Checks

The application exposes health check endpoints:

```python
from spike_transformer_compiler.monitoring import HealthChecker

# Basic health check
health = HealthChecker()
status = health.check_all()
print(f"System status: {status.overall}")

# Hardware health
hw_status = health.check_hardware()
print(f"Loihi 3 status: {hw_status.loihi3}")
```

### Metrics Collection

Key metrics monitored:
- Compilation time and success rate
- Memory usage during compilation
- Energy consumption estimates
- Hardware utilization (when available)
- Spike compression ratios
- Model accuracy after compilation

### Structured Logging

Logging configuration follows structured logging best practices:

```python
import logging
from spike_transformer_compiler.monitoring import setup_logging

# Setup structured logging
setup_logging(
    level=logging.INFO,
    format="json",  # or "text" for development
    include_metrics=True
)

logger = logging.getLogger(__name__)
logger.info(
    "Compilation started",
    extra={
        "model_name": "spikeformer_base",
        "target_hardware": "loihi3",
        "optimization_level": 3
    }
)
```

## Setup Instructions

### Local Development

1. Start monitoring stack:
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

2. Access Grafana dashboard:
- URL: http://localhost:3000
- Username: admin
- Password: admin (change on first login)

3. View Prometheus metrics:
- URL: http://localhost:9090

### Production Deployment

1. Configure external monitoring endpoints in `prometheus.yml`
2. Set up alerting destinations in `alerts.yml`
3. Configure log aggregation (ELK stack or similar)
4. Set up monitoring data retention policies

## Alerting Rules

Critical alerts configured:
- Compilation failure rate > 5%
- Memory usage > 90%
- Hardware communication errors
- Energy consumption anomalies

## Troubleshooting

Common monitoring issues:

1. **Metrics not appearing**: Check application `/metrics` endpoint
2. **Dashboard empty**: Verify Prometheus data source configuration
3. **Alerts not firing**: Check Prometheus rule evaluation logs
4. **High memory usage**: Monitor compilation pipeline memory leaks

## Custom Metrics

Add custom metrics to your application:

```python
from prometheus_client import Counter, Histogram, Gauge

# Custom metrics
COMPILATION_COUNTER = Counter(
    'spike_compiler_compilations_total',
    'Total number of model compilations',
    ['model_type', 'target_hardware']
)

COMPILATION_DURATION = Histogram(
    'spike_compiler_compilation_duration_seconds',
    'Time spent compiling models'
)

ENERGY_ESTIMATE = Gauge(
    'spike_compiler_energy_estimate_mj',
    'Estimated energy consumption per inference'
)

# Usage in compilation code
with COMPILATION_DURATION.time():
    compiled_model = compiler.compile(model)
    COMPILATION_COUNTER.labels(
        model_type=model.__class__.__name__,
        target_hardware="loihi3"
    ).inc()
    ENERGY_ESTIMATE.set(compiled_model.estimated_energy)
```
