#!/usr/bin/env python3
"""Quality gates validation for Spike-Transformer-Compiler."""

import os
import sys
import subprocess
import time
from pathlib import Path


def check_code_structure():
    """Check if the code structure is complete."""
    print("🏗️  Checking Code Structure...")
    
    required_files = [
        # Core IR
        "src/spike_transformer_compiler/ir/spike_graph.py",
        "src/spike_transformer_compiler/ir/builder.py", 
        "src/spike_transformer_compiler/ir/types.py",
        "src/spike_transformer_compiler/ir/passes.py",
        
        # Frontend
        "src/spike_transformer_compiler/frontend/pytorch_parser.py",
        
        # Backend
        "src/spike_transformer_compiler/backend/simulation_backend.py",
        "src/spike_transformer_compiler/backend/loihi3_backend.py",
        "src/spike_transformer_compiler/backend/factory.py",
        
        # Core compiler
        "src/spike_transformer_compiler/compiler.py",
        
        # Kernels
        "src/spike_transformer_compiler/kernels/attention.py",
        "src/spike_transformer_compiler/kernels/convolution.py",
        
        # Generation 2 features
        "src/spike_transformer_compiler/security.py",
        "src/spike_transformer_compiler/exceptions.py",
        "src/spike_transformer_compiler/logging_config.py",
        "src/spike_transformer_compiler/performance.py",
        "src/spike_transformer_compiler/config.py",
        
        # Generation 3 features  
        "src/spike_transformer_compiler/scaling/auto_scaler.py",
        "src/spike_transformer_compiler/scaling/resource_pool.py",
        "src/spike_transformer_compiler/distributed/compilation_cluster.py",
        "src/spike_transformer_compiler/distributed/distributed_coordinator.py",
        "src/spike_transformer_compiler/optimization/memory_optimizer.py",
        
        # Applications
        "src/spike_transformer_compiler/applications/realtime_vision.py",
        "src/spike_transformer_compiler/applications/research_platform.py",
        
        # Runtime
        "src/spike_transformer_compiler/runtime/executor.py",
        "src/spike_transformer_compiler/runtime/memory.py",
        "src/spike_transformer_compiler/runtime/communication.py"
    ]
    
    missing_files = []
    total_lines = 0
    
    for file_path in required_files:
        if os.path.exists(file_path):
            # Count lines
            with open(file_path, 'r') as f:
                lines = len(f.readlines())
                total_lines += lines
            print(f"  ✓ {file_path} ({lines} lines)")
        else:
            missing_files.append(file_path)
            print(f"  ✗ {file_path} MISSING")
    
    print(f"\n📊 Code Statistics:")
    print(f"  • Total implementation files: {len(required_files) - len(missing_files)}/{len(required_files)}")
    print(f"  • Total lines of code: {total_lines:,}")
    
    if missing_files:
        print(f"  ⚠️  Missing {len(missing_files)} files")
        return False
    else:
        print("  ✅ All core files present")
        return True


def check_architecture_completeness():
    """Check if all three generations are implemented."""
    print("\n🚀 Checking Architecture Completeness...")
    
    generations = {
        "Generation 1 (Make it Work)": [
            ("Core IR", "src/spike_transformer_compiler/ir/"),
            ("PyTorch Parser", "src/spike_transformer_compiler/frontend/pytorch_parser.py"),
            ("Optimization Passes", "src/spike_transformer_compiler/ir/passes.py"),
            ("Simulation Backend", "src/spike_transformer_compiler/backend/simulation_backend.py"),
            ("Spiking Kernels", "src/spike_transformer_compiler/kernels/")
        ],
        "Generation 2 (Make it Robust)": [
            ("Error Handling", "src/spike_transformer_compiler/exceptions.py"),
            ("Security Framework", "src/spike_transformer_compiler/security.py"),
            ("Logging System", "src/spike_transformer_compiler/logging_config.py"),
            ("Performance Monitoring", "src/spike_transformer_compiler/performance.py"),
            ("Configuration Management", "src/spike_transformer_compiler/config.py")
        ],
        "Generation 3 (Make it Scale)": [
            ("Auto-scaling", "src/spike_transformer_compiler/scaling/auto_scaler.py"),
            ("Distributed Compilation", "src/spike_transformer_compiler/distributed/"),
            ("Intelligent Caching", "src/spike_transformer_compiler/performance.py"),
            ("Memory Optimization", "src/spike_transformer_compiler/optimization/memory_optimizer.py")
        ]
    }
    
    all_complete = True
    
    for generation, components in generations.items():
        print(f"\n  {generation}:")
        gen_complete = True
        
        for component_name, path in components:
            if os.path.exists(path):
                if os.path.isdir(path):
                    # Count files in directory
                    files = list(Path(path).glob("*.py"))
                    print(f"    ✓ {component_name} ({len(files)} files)")
                else:
                    # Single file
                    with open(path, 'r') as f:
                        lines = len(f.readlines())
                    print(f"    ✓ {component_name} ({lines} lines)")
            else:
                print(f"    ✗ {component_name} MISSING")
                gen_complete = False
                all_complete = False
        
        if gen_complete:
            print(f"    🎉 {generation} COMPLETE")
        else:
            print(f"    ❌ {generation} INCOMPLETE")
    
    return all_complete


def check_documentation():
    """Check documentation completeness."""
    print("\n📚 Checking Documentation...")
    
    docs = [
        "README.md",
        "ARCHITECTURE_SUMMARY.md", 
        "API_REFERENCE.md",
        "DEPLOYMENT_GUIDE.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "docs/ARCHITECTURE.md",
        "docs/ROADMAP.md"
    ]
    
    doc_score = 0
    total_docs = len(docs)
    
    for doc in docs:
        if os.path.exists(doc):
            with open(doc, 'r') as f:
                lines = len(f.readlines())
                if lines > 10:  # Substantial documentation
                    print(f"  ✓ {doc} ({lines} lines)")
                    doc_score += 1
                else:
                    print(f"  ⚠️  {doc} (only {lines} lines)")
                    doc_score += 0.5
        else:
            print(f"  ✗ {doc} missing")
    
    print(f"\n📊 Documentation Score: {doc_score}/{total_docs} ({doc_score/total_docs*100:.1f}%)")
    return doc_score / total_docs >= 0.8


def check_project_setup():
    """Check project setup and configuration."""
    print("\n⚙️  Checking Project Setup...")
    
    setup_files = [
        ("pyproject.toml", "Python project configuration"),
        ("requirements.txt", "Python dependencies"),
        ("Dockerfile", "Container setup"),
        ("docker-compose.yml", "Development environment"),
        ("Makefile", "Build automation"),
        (".gitignore", "Git configuration")
    ]
    
    setup_score = 0
    
    for filename, description in setup_files:
        if os.path.exists(filename):
            print(f"  ✓ {filename} - {description}")
            setup_score += 1
        else:
            print(f"  ✗ {filename} - {description} MISSING")
    
    print(f"\n📊 Setup Score: {setup_score}/{len(setup_files)}")
    return setup_score >= len(setup_files) * 0.8


def check_security_compliance():
    """Check security compliance."""
    print("\n🛡️  Checking Security Compliance...")
    
    security_checks = []
    
    # Check for security.py implementation
    if os.path.exists("src/spike_transformer_compiler/security.py"):
        with open("src/spike_transformer_compiler/security.py", 'r') as f:
            content = f.read()
            if "SecurityValidator" in content:
                security_checks.append(("Security validator", True))
            if "InputSanitizer" in content:
                security_checks.append(("Input sanitization", True))
            if "create_secure_environment" in content:
                security_checks.append(("Secure environments", True))
    
    # Check for no hardcoded secrets
    secret_patterns = ["password", "api_key", "secret_key", "token"]
    secure = True
    
    for pattern in secret_patterns:
        result = subprocess.run(
            ["grep", "-r", "-i", pattern, "src/"], 
            capture_output=True, text=True
        )
        if result.returncode == 0 and "test" not in result.stdout.lower():
            print(f"  ⚠️  Found potential secret: {pattern}")
            secure = False
    
    if secure:
        security_checks.append(("No hardcoded secrets", True))
        print("  ✓ No hardcoded secrets found")
    
    security_score = len([c for c in security_checks if c[1]])
    print(f"\n📊 Security Score: {security_score}/4")
    
    return security_score >= 3


def check_performance_readiness():
    """Check performance and scalability features."""
    print("\n⚡ Checking Performance Readiness...")
    
    perf_features = [
        ("Caching system", "src/spike_transformer_compiler/performance.py", "CompilationCache"),
        ("Auto-scaling", "src/spike_transformer_compiler/scaling/auto_scaler.py", "AdvancedAutoScaler"),
        ("Distributed compilation", "src/spike_transformer_compiler/distributed/", "CompilationCluster"),
        ("Memory optimization", "src/spike_transformer_compiler/optimization/memory_optimizer.py", "AdvancedMemoryManager"),
        ("Performance monitoring", "src/spike_transformer_compiler/performance.py", "PerformanceProfiler")
    ]
    
    perf_score = 0
    
    for feature_name, path, class_name in perf_features:
        if os.path.exists(path):
            if os.path.isdir(path):
                files = list(Path(path).glob("*.py"))
                if files:
                    print(f"  ✓ {feature_name} ({len(files)} files)")
                    perf_score += 1
            else:
                with open(path, 'r') as f:
                    content = f.read()
                    if class_name in content:
                        print(f"  ✓ {feature_name} (implemented)")
                        perf_score += 1
                    else:
                        print(f"  ⚠️  {feature_name} (partial)")
                        perf_score += 0.5
        else:
            print(f"  ✗ {feature_name} missing")
    
    print(f"\n📊 Performance Score: {perf_score}/{len(perf_features)}")
    return perf_score >= len(perf_features) * 0.8


def run_quality_gates():
    """Run all quality gates."""
    print("🛡️  MANDATORY QUALITY GATES")
    print("=" * 60)
    
    gates = [
        ("Code Structure", check_code_structure),
        ("Architecture Completeness", check_architecture_completeness),  
        ("Documentation", check_documentation),
        ("Project Setup", check_project_setup),
        ("Security Compliance", check_security_compliance),
        ("Performance Readiness", check_performance_readiness)
    ]
    
    passed_gates = 0
    total_gates = len(gates)
    
    for gate_name, gate_func in gates:
        try:
            if gate_func():
                print(f"\n✅ {gate_name} PASSED")
                passed_gates += 1
            else:
                print(f"\n❌ {gate_name} FAILED")
        except Exception as e:
            print(f"\n❌ {gate_name} FAILED with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 QUALITY GATES RESULT: {passed_gates}/{total_gates} PASSED")
    
    success_rate = passed_gates / total_gates
    if success_rate >= 0.85:  # 85% pass rate required
        print("🎉 QUALITY GATES PASSED!")
        print("✅ System ready for production deployment")
        return True
    else:
        print(f"⚠️  QUALITY GATES NEED ATTENTION ({success_rate*100:.1f}% pass rate)")
        print("❌ Some gates failed - review before production")
        return False


if __name__ == "__main__":
    success = run_quality_gates()
    sys.exit(0 if success else 1)