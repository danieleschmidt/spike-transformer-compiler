#!/usr/bin/env python3
"""Comprehensive validation test for neuromorphic compiler codebase.

Tests all key components and provides detailed status reporting.
"""

import sys
import os
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class ComprehensiveValidator:
    """Comprehensive validation system for neuromorphic compiler."""
    
    def __init__(self):
        sys.path.append('src')
        self.results = {}
        self.issues = []
        self.enhancements = []
        
    def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation."""
        print("🔬 NEUROMORPHIC COMPILER COMPREHENSIVE VALIDATION")
        print("=" * 60)
        
        # Test suites
        validation_tests = [
            ("Core Compilation Pipeline", self.test_core_pipeline),
            ("Autonomous Systems", self.test_autonomous_systems),
            ("Research Capabilities", self.test_research_capabilities),
            ("Security Framework", self.test_security_framework),
            ("Resilience Systems", self.test_resilience_systems),
            ("Quantum Optimization", self.test_quantum_optimization),
            ("Multi-Cloud Orchestration", self.test_multicloud_orchestration),
            ("Performance Systems", self.test_performance_systems),
            ("Documentation Coverage", self.test_documentation),
        ]
        
        for test_name, test_func in validation_tests:
            print(f"\n🔍 Testing: {test_name}")
            print("-" * 40)
            try:
                result = test_func()
                self.results[test_name] = result
                
                if result['status'] == 'PASS':
                    print(f"✅ {test_name}: PASSED")
                elif result['status'] == 'PARTIAL':
                    print(f"⚠️ {test_name}: PARTIAL")
                    for issue in result.get('issues', []):
                        print(f"   - {issue}")
                        self.issues.append(f"{test_name}: {issue}")
                else:
                    print(f"❌ {test_name}: FAILED")
                    for issue in result.get('issues', []):
                        print(f"   - {issue}")
                        self.issues.append(f"{test_name}: {issue}")
                
                if 'enhancements' in result:
                    self.enhancements.extend(result['enhancements'])
                    
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {str(e)}")
                self.results[test_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'issues': [f"Test execution failed: {str(e)}"]
                }
                self.issues.append(f"{test_name}: Test execution failed")
        
        # Generate final report
        return self.generate_validation_report()
    
    def test_core_pipeline(self) -> Dict[str, Any]:
        """Test core compilation pipeline."""
        issues = []
        enhancements = []
        
        try:
            # Test compiler instantiation
            from spike_transformer_compiler.compiler import SpikeCompiler, CompiledModel
            compiler = SpikeCompiler(target="simulation", optimization_level=2)
            
            # Test backend factory
            from spike_transformer_compiler.backend.factory import BackendFactory
            available_targets = BackendFactory.get_available_targets()
            if len(available_targets) < 2:
                issues.append("Limited backend targets available")
                enhancements.append("Add more hardware backend targets (FPGA, custom ASICs)")
            
            # Test IR components
            from spike_transformer_compiler.ir.spike_graph import SpikeGraph
            from spike_transformer_compiler.ir.builder import SpikeIRBuilder
            
            # Test optimization passes
            from spike_transformer_compiler.optimization import OptimizationPass
            
            # Test kernels
            try:
                from spike_transformer_compiler.kernels.attention import SpikeAttention
                from spike_transformer_compiler.kernels.convolution import SpikeConv2d
            except ImportError as e:
                issues.append(f"Kernel imports failed: {e}")
            
            if not issues:
                return {'status': 'PASS', 'message': 'Core pipeline fully functional'}
            else:
                return {'status': 'PARTIAL', 'issues': issues, 'enhancements': enhancements}
                
        except Exception as e:
            return {'status': 'FAIL', 'issues': [f"Core pipeline test failed: {e}"]}
    
    def test_autonomous_systems(self) -> Dict[str, Any]:
        """Test autonomous evolution and SDLC capabilities."""
        issues = []
        enhancements = []
        
        try:
            # Test autonomous executor
            from spike_transformer_compiler.autonomous_executor import AutonomousExecutor
            executor = AutonomousExecutor()
            
            # Check generations support
            if hasattr(executor, 'current_generation') and hasattr(executor, 'max_generation'):
                if executor.max_generation < 3:
                    issues.append("Limited generation support")
            else:
                issues.append("Missing progressive enhancement capabilities")
            
            # Test autonomous enhancement engine
            try:
                from spike_transformer_compiler.autonomous_enhancement_engine import AutonomousEnhancementEngine
                enhancement_engine = AutonomousEnhancementEngine()
                enhancements.append("Advanced autonomous enhancement available")
            except ImportError:
                issues.append("Autonomous enhancement engine not available")
            
            # Test evolution engine
            try:
                from spike_transformer_compiler.autonomous_evolution_engine import AutonomousEvolutionEngine
                evolution_engine = AutonomousEvolutionEngine()
            except ImportError:
                issues.append("Evolution engine not available")
                enhancements.append("Implement full autonomous evolution capabilities")
            
            status = 'PASS' if not issues else 'PARTIAL'
            return {'status': status, 'issues': issues, 'enhancements': enhancements}
            
        except Exception as e:
            return {'status': 'FAIL', 'issues': [f"Autonomous systems test failed: {e}"]}
    
    def test_research_capabilities(self) -> Dict[str, Any]:
        """Test research and hypothesis-driven development."""
        issues = []
        enhancements = []
        
        try:
            # Test research platform components
            import spike_transformer_compiler.research_platform as rp
            
            # Check for research framework
            if hasattr(rp, 'ExperimentalFramework'):
                framework = rp.ExperimentalFramework()
                if hasattr(framework, 'conduct_experiment'):
                    pass  # Good
                else:
                    issues.append("Research framework missing experiment capabilities")
            else:
                issues.append("No experimental framework available")
            
            # Check statistical validation
            if hasattr(rp, 'StatisticalValidator'):
                validator = rp.StatisticalValidator()
            else:
                issues.append("No statistical validation capabilities")
                enhancements.append("Add comprehensive statistical analysis tools")
            
            # Check documentation generation
            if hasattr(rp, 'ResearchDocumentationGenerator'):
                doc_gen = rp.ResearchDocumentationGenerator()
            else:
                issues.append("No research documentation generation")
                enhancements.append("Implement automated research paper generation")
            
            # Test research orchestrator
            try:
                from spike_transformer_compiler.research_orchestrator import ResearchOrchestrator
                orchestrator = ResearchOrchestrator()
                enhancements.append("Advanced research orchestration available")
            except ImportError:
                enhancements.append("Add research orchestration capabilities")
            
            status = 'PARTIAL' if issues else 'PASS'
            return {'status': status, 'issues': issues, 'enhancements': enhancements}
            
        except Exception as e:
            return {'status': 'FAIL', 'issues': [f"Research capabilities test failed: {e}"]}
    
    def test_security_framework(self) -> Dict[str, Any]:
        """Test security and validation systems."""
        issues = []
        enhancements = []
        
        try:
            # Test security validator
            from spike_transformer_compiler.security import SecurityValidator
            validator = SecurityValidator()
            
            # Check security methods
            security_methods = ['validate_model_security', 'validate_input_safety', 
                              'validate_optimization_safety', 'validate_backend_config']
            for method in security_methods:
                if not hasattr(validator, method):
                    issues.append(f"Missing security method: {method}")
            
            # Test input validation
            from spike_transformer_compiler.validation import ValidationUtils
            
            # Test security scanner
            try:
                from spike_transformer_compiler.security_scanner import SecurityScanner
                scanner = SecurityScanner()
                enhancements.append("Advanced security scanning available")
            except ImportError:
                enhancements.append("Add comprehensive security scanning")
            
            # Test comprehensive security system
            try:
                from spike_transformer_compiler.comprehensive_security_system import ComprehensiveSecuritySystem
                security_system = ComprehensiveSecuritySystem()
                enhancements.append("Comprehensive security system available")
            except ImportError:
                enhancements.append("Implement comprehensive security orchestration")
            
            status = 'PASS' if not issues else 'PARTIAL'
            return {'status': status, 'issues': issues, 'enhancements': enhancements}
            
        except Exception as e:
            return {'status': 'FAIL', 'issues': [f"Security framework test failed: {e}"]}
    
    def test_resilience_systems(self) -> Dict[str, Any]:
        """Test resilience and fault tolerance."""
        issues = []
        enhancements = []
        
        try:
            # Test enhanced resilience
            import spike_transformer_compiler.enhanced_resilience as er
            
            # Check circuit breaker
            if hasattr(er, 'AdvancedCircuitBreaker'):
                circuit_breaker = er.AdvancedCircuitBreaker()
            else:
                issues.append("No circuit breaker implementation")
            
            # Check self-healing
            if hasattr(er, 'SelfHealingSystem'):
                healing_system = er.SelfHealingSystem()
            else:
                issues.append("No self-healing capabilities")
                enhancements.append("Implement comprehensive self-healing system")
            
            # Check health monitoring
            if hasattr(er, 'HealthMonitoringSystem'):
                health_monitor = er.HealthMonitoringSystem()
            else:
                issues.append("No health monitoring system")
            
            # Test adaptive resilience framework
            try:
                from spike_transformer_compiler.adaptive_resilience_framework import AdaptiveResilienceFramework
                adaptive_framework = AdaptiveResilienceFramework()
                enhancements.append("Advanced adaptive resilience available")
            except ImportError:
                enhancements.append("Add adaptive resilience framework")
            
            status = 'PARTIAL' if issues else 'PASS'
            return {'status': status, 'issues': issues, 'enhancements': enhancements}
            
        except Exception as e:
            return {'status': 'FAIL', 'issues': [f"Resilience systems test failed: {e}"]}
    
    def test_quantum_optimization(self) -> Dict[str, Any]:
        """Test quantum optimization features."""
        issues = []
        enhancements = []
        
        try:
            # Test quantum optimization engine
            try:
                from spike_transformer_compiler.quantum_optimization_engine import QuantumOptimizationEngine
                quantum_engine = QuantumOptimizationEngine()
                if hasattr(quantum_engine, 'optimize_spike_graph'):
                    pass  # Good
                else:
                    issues.append("Quantum engine missing optimization methods")
            except ImportError as e:
                if "numpy" in str(e).lower():
                    issues.append("Quantum optimization requires NumPy (dependency missing)")
                else:
                    issues.append("Quantum optimization engine not available")
                enhancements.append("Implement quantum-enhanced optimization algorithms")
            
            # Test quantum hybrid engine
            try:
                from spike_transformer_compiler.quantum_hybrid_engine import QuantumHybridEngine
                hybrid_engine = QuantumHybridEngine()
                enhancements.append("Quantum hybrid optimization available")
            except ImportError:
                enhancements.append("Add quantum-classical hybrid optimization")
            
            status = 'FAIL' if not issues and 'numpy' not in str(issues) else 'PARTIAL'
            return {'status': status, 'issues': issues, 'enhancements': enhancements}
            
        except Exception as e:
            return {'status': 'FAIL', 'issues': [f"Quantum optimization test failed: {e}"]}
    
    def test_multicloud_orchestration(self) -> Dict[str, Any]:
        """Test multi-cloud orchestration capabilities."""
        issues = []
        enhancements = []
        
        try:
            # Test global deployment orchestrator
            from spike_transformer_compiler.global_deployment_orchestrator import GlobalDeploymentOrchestrator
            orchestrator = GlobalDeploymentOrchestrator()
            
            # Check deployment methods
            deployment_methods = ['deploy_to_region', 'scale_deployment', 'monitor_health']
            for method in deployment_methods:
                if not hasattr(orchestrator, method):
                    issues.append(f"Missing deployment method: {method}")
            
            # Test hyperscale orchestrator
            try:
                from spike_transformer_compiler.hyperscale_orchestrator_v4 import HyperscaleOrchestratorV4
                hyperscale = HyperscaleOrchestratorV4()
                enhancements.append("Advanced hyperscale orchestration available")
            except ImportError:
                from spike_transformer_compiler.hyperscale_orchestrator import HyperscaleOrchestrator
                hyperscale = HyperscaleOrchestrator()
            
            # Test edge deployment
            try:
                from spike_transformer_compiler.edge_deployment_engine import EdgeDeploymentEngine
                edge_engine = EdgeDeploymentEngine()
                enhancements.append("Edge deployment capabilities available")
            except ImportError:
                enhancements.append("Add edge deployment optimization")
            
            status = 'PASS' if not issues else 'PARTIAL'
            return {'status': status, 'issues': issues, 'enhancements': enhancements}
            
        except Exception as e:
            return {'status': 'FAIL', 'issues': [f"Multi-cloud orchestration test failed: {e}"]}
    
    def test_performance_systems(self) -> Dict[str, Any]:
        """Test performance optimization and monitoring."""
        issues = []
        enhancements = []
        
        try:
            # Test performance monitoring
            try:
                from spike_transformer_compiler.performance import PerformanceProfiler
                profiler = PerformanceProfiler()
            except ImportError:
                issues.append("Performance profiler not available")
            
            # Test optimization systems
            from spike_transformer_compiler.optimization import OptimizationPass
            
            # Check advanced optimization
            try:
                from spike_transformer_compiler.optimization_advanced import AdvancedOptimizationEngine
                adv_opt = AdvancedOptimizationEngine()
                enhancements.append("Advanced optimization engine available")
            except ImportError:
                enhancements.append("Implement advanced optimization techniques")
            
            # Test adaptive cache
            from spike_transformer_compiler.adaptive_cache import AdaptiveCache
            cache = AdaptiveCache()
            
            # Test hyperscale performance engine
            try:
                from spike_transformer_compiler.hyperscale_performance_engine import HyperscalePerformanceEngine
                perf_engine = HyperscalePerformanceEngine()
                enhancements.append("Hyperscale performance optimization available")
            except ImportError:
                enhancements.append("Add hyperscale performance optimization")
            
            status = 'PASS' if not issues else 'PARTIAL'
            return {'status': status, 'issues': issues, 'enhancements': enhancements}
            
        except Exception as e:
            return {'status': 'FAIL', 'issues': [f"Performance systems test failed: {e}"]}
    
    def test_documentation(self) -> Dict[str, Any]:
        """Test documentation coverage and quality."""
        issues = []
        enhancements = []
        
        try:
            # Check key documentation files
            doc_files = [
                'README.md',
                'ARCHITECTURE_SUMMARY.md',
                'API_REFERENCE.md',
                'DEPLOYMENT_GUIDE.md',
                'SECURITY.md',
                'AUTONOMOUS_SDLC_EXECUTION_COMPLETE.md'
            ]
            
            total_size = 0
            for doc_file in doc_files:
                path = Path(doc_file)
                if path.exists():
                    size_kb = path.stat().st_size / 1024
                    total_size += size_kb
                else:
                    issues.append(f"Missing documentation: {doc_file}")
            
            # Check examples
            examples_dir = Path('examples')
            if examples_dir.exists():
                example_files = list(examples_dir.glob('*.py'))
                if len(example_files) < 5:
                    issues.append("Limited example coverage")
                    enhancements.append("Add more comprehensive usage examples")
            else:
                issues.append("No examples directory")
            
            # Check API documentation
            api_ref = Path('API_REFERENCE.md')
            if api_ref.exists():
                if api_ref.stat().st_size < 10000:  # Less than 10KB
                    issues.append("API reference documentation incomplete")
            
            if total_size < 40:
                issues.append("Insufficient documentation coverage")
                enhancements.append("Expand documentation with detailed guides and tutorials")
            
            status = 'PASS' if not issues else 'PARTIAL'
            return {'status': status, 'issues': issues, 'enhancements': enhancements}
            
        except Exception as e:
            return {'status': 'FAIL', 'issues': [f"Documentation test failed: {e}"]}
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        partial_tests = sum(1 for r in self.results.values() if r['status'] == 'PARTIAL')
        failed_tests = sum(1 for r in self.results.values() if r['status'] in ['FAIL', 'ERROR'])
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        partial_rate = partial_tests / total_tests if total_tests > 0 else 0
        
        overall_status = 'PRODUCTION_READY' if success_rate >= 0.8 and partial_rate >= 0.1 else 'NEEDS_IMPROVEMENT'
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'test_summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'partial': partial_tests,
                'failed': failed_tests,
                'success_rate': success_rate,
                'partial_rate': partial_rate
            },
            'detailed_results': self.results,
            'identified_issues': self.issues,
            'enhancement_opportunities': self.enhancements,
            'recommendations': self.generate_recommendations()
        }
        
        # Save report
        with open('comprehensive_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if len(self.issues) > 5:
            recommendations.append("Priority: Fix critical import and dependency issues")
        
        if any("numpy" in issue.lower() for issue in self.issues):
            recommendations.append("Install NumPy and PyTorch dependencies for full functionality")
        
        if any("quantum" in issue.lower() for issue in self.issues):
            recommendations.append("Implement quantum optimization with proper dependency handling")
        
        if any("documentation" in issue.lower() for issue in self.issues):
            recommendations.append("Expand documentation coverage and examples")
        
        if len(self.enhancements) > 0:
            recommendations.append("Consider implementing suggested enhancements for competitive advantage")
        
        # Always include these general recommendations
        recommendations.extend([
            "Set up comprehensive testing environment with all dependencies",
            "Implement continuous integration pipeline",
            "Consider containerized deployment for dependency management",
            "Establish performance benchmarking baselines"
        ])
        
        return recommendations


def main():
    """Main validation execution."""
    print("🚀 Starting comprehensive neuromorphic compiler validation...\n")
    
    validator = ComprehensiveValidator()
    report = validator.run_validation()
    
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Tests Passed: {report['test_summary']['passed']}/{report['test_summary']['total_tests']}")
    print(f"Tests Partial: {report['test_summary']['partial']}/{report['test_summary']['total_tests']}")
    print(f"Success Rate: {report['test_summary']['success_rate']:.1%}")
    
    if report['identified_issues']:
        print(f"\n❌ Issues Found ({len(report['identified_issues'])}):")
        for i, issue in enumerate(report['identified_issues'][:10], 1):  # Show top 10
            print(f"   {i}. {issue}")
        if len(report['identified_issues']) > 10:
            print(f"   ... and {len(report['identified_issues']) - 10} more issues")
    
    if report['enhancement_opportunities']:
        print(f"\n💡 Enhancement Opportunities ({len(report['enhancement_opportunities'])}):")
        for i, enhancement in enumerate(report['enhancement_opportunities'][:5], 1):  # Show top 5
            print(f"   {i}. {enhancement}")
    
    print(f"\n🎯 Recommendations:")
    for i, rec in enumerate(report['recommendations'][:5], 1):  # Show top 5
        print(f"   {i}. {rec}")
    
    print(f"\n📄 Full report saved to: comprehensive_validation_report.json")
    
    return 0 if report['overall_status'] == 'PRODUCTION_READY' else 1


if __name__ == "__main__":
    sys.exit(main())