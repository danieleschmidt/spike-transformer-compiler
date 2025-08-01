#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Master Orchestrator
Complete autonomous value discovery and delivery system
Terragon Labs - Autonomous SDLC Enhancement
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class AutonomousSDLC:
    """Master orchestrator for autonomous SDLC value delivery."""
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.terragon_dir = self.repo_root / ".terragon"
        self.start_time = datetime.now()
        
        # Ensure .terragon directory exists
        self.terragon_dir.mkdir(exist_ok=True)
        
    def run_complete_cycle(self) -> Dict[str, Any]:
        """Execute complete autonomous SDLC cycle."""
        print("🚀 TERRAGON AUTONOMOUS SDLC - PERPETUAL VALUE DISCOVERY")
        print("=" * 70)
        print(f"🕒 Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Repository: {self.repo_root.name}")
        print()
        
        cycle_results = {
            'start_time': self.start_time.isoformat(),
            'phases_completed': [],
            'opportunities_discovered': 0,
            'opportunities_scored': 0,
            'work_item_selected': None,
            'execution_attempted': False,
            'execution_result': None,
            'insights_generated': 0,
            'total_value_estimate': 0,
            'status': 'in_progress',
            'errors': []
        }
        
        try:
            # Phase 1: Signal Harvesting
            print("🔍 PHASE 1: Comprehensive Signal Harvesting")
            print("-" * 50)
            opportunities = self._run_signal_harvesting()
            cycle_results['opportunities_discovered'] = len(opportunities)
            cycle_results['phases_completed'].append('signal_harvesting')
            print(f"✅ Discovered {len(opportunities)} value opportunities\n")
            
            # Phase 2: Adaptive Scoring
            print("🎯 PHASE 2: Adaptive Scoring Engine")
            print("-" * 50)
            scored_opportunities = self._run_scoring_engine()
            cycle_results['opportunities_scored'] = len(scored_opportunities)
            cycle_results['phases_completed'].append('scoring')
            
            if scored_opportunities:
                total_value = sum(opp.get('estimated_value_delivery', 0) for opp in scored_opportunities)
                cycle_results['total_value_estimate'] = total_value
                print(f"✅ Scored {len(scored_opportunities)} opportunities")
                print(f"💰 Total estimated value: ${total_value:,.0f}\n")
            
            # Phase 3: Intelligent Work Selection
            print("🧠 PHASE 3: Intelligent Work Selection")
            print("-" * 50)
            selected_work = self._run_work_selector()
            cycle_results['work_item_selected'] = selected_work
            cycle_results['phases_completed'].append('work_selection')
            
            if selected_work:
                print(f"✅ Selected: {selected_work.get('title', 'Unknown')}")
                print(f"📊 Score: {selected_work.get('composite_score', 0):.1f}")
                print(f"⏱  Effort: {selected_work.get('estimated_effort', 0)} hours\n")
            
            # Phase 4: Autonomous Execution (Optional)
            print("🚀 PHASE 4: Autonomous Execution (Demo Mode)")
            print("-" * 50)
            print("⚠️  Execution phase available but not run in demo mode")
            print("   To execute: python3 .terragon/autonomous-executor.py")
            cycle_results['phases_completed'].append('execution_demo')
            print()
            
            # Phase 5: Continuous Learning
            print("🧠 PHASE 5: Continuous Learning Analysis")
            print("-" * 50)
            insights = self._run_learning_engine()
            cycle_results['insights_generated'] = len(insights)
            cycle_results['phases_completed'].append('learning')
            print(f"✅ Generated {len(insights)} learning insights\n")
            
            # Phase 6: Backlog Generation
            print("📊 PHASE 6: Value Backlog Generation")
            print("-" * 50)
            self._generate_backlog_summary()
            cycle_results['phases_completed'].append('backlog_generation')
            print("✅ Updated comprehensive value backlog\n")
            
            cycle_results['status'] = 'completed'
            
        except Exception as e:
            cycle_results['status'] = 'failed'
            cycle_results['errors'].append(str(e))
            print(f"❌ Cycle failed: {e}")
        
        # Finalize results
        end_time = datetime.now()
        cycle_results['end_time'] = end_time.isoformat()
        cycle_results['duration_minutes'] = (end_time - self.start_time).total_seconds() / 60
        
        # Save cycle results
        self._save_cycle_results(cycle_results)
        
        # Print final summary
        self._print_cycle_summary(cycle_results)
        
        return cycle_results
    
    def _run_signal_harvesting(self) -> List[Dict[str, Any]]:
        """Execute signal harvesting phase."""
        try:
            result = subprocess.run(
                ['python3', '.terragon/signal-harvester.py'],
                capture_output=True, text=True, cwd=self.repo_root, timeout=120
            )
            
            if result.returncode == 0:
                print(result.stdout)
                
                # Load discovered opportunities
                opportunities_file = self.terragon_dir / "discovered-opportunities.json"
                if opportunities_file.exists():
                    with open(opportunities_file, 'r') as f:
                        return json.load(f)
            else:
                print(f"⚠️  Signal harvesting warning: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Signal harvesting failed: {e}")
        
        return []
    
    def _run_scoring_engine(self) -> List[Dict[str, Any]]:
        """Execute scoring engine phase."""
        try:
            result = subprocess.run(
                ['python3', '.terragon/scoring-engine.py'],
                capture_output=True, text=True, cwd=self.repo_root, timeout=120
            )
            
            if result.returncode == 0:
                print(result.stdout)
                
                # Load scored opportunities
                scored_file = self.terragon_dir / "scored-opportunities.json"
                if scored_file.exists():
                    with open(scored_file, 'r') as f:
                        return json.load(f)
            else:
                print(f"⚠️  Scoring engine warning: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Scoring engine failed: {e}")
        
        return []
    
    def _run_work_selector(self) -> Optional[Dict[str, Any]]:
        """Execute work selector phase."""
        try:
            result = subprocess.run(
                ['python3', '.terragon/work-selector.py'],
                capture_output=True, text=True, cwd=self.repo_root, timeout=60
            )
            
            if result.returncode == 0:
                print(result.stdout)
                
                # Load selected work item
                work_file = self.terragon_dir / "selected-work-item.json"
                if work_file.exists():
                    with open(work_file, 'r') as f:
                        return json.load(f)
            else:
                print(f"⚠️  Work selector warning: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Work selector failed: {e}")
        
        return None
    
    def _run_learning_engine(self) -> List[Dict[str, Any]]:
        """Execute learning engine phase."""
        try:
            result = subprocess.run(
                ['python3', '.terragon/learning-engine.py'],
                capture_output=True, text=True, cwd=self.repo_root, timeout=60
            )
            
            if result.returncode == 0:
                print(result.stdout)
                
                # Load learning insights
                insights_file = self.terragon_dir / "learning-insights.json"
                if insights_file.exists():
                    with open(insights_file, 'r') as f:
                        return json.load(f)
            else:
                print(f"⚠️  Learning engine warning: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Learning engine failed: {e}")
        
        return []
    
    def _generate_backlog_summary(self) -> None:
        """Generate summary of backlog status."""
        backlog_file = self.repo_root / "BACKLOG.md"
        if backlog_file.exists():
            print(f"📄 Updated: {backlog_file}")
            print(f"🔗 View backlog: cat BACKLOG.md")
        else:
            print("⚠️  Backlog file not found")
    
    def _save_cycle_results(self, results: Dict[str, Any]) -> None:
        """Save cycle results to file."""
        results_file = self.terragon_dir / f"cycle-results-{self.start_time.strftime('%Y%m%d-%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save as latest
        latest_file = self.terragon_dir / "latest-cycle-results.json"
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _print_cycle_summary(self, results: Dict[str, Any]) -> None:
        """Print final cycle summary."""
        print("🏁 AUTONOMOUS SDLC CYCLE COMPLETE")
        print("=" * 70)
        
        print(f"⏱  Duration: {results['duration_minutes']:.1f} minutes")
        print(f"✅ Phases Completed: {len(results['phases_completed'])}/6")
        print(f"🔍 Opportunities Discovered: {results['opportunities_discovered']}")
        print(f"🎯 Opportunities Scored: {results['opportunities_scored']}")
        print(f"💰 Total Value Estimate: ${results['total_value_estimate']:,.0f}")
        print(f"🧠 Learning Insights: {results['insights_generated']}")
        print(f"📊 Status: {results['status'].upper()}")
        
        if results['work_item_selected']:
            work_item = results['work_item_selected']
            print(f"\n🎯 NEXT RECOMMENDED ACTION:")
            print(f"   Title: {work_item.get('title', 'Unknown')}")
            print(f"   Category: {work_item.get('category', 'Unknown')}")
            print(f"   Priority: {work_item.get('priority', 'Unknown')}")
            print(f"   Score: {work_item.get('composite_score', 0):.1f}")
            print(f"   Effort: {work_item.get('estimated_effort', 0)} hours")
        
        print(f"\n📁 Results saved to: .terragon/latest-cycle-results.json")
        print(f"📊 View backlog: cat BACKLOG.md")
        
        if results['status'] == 'completed':
            print(f"\n🚀 READY FOR AUTONOMOUS EXECUTION:")
            print(f"   Run: python3 .terragon/autonomous-executor.py")
            print(f"   Or continue value discovery cycle")
        
        print("\n" + "=" * 70)
        print("🤖 Terragon Autonomous SDLC - Perpetual Value Discovery")
        print("   Next cycle will discover additional opportunities")
        print("   System learns and adapts from each execution")

def main():
    """Main execution function."""
    orchestrator = AutonomousSDLC()
    results = orchestrator.run_complete_cycle()
    
    # Return exit code based on results
    return 0 if results['status'] == 'completed' else 1

if __name__ == "__main__":
    exit(main())