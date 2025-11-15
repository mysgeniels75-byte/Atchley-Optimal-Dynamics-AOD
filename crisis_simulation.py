"""
AOD Crisis Simulation - 10-Cycle Demonstration
===============================================

This simulation demonstrates the complete AOD system dynamics including:
1. Initial stable operation (Cycles 1-3)
2. Gradual degradation (Cycle 4)
3. Crisis development (Cycles 5-7)
4. Crisis detection (Cycle 8)
5. C_escape execution (Cycle 9)
6. Post-recovery stability (Cycle 10)

Expected Results:
- R drops from 0.88 → 0.80 during crisis
- C_escape spikes cost to 100x at cycle 9
- R recovers to ~0.90+ after escape
- System demonstrates resilience and self-correction
"""

import numpy as np
import matplotlib.pyplot as plt
from aod_core import AODCore, TriplexGlyphion
from typing import List, Dict
import json


class CrisisSimulation:
    """
    Orchestrates the 10-cycle crisis demonstration

    This follows the exact sequence described in the AOD specification:
    - Cycles 1-3: Exploitation mode (stable)
    - Cycle 4: Planned degradation
    - Cycles 5-7: Crisis setup (Planning module stagnation)
    - Cycle 8: Global crisis trigger
    - Cycle 9: C_escape execution
    - Cycle 10: Post-escape stability
    """

    def __init__(self):
        self.aod_system = AODCore(R_MIN=0.85, H_opt=1.66)
        self.cycle_events: List[Dict] = []

    def run_simulation(self) -> Dict:
        """
        Execute the complete 10-cycle simulation

        Returns:
            Dictionary containing all simulation results and metrics
        """
        print("=" * 80)
        print("AOD CRISIS SIMULATION - 10 Cycle Demonstration")
        print("=" * 80)
        print("\nObjective: Demonstrate C_escape mechanism under crisis conditions")
        print(f"Initial Conditions:")
        print(f"  Planning: F_Hrz=0.90, V_Path=0.50")
        print(f"  Innovation: P_Viable=0.85, N_Novel=0.50")
        print(f"  R_MIN threshold: {self.aod_system.R_MIN}")
        print("\n" + "-" * 80)

        # Define cycle-specific behaviors
        cycle_schedule = [
            # Cycle 1-3: Stable exploitation
            {"cycle": 1, "mode": "exploitation", "forced_crisis": False,
             "description": "Exploitation Mode - Stable operation"},
            {"cycle": 2, "mode": "exploitation", "forced_crisis": False,
             "description": "Exploitation Mode - Stable operation"},
            {"cycle": 3, "mode": "exploitation", "forced_crisis": False,
             "description": "Exploitation Mode - Stable operation"},

            # Cycle 4: Planned degradation
            {"cycle": 4, "mode": "degradation", "forced_crisis": False,
             "description": "Planned Degradation - Begin controlled decline"},

            # Cycles 5-7: Crisis setup
            {"cycle": 5, "mode": "crisis_setup", "forced_crisis": True,
             "description": "Crisis Setup - Planning module stagnation (F_Hrz < R_MIN)"},
            {"cycle": 6, "mode": "crisis_setup", "forced_crisis": True,
             "description": "Crisis Deepening - Continued degradation"},
            {"cycle": 7, "mode": "crisis_setup", "forced_crisis": True,
             "description": "Crisis Critical - Saddle point imminent"},

            # Cycle 8: Crisis trigger (system detects)
            {"cycle": 8, "mode": "crisis_trigger", "forced_crisis": True,
             "description": "Global Crisis Trigger - S_sys = 1 detected"},

            # Cycle 9: C_escape (automatic by system)
            {"cycle": 9, "mode": "auto", "forced_crisis": False,
             "description": "C_escape Execution - High-cost recovery initiated"},

            # Cycle 10: Recovery
            {"cycle": 10, "mode": "recovery", "forced_crisis": False,
             "description": "Post-Escape Stability - Return to optimal region"},
        ]

        # Execute simulation
        for schedule in cycle_schedule:
            cycle = schedule["cycle"]
            description = schedule["description"]
            forced_crisis = schedule["forced_crisis"]

            # Execute cycle
            state = self.aod_system.step(forced_crisis=forced_crisis)

            # Record event
            event = {
                "cycle": cycle,
                "description": description,
                "R_global": state.R_global,
                "H_global": state.H_global,
                "C_total": state.C_total,
                "J_AOD": state.J_AOD,
                "S_sys": state.S_sys,
                "crisis_active": state.crisis_active,
                "planning_R": state.planning_R,
                "planning_H": state.planning_H,
                "innovation_R": state.innovation_R,
                "innovation_H": state.innovation_H
            }
            self.cycle_events.append(event)

            # Print cycle summary
            self._print_cycle_summary(event)

        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)

        # Generate analysis
        analysis = self._analyze_results()
        self._print_analysis(analysis)

        return {
            "events": self.cycle_events,
            "analysis": analysis
        }

    def _print_cycle_summary(self, event: Dict):
        """Print formatted summary for each cycle"""
        crisis_indicator = "🚨 CRISIS" if event["crisis_active"] else "✓ Normal"
        s_sys_indicator = f"S_sys={event['S_sys']}"

        print(f"\nCycle {event['cycle']:2d} | {event['description']}")
        print(f"  Status: {crisis_indicator:12s} | {s_sys_indicator}")
        print(f"  Global: R={event['R_global']:.3f}, H={event['H_global']:.3f}, "
              f"C={event['C_total']:6.1f}, J={event['J_AOD']:.4f}")
        print(f"  Planning:    R={event['planning_R']:.3f}, H={event['planning_H']:.3f}")
        print(f"  Innovation:  R={event['innovation_R']:.3f}, H={event['innovation_H']:.3f}")

    def _analyze_results(self) -> Dict:
        """Analyze simulation results and validate expected behavior"""
        events = self.cycle_events

        # Extract key metrics
        R_initial = events[0]["R_global"]
        R_min_crisis = min(e["R_global"] for e in events[4:8])  # Cycles 5-8
        R_post_escape = events[9]["R_global"]  # Cycle 10

        H_initial = events[0]["H_global"]
        H_min_crisis = min(e["H_global"] for e in events[4:8])
        H_post_escape = events[9]["H_global"]

        C_normal = events[0]["C_total"]
        C_escape = events[8]["C_total"]  # Cycle 9 (index 8)

        J_initial = events[0]["J_AOD"]
        J_min_crisis = min(e["J_AOD"] for e in events[4:8])
        J_post_escape = events[9]["J_AOD"]

        # Crisis detection
        crisis_cycles = [e["cycle"] for e in events if e["crisis_active"]]

        # Validation checks
        validations = {
            "R_degradation": R_min_crisis < self.aod_system.R_MIN,
            "C_escape_executed": C_escape > 10 * C_normal,
            "R_recovery": R_post_escape > R_min_crisis,
            "H_recovery": H_post_escape > H_min_crisis,
            "J_improvement": J_post_escape > J_min_crisis,
            "crisis_detected": len(crisis_cycles) > 0
        }

        all_valid = all(validations.values())

        return {
            "metrics": {
                "R_initial": R_initial,
                "R_min_crisis": R_min_crisis,
                "R_post_escape": R_post_escape,
                "R_recovery_delta": R_post_escape - R_min_crisis,
                "H_initial": H_initial,
                "H_min_crisis": H_min_crisis,
                "H_post_escape": H_post_escape,
                "C_normal": C_normal,
                "C_escape": C_escape,
                "C_spike_ratio": C_escape / C_normal,
                "J_initial": J_initial,
                "J_min_crisis": J_min_crisis,
                "J_post_escape": J_post_escape,
                "J_recovery_ratio": J_post_escape / J_min_crisis
            },
            "crisis_cycles": crisis_cycles,
            "validations": validations,
            "all_validations_passed": all_valid
        }

    def _print_analysis(self, analysis: Dict):
        """Print detailed analysis of simulation results"""
        print("\n" + "=" * 80)
        print("CRISIS RECOVERY ANALYSIS")
        print("=" * 80)

        metrics = analysis["metrics"]

        print("\n📊 Key Metrics:")
        print(f"  Initial State (Cycle 1):")
        print(f"    R = {metrics['R_initial']:.3f}")
        print(f"    H = {metrics['H_initial']:.3f}")
        print(f"    J = {metrics['J_initial']:.4f}")

        print(f"\n  Crisis State (Cycles 5-8):")
        print(f"    R_min = {metrics['R_min_crisis']:.3f} "
              f"({'BELOW' if metrics['R_min_crisis'] < 0.85 else 'ABOVE'} R_MIN=0.85)")
        print(f"    H_min = {metrics['H_min_crisis']:.3f}")
        print(f"    J_min = {metrics['J_min_crisis']:.4f}")

        print(f"\n  Post-Escape State (Cycle 10):")
        print(f"    R = {metrics['R_post_escape']:.3f} "
              f"(Δ = +{metrics['R_recovery_delta']:.3f})")
        print(f"    H = {metrics['H_post_escape']:.3f}")
        print(f"    J = {metrics['J_post_escape']:.4f} "
              f"({metrics['J_recovery_ratio']:.2f}x improvement)")

        print(f"\n💰 Cost Analysis:")
        print(f"  Normal operation cost: {metrics['C_normal']:.1f}")
        print(f"  C_escape spike cost:   {metrics['C_escape']:.1f} "
              f"({metrics['C_spike_ratio']:.1f}x increase)")

        print(f"\n⚡ Crisis Cycles Detected: {analysis['crisis_cycles']}")

        print("\n✅ Validation Results:")
        for check, passed in analysis["validations"].items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status:8s} - {check}")

        if analysis["all_validations_passed"]:
            print("\n🎉 ALL VALIDATIONS PASSED - C_escape mechanism confirmed functional!")
        else:
            print("\n⚠️  Some validations failed - review simulation parameters")

    def visualize_results(self, save_path: str = None):
        """Generate visualization of simulation dynamics"""
        events = self.cycle_events
        cycles = [e["cycle"] for e in events]
        R_global = [e["R_global"] for e in events]
        H_global = [e["H_global"] for e in events]
        C_total = [e["C_total"] for e in events]
        J_AOD = [e["J_AOD"] for e in events]
        crisis_active = [e["crisis_active"] for e in events]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('AOD Crisis Simulation - 10 Cycle Dynamics', fontsize=16, fontweight='bold')

        # Plot 1: Robustness (R)
        ax1 = axes[0, 0]
        ax1.plot(cycles, R_global, 'o-', linewidth=2, markersize=8, color='steelblue', label='R_global')
        ax1.axhline(y=0.85, color='red', linestyle='--', label='R_MIN threshold')
        ax1.fill_between(cycles, 0, 1, where=crisis_active, alpha=0.2, color='red', label='Crisis')
        ax1.set_xlabel('Cycle')
        ax1.set_ylabel('Robustness (R)')
        ax1.set_title('Global Robustness Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.7, 1.0])

        # Plot 2: Entropy (H)
        ax2 = axes[0, 1]
        ax2.plot(cycles, H_global, 'o-', linewidth=2, markersize=8, color='forestgreen', label='H_global')
        ax2.axhline(y=1.66, color='orange', linestyle='--', label='H_opt target')
        ax2.fill_between(cycles, 0, 2, where=crisis_active, alpha=0.2, color='red')
        ax2.set_xlabel('Cycle')
        ax2.set_ylabel('Entropy (H)')
        ax2.set_title('Information Entropy Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Cost (C) - Log scale
        ax3 = axes[1, 0]
        ax3.semilogy(cycles, C_total, 'o-', linewidth=2, markersize=8, color='crimson', label='C_total')
        ax3.fill_between(cycles, 0.1, 1000, where=crisis_active, alpha=0.2, color='red')
        ax3.set_xlabel('Cycle')
        ax3.set_ylabel('Total Cost (C) [log scale]')
        ax3.set_title('Cost Dynamics (C_escape spike at Cycle 9)')
        ax3.legend()
        ax3.grid(True, alpha=0.3, which='both')

        # Plot 4: Objective Function (J)
        ax4 = axes[1, 1]
        ax4.plot(cycles, J_AOD, 'o-', linewidth=2, markersize=8, color='purple', label='J_AOD = (R×H)/C')
        ax4.fill_between(cycles, 0, max(J_AOD) * 1.2, where=crisis_active, alpha=0.2, color='red')
        ax4.set_xlabel('Cycle')
        ax4.set_ylabel('Objective Function (J)')
        ax4.set_title('Global Objective Function (Higher is Better)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Visualization saved to: {save_path}")

        plt.show()

    def export_results(self, filepath: str):
        """Export simulation results to JSON"""
        results = {
            "events": self.cycle_events,
            "analysis": self._analyze_results()
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results exported to: {filepath}")


def demonstrate_triplex_glyphion(aod_system: AODCore):
    """
    Demonstrate Triplex Glyphion semantic compression

    This shows how the Glyphion structure enables:
    - High compression ratios (C minimization)
    - Axiomatic validation (R maximization)
    - Contextual adaptability (H_opt tuning)
    """
    print("\n" + "=" * 80)
    print("TRIPLEX GLYPHION DEMONSTRATION")
    print("=" * 80)

    # Example 1: Planning directive
    print("\n[Example 1] Planning Directive Compression")
    planning_directive = (
        "Execute multi-horizon strategic planning with emphasis on robust path "
        "generation while maintaining adaptive variance to handle unexpected "
        "environmental perturbations in real-time decision making contexts"
    )

    glyphion1 = aod_system.create_glyphion(
        semantic_payload=planning_directive,
        confidence=0.92
    )

    print(f"Original payload: {len(planning_directive)} chars")
    print(f"  \"{planning_directive[:60]}...\"")
    print(f"\nCompressed Glyphion:")
    print(f"  Concept ID: {glyphion1.concept_id}")
    print(f"  Compression: {glyphion1.compression_ratio:.2f}x")
    print(f"  Axiom Compliance: {glyphion1.axiom_compliance_score:.3f}")
    print(f"  Cost (C_M): {glyphion1.get_cost():.6f}")
    print(f"  Valid: {'✓ YES' if glyphion1.is_valid() else '✗ NO'}")

    # Example 2: Innovation concept
    print("\n[Example 2] Innovation Concept Compression")
    innovation_concept = (
        "Generate novel architectural patterns for distributed AI agents "
        "using memristor-based analog computation with topological holography"
    )

    glyphion2 = aod_system.create_glyphion(
        semantic_payload=innovation_concept,
        confidence=0.78
    )

    print(f"Original payload: {len(innovation_concept)} chars")
    print(f"  \"{innovation_concept[:60]}...\"")
    print(f"\nCompressed Glyphion:")
    print(f"  Concept ID: {glyphion2.concept_id}")
    print(f"  Compression: {glyphion2.compression_ratio:.2f}x")
    print(f"  Axiom Compliance: {glyphion2.axiom_compliance_score:.3f}")
    print(f"  Cost (C_M): {glyphion2.get_cost():.6f}")

    # Storage summary
    print(f"\n📦 Glyphion Storage Summary:")
    print(f"  Total stored: {len(aod_system.glyphion_storage)}")
    total_cost = sum(g.get_cost() for g in aod_system.glyphion_storage)
    avg_compression = np.mean([g.compression_ratio for g in aod_system.glyphion_storage])
    print(f"  Total cost: {total_cost:.6f}")
    print(f"  Avg compression: {avg_compression:.2f}x")


if __name__ == "__main__":
    # Run complete simulation
    sim = CrisisSimulation()
    results = sim.run_simulation()

    # Demonstrate Triplex Glyphion
    demonstrate_triplex_glyphion(sim.aod_system)

    # Visualize results
    print("\n" + "=" * 80)
    print("Generating visualization...")
    sim.visualize_results(save_path="aod_crisis_simulation.png")

    # Export results
    sim.export_results("aod_simulation_results.json")

    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 80)
