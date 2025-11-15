"""
Basic Usage Examples for AOD Theory Implementation
===================================================

This script demonstrates the fundamental operations of the AOD system:
1. Creating Triplex Glyphions
2. Validating against 16 Axioms
3. Running simple simulations
"""

import sys
sys.path.append('..')

from aod_core import (
    TriplexGlyphion,
    AxiomValidator,
    AODCore,
    PlanningModule,
    InnovationModule,
    Axiom
)


def example_1_glyphion_creation():
    """Example 1: Create and validate Triplex Glyphions"""
    print("=" * 80)
    print("EXAMPLE 1: Triplex Glyphion Creation")
    print("=" * 80)

    validator = AxiomValidator(R_MIN=0.85, H_opt=1.66)

    # Create a planning-related Glyphion
    glyphion_plan = TriplexGlyphion.create(
        semantic_payload="Optimize multi-agent routing with minimal latency",
        axiom_validator=validator,
        confidence=0.92,
        intent="planning"
    )

    print("\n[Planning Glyphion]")
    print(f"  Original: 'Optimize multi-agent routing with minimal latency'")
    print(f"  Concept ID: {glyphion_plan.concept_id}")
    print(f"  Compression: {glyphion_plan.compression_ratio:.2f}x")
    print(f"  Axiom Compliance: {glyphion_plan.axiom_compliance_score:.3f}")
    print(f"  Valid: {glyphion_plan.is_valid()}")
    print(f"  Cost: {glyphion_plan.get_cost():.6f}")

    # Create an innovation-related Glyphion
    glyphion_innov = TriplexGlyphion.create(
        semantic_payload="Develop novel neural architecture with memristor arrays",
        axiom_validator=validator,
        confidence=0.78,
        intent="innovation"
    )

    print("\n[Innovation Glyphion]")
    print(f"  Original: 'Develop novel neural architecture with memristor arrays'")
    print(f"  Concept ID: {glyphion_innov.concept_id}")
    print(f"  Compression: {glyphion_innov.compression_ratio:.2f}x")
    print(f"  Axiom Compliance: {glyphion_innov.axiom_compliance_score:.3f}")
    print(f"  Valid: {glyphion_innov.is_valid()}")

    print("\n✓ Example 1 complete\n")


def example_2_axiom_validation():
    """Example 2: Detailed axiom validation"""
    print("=" * 80)
    print("EXAMPLE 2: 16 Axiom Validation")
    print("=" * 80)

    validator = AxiomValidator()

    test_payloads = [
        ("High quality", "This is a well-structured semantic payload with optimal entropy"),
        ("Redundant", "test test test test test test test"),
        ("Minimal", "OK"),
    ]

    for name, payload in test_payloads:
        print(f"\n[{name} Payload]")
        print(f"  Content: '{payload}'")

        compliance = validator.validate(payload)
        overall_score = compliance.mean()

        print(f"  Overall Compliance: {overall_score:.3f}")

        # Show compliance by category
        core = compliance[0:5].mean()
        info = compliance[5:8].mean()
        robust = compliance[8:11].mean()
        control = compliance[11:14].mean()
        meta = compliance[14:16].mean()

        print(f"    Core Optimization (1-5):   {core:.3f}")
        print(f"    Information Theory (6-8):  {info:.3f}")
        print(f"    Robustness (9-11):         {robust:.3f}")
        print(f"    Control & Stability (12-14): {control:.3f}")
        print(f"    Meta-Axioms (15-16):       {meta:.3f}")

    print("\n✓ Example 2 complete\n")


def example_3_simple_simulation():
    """Example 3: Simple AOD system simulation"""
    print("=" * 80)
    print("EXAMPLE 3: Simple AOD Simulation (5 cycles)")
    print("=" * 80)

    aod = AODCore(R_MIN=0.85, H_opt=1.66)

    print("\nInitial State:")
    print(f"  Planning:   F_Hrz={aod.planning.state.fidelity:.3f}, "
          f"V_Path={aod.planning.state.variance:.3f}")
    print(f"  Innovation: P_Viable={aod.innovation.state.fidelity:.3f}, "
          f"N_Novel={aod.innovation.state.variance:.3f}")

    print("\nRunning 5 cycles...")
    print("-" * 80)

    for i in range(5):
        state = aod.step(forced_crisis=False)

        crisis_marker = "🚨 CRISIS" if state.crisis_active else "✓"

        print(f"Cycle {state.cycle}: {crisis_marker} | "
              f"R={state.R_global:.3f}, "
              f"H={state.H_global:.3f}, "
              f"C={state.C_total:6.1f}, "
              f"J={state.J_AOD:.4f}")

    print("\nFinal State:")
    final = aod.history[-1]
    print(f"  R_global: {final.R_global:.3f}")
    print(f"  H_global: {final.H_global:.3f}")
    print(f"  J_AOD: {final.J_AOD:.4f}")

    print("\n✓ Example 3 complete\n")


def example_4_module_operations():
    """Example 4: Direct module manipulation"""
    print("=" * 80)
    print("EXAMPLE 4: Module Operations")
    print("=" * 80)

    # Create isolated modules
    planning = PlanningModule(initial_fidelity=0.90, initial_variance=0.50)
    innovation = InnovationModule(initial_viability=0.85, initial_novelty=0.50)

    print("\n[Initial State]")
    print(f"  Planning:   R={planning.state.robustness:.3f}, "
          f"H={planning.state.entropy:.3f}, "
          f"J={planning.state.compute_objective():.4f}")
    print(f"  Innovation: R={innovation.state.robustness:.3f}, "
          f"H={innovation.state.entropy:.3f}, "
          f"J={innovation.state.compute_objective():.4f}")

    # Run routine cycles
    print("\n[After 3 routine cycles]")
    for _ in range(3):
        planning.step(mode="routine")
        innovation.step(mode="routine")

    print(f"  Planning:   R={planning.state.robustness:.3f}, "
          f"H={planning.state.entropy:.3f}, "
          f"Crisis={planning.state.in_crisis}")
    print(f"  Innovation: R={innovation.state.robustness:.3f}, "
          f"H={innovation.state.entropy:.3f}, "
          f"Crisis={innovation.state.in_crisis}")

    # Execute C_escape
    print("\n[Execute C_escape]")
    planning.step(mode="crisis_escape")
    innovation.step(mode="crisis_escape")

    print(f"  Planning:   R={planning.state.robustness:.3f} (recovered), "
          f"C={planning.state.cost:.1f} (spike)")
    print(f"  Innovation: R={innovation.state.robustness:.3f} (recovered), "
          f"C={innovation.state.cost:.1f} (spike)")

    print("\n✓ Example 4 complete\n")


def example_5_glyphion_storage():
    """Example 5: Glyphion storage and retrieval"""
    print("=" * 80)
    print("EXAMPLE 5: Glyphion Storage System")
    print("=" * 80)

    aod = AODCore()

    # Create multiple Glyphions
    concepts = [
        "Optimize network topology for minimal latency",
        "Generate novel solution to traveling salesman problem",
        "Balance exploration and exploitation in reinforcement learning",
        "Implement energy-efficient neural computation",
        "Design resilient distributed consensus algorithm"
    ]

    print("\nCreating and storing Glyphions...")
    print("-" * 80)

    for i, concept in enumerate(concepts):
        glyphion = aod.create_glyphion(
            semantic_payload=concept,
            confidence=0.85 + i * 0.02
        )
        print(f"{i+1}. {concept[:50]+'...' if len(concept) > 50 else concept}")
        print(f"   ID: {glyphion.concept_id}, "
              f"Compression: {glyphion.compression_ratio:.1f}x, "
              f"R: {glyphion.axiom_compliance_score:.3f}")

    # Summary
    print(f"\nStorage Summary:")
    print(f"  Total Glyphions: {len(aod.glyphion_storage)}")
    total_cost = sum(g.get_cost() for g in aod.glyphion_storage)
    avg_compression = sum(g.compression_ratio for g in aod.glyphion_storage) / len(aod.glyphion_storage)
    avg_compliance = sum(g.axiom_compliance_score for g in aod.glyphion_storage) / len(aod.glyphion_storage)

    print(f"  Total Cost: {total_cost:.6f}")
    print(f"  Avg Compression: {avg_compression:.2f}x")
    print(f"  Avg Axiom Compliance: {avg_compliance:.3f}")

    print("\n✓ Example 5 complete\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AOD THEORY - BASIC USAGE EXAMPLES")
    print("=" * 80 + "\n")

    example_1_glyphion_creation()
    example_2_axiom_validation()
    example_3_simple_simulation()
    example_4_module_operations()
    example_5_glyphion_storage()

    print("=" * 80)
    print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Run 'python ../crisis_simulation.py' for full crisis demonstration")
    print("  - See README_IMPLEMENTATION.md for detailed documentation")
    print("=" * 80 + "\n")
