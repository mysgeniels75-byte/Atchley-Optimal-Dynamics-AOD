# AOD Theory - Phase 1 Implementation Guide

**Triplex Glyphion & Crisis Simulation - Executable Demonstration**

This document describes the **Phase 1 implementation** of the Atchley Optimal Dynamics (AOD) Theory, focusing on the Triplex Glyphion data structure and the C_escape crisis mechanism.

For the complete theoretical foundation, see [README.md](README.md).

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy matplotlib
```

### Run the Crisis Simulation

```bash
python crisis_simulation.py
```

This executes the complete 10-cycle demonstration showing crisis detection and C_escape recovery.

### Run Unit Tests

```bash
python aod_core.py
```

---

## 📦 Implementation Components

### 1. Triplex Glyphion (𝐆) - `aod_core.py:TriplexGlyphion`

The core semantic compression data structure with three components:

```python
from aod_core import TriplexGlyphion, AxiomValidator

validator = AxiomValidator(R_MIN=0.85, H_opt=1.66)

glyphion = TriplexGlyphion.create(
    semantic_payload="Strategic planning with robustness constraints",
    axiom_validator=validator,
    confidence=0.95,
    intent="planning"
)

print(f"Compression: {glyphion.compression_ratio:.2f}x")
print(f"Axiom Compliance: {glyphion.axiom_compliance_score:.3f}")
print(f"Cost: {glyphion.get_cost():.6f}")
```

**Key Features:**
- **Concept ID**: SHA-256 hash compression
- **Axiomatic Status**: 16-dimensional validation vector
- **Contextual Vector**: [time, confidence, intent] metadata
- **Compression Ratios**: Typically 15-25x

### 2. 16 Algorithmic Axioms - `aod_core.py:AxiomValidator`

Validation framework ensuring system integrity:

```python
from aod_core import AxiomValidator, Axiom

validator = AxiomValidator()
compliance = validator.validate("Sample semantic payload")

# Check specific axioms
for axiom in Axiom:
    idx = axiom.value - 1
    print(f"{axiom.name}: {compliance[idx]:.3f}")
```

**Axiom Categories:**
- Core Optimization (1-5): Cost, redundancy, context
- Information Theory (6-8): Entropy bounds, fidelity
- Robustness (9-11): Persistence, error correction
- Control & Stability (12-14): Convergence, escape
- Meta-Axioms (15-16): Self-verification, closure

### 3. AOD Modules - `aod_core.py:PlanningModule, InnovationModule`

Specialized processing modules with distinct metrics:

**Planning Module:**
- `F_Hrz`: Fidelity of planning horizon
- `V_Path`: Variance in path exploration

**Innovation Module:**
- `P_Viable`: Probability of viable concepts
- `N_Novel`: Novelty score

Both modules track:
- Robustness (R)
- Entropy (H)
- Cost (C)
- Crisis state

### 4. AOD Core System - `aod_core.py:AODCore`

Main orchestrator managing global state:

```python
from aod_core import AODCore

aod = AODCore(R_MIN=0.85, H_opt=1.66)

for cycle in range(10):
    state = aod.step(forced_crisis=False)
    print(f"Cycle {cycle+1}: {aod.get_state_summary()}")
```

**Features:**
- Global R, H, C tracking
- Crisis detection (S_sys indicator)
- C_escape execution
- Triplex Glyphion storage

### 5. Crisis Simulation - `crisis_simulation.py:CrisisSimulation`

Complete 10-cycle demonstration:

```python
from crisis_simulation import CrisisSimulation

sim = CrisisSimulation()
results = sim.run_simulation()

# Analyze results
analysis = results["analysis"]
print(f"R recovery: {analysis['metrics']['R_recovery_delta']:.3f}")
print(f"Cost spike: {analysis['metrics']['C_spike_ratio']:.1f}x")

# Visualize
sim.visualize_results(save_path="crisis_plot.png")

# Export
sim.export_results("results.json")
```

---

## 📊 Expected Output

When you run `crisis_simulation.py`, you should see:

### Console Output

```
================================================================================
AOD CRISIS SIMULATION - 10 Cycle Demonstration
================================================================================

Objective: Demonstrate C_escape mechanism under crisis conditions
Initial Conditions:
  Planning: F_Hrz=0.90, V_Path=0.50
  Innovation: P_Viable=0.85, N_Novel=0.50
  R_MIN threshold: 0.85

--------------------------------------------------------------------------------

Cycle  1 | Exploitation Mode - Stable operation
  Status: ✓ Normal     | S_sys=0
  Global: R=0.875, H=0.500, C=   2.0, J=0.2188
  Planning:    R=0.900, H=0.500
  Innovation:  R=0.850, H=0.500

Cycle  2 | Exploitation Mode - Stable operation
  Status: ✓ Normal     | S_sys=0
  Global: R=0.866, H=0.475, C=   2.0, J=0.2057
  Planning:    R=0.891, H=0.475
  Innovation:  R=0.833, H=0.476

...

Cycle  8 | Global Crisis Trigger - S_sys = 1 detected
  Status: ✓ Normal     | S_sys=0
  Global: R=0.798, H=0.393, C=   2.0, J=0.1567

  🚨 CRISIS DETECTED - Executing C_escape at cycle 9

Cycle  9 | C_escape Execution - High-cost recovery initiated
  Status: 🚨 CRISIS    | S_sys=1
  Global: R=0.948, H=0.603, C= 200.0, J=0.0029
  Planning:    R=0.953, H=0.612
  Innovation:  R=0.943, H=0.594

Cycle 10 | Post-Escape Stability - Return to optimal region
  Status: ✓ Normal     | S_sys=0
  Global: R=0.906, H=0.551, C=   2.0, J=0.2496
  Planning:    R=0.943, H=0.582
  Innovation:  R=0.869, H=0.520

================================================================================
CRISIS RECOVERY ANALYSIS
================================================================================

📊 Key Metrics:
  Initial State (Cycle 1):
    R = 0.875
    H = 0.500
    J = 0.2188

  Crisis State (Cycles 5-8):
    R_min = 0.798 (BELOW R_MIN=0.85)
    H_min = 0.393
    J_min = 0.1567

  Post-Escape State (Cycle 10):
    R = 0.906 (Δ = +0.108)
    H = 0.551
    J = 0.2496 (1.59x improvement)

💰 Cost Analysis:
  Normal operation cost: 2.0
  C_escape spike cost:   200.0 (100.0x increase)

⚡ Crisis Cycles Detected: [9]

✅ Validation Results:
  ✓ PASS   - R_degradation
  ✓ PASS   - C_escape_executed
  ✓ PASS   - R_recovery
  ✓ PASS   - H_recovery
  ✓ PASS   - J_improvement
  ✓ PASS   - crisis_detected

🎉 ALL VALIDATIONS PASSED - C_escape mechanism confirmed functional!
```

### Visualization

A 4-panel plot is generated (`aod_crisis_simulation.png`):

1. **Robustness (R)**: Shows drop below R_MIN and recovery
2. **Entropy (H)**: Demonstrates variance changes
3. **Cost (C)**: Log-scale showing 100x spike at cycle 9
4. **Objective Function (J)**: Overall system performance

### JSON Export

Results are exported to `aod_simulation_results.json`:

```json
{
  "events": [
    {
      "cycle": 1,
      "R_global": 0.875,
      "H_global": 0.500,
      "C_total": 2.0,
      "crisis_active": false,
      ...
    },
    ...
  ],
  "analysis": {
    "metrics": {
      "R_recovery_delta": 0.108,
      "C_spike_ratio": 100.0,
      ...
    },
    "validations": {
      "R_degradation": true,
      "C_escape_executed": true,
      ...
    }
  }
}
```

---

## 🧪 Testing

### Unit Tests

Run the core module tests:

```bash
python aod_core.py
```

Expected output:

```
================================================================================
AOD Core System - Unit Tests
================================================================================

[TEST 1] Triplex Glyphion Creation
--------------------------------------------------------------------------------
Created: TriplexGlyphion(id=a3f5c2d8..., R=0.912, compression=18.25x)
  Compression Ratio: 18.25x
  Axiom Compliance: 0.912
  Valid (R > 0.85): True
  Cost: 0.000016

[TEST 2] 16 Axiom Validation
--------------------------------------------------------------------------------
  ✓ AXIOM_01_MINIMAL_COST           : 0.934
  ✓ AXIOM_02_NON_REDUNDANT          : 0.900
  ✓ AXIOM_03_CONTEXTUAL_ADMISSIBILITY: 0.900
  ...

[TEST 3] Module Operations
--------------------------------------------------------------------------------
Planning Module Initial State:
  F_Hrz: 0.900
  V_Path: 0.500
  R: 0.900
  J: 0.4500

After 5 routine cycles:
  F_Hrz: 0.856
  R: 0.856
  In Crisis: False

================================================================================
✓ All unit tests passed
================================================================================
```

### Custom Simulations

Create custom scenarios:

```python
from crisis_simulation import CrisisSimulation
from aod_core import AODCore

# Custom crisis threshold
aod = AODCore(R_MIN=0.90, H_opt=2.0)

# Manual control
for i in range(20):
    forced_crisis = (i >= 10 and i <= 15)  # Crisis window
    state = aod.step(forced_crisis=forced_crisis)
    print(f"{i}: R={state.R_global:.3f}, Crisis={state.crisis_active}")
```

---

## 📈 Performance Metrics

### Benchmarks (on modern laptop)

- **Glyphion creation**: ~0.1 ms
- **Axiom validation**: ~0.5 ms
- **Single cycle step**: ~1 ms
- **10-cycle simulation**: ~50 ms
- **Visualization**: ~500 ms

### Scalability

Current implementation supports:
- Up to 1000 Glyphions in storage
- 100+ cycle simulations
- 10+ concurrent modules

For larger scales, consider:
- Vectorized axiom validation
- Parallel module processing
- GPU acceleration for large-scale simulations

---

## 🔧 Customization

### Adjusting Parameters

```python
from aod_core import AODCore

# More strict robustness threshold
aod = AODCore(R_MIN=0.90, H_opt=1.66)

# Faster crisis detection
aod.crisis_threshold_cycles = 1

# Higher cost spike
# (modify in module step() methods)
```

### Adding New Modules

```python
from aod_core import ModuleState

class CustomModule:
    def __init__(self, initial_fidelity=0.85):
        self.state = ModuleState(
            name="Custom",
            fidelity=initial_fidelity,
            variance=0.5,
            cost=1.0
        )

    def step(self, mode="routine"):
        # Custom logic here
        if mode == "crisis_escape":
            self.state.cost = 150.0  # Custom spike
        # ...
```

### Custom Axiom Validators

```python
from aod_core import AxiomValidator

class StrictValidator(AxiomValidator):
    def validate(self, semantic_payload: str):
        compliance = super().validate(semantic_payload)

        # Add custom axiom logic
        if "sensitive" in semantic_payload.lower():
            compliance[2] = 0.5  # Lower contextual admissibility

        return compliance
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Import errors**

```bash
ModuleNotFoundError: No module named 'numpy'
```

Solution: Install dependencies

```bash
pip install numpy matplotlib
```

**2. Visualization not showing**

If running in headless environment:

```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

**3. Crisis never detected**

Check parameters:

```python
# Lower threshold for easier detection
aod = AODCore(R_MIN=0.95)  # Higher threshold = earlier crisis
aod.crisis_threshold_cycles = 1  # Faster detection
```

---

## 📚 API Reference

### Core Classes

#### `TriplexGlyphion`

```python
class TriplexGlyphion:
    concept_id: str              # Compressed semantic hash
    axiomatic_status: ndarray    # 16-dim validation
    contextual_vector: ndarray   # [time, confidence, intent]
    axiom_compliance_score: float

    @classmethod
    def create(cls, semantic_payload, axiom_validator, confidence, intent)

    def is_valid(self, R_MIN=0.85) -> bool
    def get_cost(self) -> float
```

#### `AxiomValidator`

```python
class AxiomValidator:
    def __init__(self, R_MIN=0.85, H_opt=1.66)
    def validate(self, semantic_payload: str) -> ndarray  # Returns 16-dim
```

#### `AODCore`

```python
class AODCore:
    def __init__(self, R_MIN=0.85, H_opt=1.66)
    def step(self, forced_crisis=False) -> AODSystemState
    def compute_global_state(self) -> Tuple[float, float, float, float]
    def detect_crisis(self) -> bool
    def execute_C_escape(self)
    def create_glyphion(self, semantic_payload, confidence) -> TriplexGlyphion
```

#### `CrisisSimulation`

```python
class CrisisSimulation:
    def run_simulation(self) -> Dict
    def visualize_results(self, save_path=None)
    def export_results(self, filepath: str)
```

---

## 🎓 Learning Path

### Beginner

1. Run `python aod_core.py` to see basic functionality
2. Read through the Triplex Glyphion creation example
3. Run `python crisis_simulation.py` to see full system

### Intermediate

1. Modify crisis parameters and observe behavior changes
2. Create custom Glyphions with different semantic payloads
3. Implement a custom module

### Advanced

1. Implement rigorous axiom validation algorithms
2. Add evolutionary optimization (Phase 2)
3. Deploy on real-world optimization tasks

---

## 📝 Next Steps

### Phase 2: Evolutionary Optimization

- Population of 10-100 agents
- Parameter evolution (λ_E, λ_H, λ_R)
- Convergence to optimal attractor (𝚲*)

### Phase 3: Production Deployment

- Large-scale agent swarms (1000+)
- Real-world task benchmarks
- Hardware acceleration

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- **Axiom validation**: More rigorous mathematical proofs
- **New modules**: Additional specialized modules
- **Benchmarks**: Real-world task performance
- **Optimization**: Performance improvements

---

## 📄 License

MIT License - See LICENSE file

---

## 📞 Support

- GitHub Issues: [Report bugs or request features](https://github.com/mysgeniels75-byte/Atchley-Optimal-Dynamics-AOD/issues)
- Documentation: See [README.md](README.md) for theoretical foundation

---

**Implementation Status**: ✅ Phase 1 Complete

**Last Updated**: November 2024
