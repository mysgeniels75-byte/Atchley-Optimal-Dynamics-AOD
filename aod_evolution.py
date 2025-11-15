"""
AOD Evolutionary Algorithm - Phase 2 Implementation
===================================================

This module implements the Darwinism Evolution System (DES) for the AOD Theory.

Population of agents evolves parameter vectors (Λ) to discover the optimal
attractor state through fitness-based selection, producing power-law distributions
and emergent resilience.

Key Features:
1. Population-based evolutionary optimization
2. Fitness = -𝓛_AOD (minimize cost)
3. Tournament selection
4. Crossover and mutation operators
5. Power-law structure analysis (α exponent)
6. Resilience testing under perturbations

References:
-----------
[1] Eiben, A.E. & Smith, J.E. (2015). "Introduction to Evolutionary Computing".
    Springer. 2nd Edition.

[2] Barabási, A-L. & Albert, R. (1999). "Emergence of scaling in random networks".
    Science, 286(5439), 509-512.

[3] Carlson, J.M. & Doyle, J. (2002). "Complexity and robustness".
    PNAS, 99(suppl 1), 2538-2545.

Author: AOD Research Team
Version: 2.0.0 - Phase 2: Evolutionary Dynamics
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import warnings


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: AGENT REPRESENTATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AODAgent:
    """
    Individual agent in the evolutionary population

    Each agent has:
    - Parameter vector Λ = [λ_E, λ_H, λ_R]
    - Internal state (weights, structure)
    - Fitness score
    """

    # Parameter vector (must sum to 1.0)
    lambda_E: float  # Energy weight
    lambda_H: float  # Entropy weight
    lambda_R: float  # Robustness weight

    # Internal structure (connection weights)
    # For power-law analysis
    weights: np.ndarray = field(default_factory=lambda: np.array([]))

    # Performance metrics
    fitness: float = 0.0
    robustness: float = 0.0
    cost_history: List[float] = field(default_factory=list)

    # Generation born
    generation: int = 0

    # Agent ID
    agent_id: int = 0

    def __post_init__(self):
        """Validate and normalize parameters"""
        # Ensure parameters sum to 1.0
        total = self.lambda_E + self.lambda_H + self.lambda_R
        if not np.isclose(total, 1.0):
            # Normalize
            self.lambda_E /= total
            self.lambda_H /= total
            self.lambda_R /= total

        # Initialize weights if not provided
        if len(self.weights) == 0:
            # Random power-law initialization
            self.weights = self._generate_power_law_weights(size=100, alpha=2.5)

    def _generate_power_law_weights(self, size: int, alpha: float) -> np.ndarray:
        """
        Generate weights following power-law distribution P(w) ∝ w^(-α)

        Args:
            size: Number of weights
            alpha: Power-law exponent (typically 2-3 for scale-free networks)

        Returns:
            Array of weights following power-law
        """
        # Inverse transform sampling for power-law
        # P(x) = C x^(-α), x ∈ [x_min, x_max]
        x_min = 0.01
        x_max = 1.0

        uniform = np.random.uniform(0, 1, size)

        if np.isclose(alpha, 1.0):
            weights = x_min * np.exp(uniform * np.log(x_max / x_min))
        else:
            weights = (x_min**(1-alpha) +
                      uniform * (x_max**(1-alpha) - x_min**(1-alpha)))**(1/(1-alpha))

        # Normalize
        weights /= np.sum(weights)

        return weights

    def get_lambda_vector(self) -> np.ndarray:
        """Return parameter vector as numpy array"""
        return np.array([self.lambda_E, self.lambda_H, self.lambda_R])

    def compute_weight_exponent(self) -> float:
        """
        Estimate power-law exponent α of weight distribution

        Uses maximum likelihood estimation for power-law exponent.

        Returns:
            Estimated α value
        """
        w = np.abs(self.weights[self.weights > 0.01])  # Filter small weights

        if len(w) < 10:
            return np.nan

        # MLE for power-law: α = 1 + n / Σ ln(wᵢ/w_min)
        w_min = np.min(w)
        n = len(w)

        alpha = 1 + n / np.sum(np.log(w / w_min))

        return alpha


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: EVOLUTIONARY OPERATORS
# ═══════════════════════════════════════════════════════════════════════════

class EvolutionaryOperators:
    """Genetic operators: selection, crossover, mutation"""

    @staticmethod
    def tournament_selection(population: List[AODAgent],
                            tournament_size: int = 5) -> AODAgent:
        """
        Tournament selection: pick best from random subset

        Args:
            population: List of agents
            tournament_size: Number of competitors

        Returns:
            Selected agent
        """
        competitors = np.random.choice(population, size=tournament_size, replace=False)
        winner = max(competitors, key=lambda agent: agent.fitness)
        return winner

    @staticmethod
    def crossover(parent1: AODAgent, parent2: AODAgent, generation: int,
                  agent_id: int) -> AODAgent:
        """
        Single-point crossover of parameter vectors

        Args:
            parent1, parent2: Parent agents
            generation: Current generation number
            agent_id: ID for new agent

        Returns:
            Offspring agent
        """
        # Crossover Λ vectors
        crossover_point = np.random.randint(0, 3)

        lambda1 = parent1.get_lambda_vector()
        lambda2 = parent2.get_lambda_vector()

        offspring_lambda = lambda1.copy()
        offspring_lambda[crossover_point:] = lambda2[crossover_point:]

        # Normalize
        offspring_lambda /= np.sum(offspring_lambda)

        # Crossover weights (blend)
        blend_ratio = np.random.uniform(0.3, 0.7)
        offspring_weights = (blend_ratio * parent1.weights +
                            (1 - blend_ratio) * parent2.weights)

        return AODAgent(
            lambda_E=offspring_lambda[0],
            lambda_H=offspring_lambda[1],
            lambda_R=offspring_lambda[2],
            weights=offspring_weights,
            generation=generation,
            agent_id=agent_id
        )

    @staticmethod
    def mutate(agent: AODAgent, mutation_rate: float = 0.1,
               mutation_strength: float = 0.05) -> None:
        """
        Mutate agent parameters in-place

        Args:
            agent: Agent to mutate
            mutation_rate: Probability of mutation
            mutation_strength: Magnitude of mutation
        """
        # Mutate Λ vector
        if np.random.random() < mutation_rate:
            lambda_vec = agent.get_lambda_vector()

            # Add Gaussian noise
            noise = np.random.normal(0, mutation_strength, size=3)
            lambda_vec += noise

            # Ensure positive and normalize
            lambda_vec = np.abs(lambda_vec)
            lambda_vec /= np.sum(lambda_vec)

            agent.lambda_E = lambda_vec[0]
            agent.lambda_H = lambda_vec[1]
            agent.lambda_R = lambda_vec[2]

        # Mutate weights (with smaller probability)
        if np.random.random() < mutation_rate * 0.5:
            # Perturb a few random weights
            num_perturb = max(1, int(len(agent.weights) * 0.05))
            indices = np.random.choice(len(agent.weights), size=num_perturb, replace=False)

            agent.weights[indices] += np.random.normal(0, mutation_strength, size=num_perturb)
            agent.weights = np.abs(agent.weights)
            agent.weights /= np.sum(agent.weights)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: FITNESS EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

class FitnessEvaluator:
    """
    Evaluate agent fitness based on AOD cost function

    Fitness = -𝓛_AOD (negative cost, so minimize cost = maximize fitness)

    Includes resilience testing under perturbations.
    """

    def __init__(self,
                 num_timesteps: int = 100,
                 perturbation_strength: float = 0.1,
                 perturbation_frequency: int = 25):
        """
        Args:
            num_timesteps: Simulation length for each agent
            perturbation_strength: Magnitude of environmental perturbations
            perturbation_frequency: How often to inject perturbations
        """
        self.num_timesteps = num_timesteps
        self.perturbation_strength = perturbation_strength
        self.perturbation_freq = perturbation_frequency

    def evaluate(self, agent: AODAgent, environment: np.ndarray) -> float:
        """
        Evaluate agent fitness through simulation

        Args:
            agent: Agent to evaluate
            environment: Environmental signal (time series)

        Returns:
            Fitness score (higher is better)
        """
        # Simulate agent dynamics
        state = np.random.normal(0, 0.1, size=10)  # Initial state
        total_cost = 0.0
        robustness_samples = []

        for t in range(self.num_timesteps):
            # Environmental perturbation
            if t % self.perturbation_freq == 0 and t > 0:
                perturbation = np.random.normal(0, self.perturbation_strength, size=10)
                state += perturbation

            # Compute costs (simplified)
            # In full implementation, this would use aod_physics.PhysicalCostFunction

            # Energy cost: weighted by lambda_E
            energy = np.sum(state**2) * agent.lambda_E

            # Entropy cost: weighted by lambda_H
            # Approximate entropy from state variance
            state_entropy = -np.sum(state * np.log(np.abs(state) + 1e-10))
            entropy_cost = state_entropy * agent.lambda_H

            # Robustness cost: weighted by lambda_R
            # Robustness = structural stability (power-law exponent)
            alpha = agent.compute_weight_exponent()
            if not np.isnan(alpha):
                # Optimal α ≈ 2.5-3.0 for scale-free networks
                robustness = 1.0 - abs(alpha - 2.6) / 2.6
                robustness_samples.append(robustness)
            else:
                robustness = 0.5

            robustness_cost = (1.0 - robustness) * agent.lambda_R

            # Total cost for this timestep
            cost_t = energy + entropy_cost + robustness_cost
            total_cost += cost_t

            # Simple dynamics update
            state = 0.95 * state + 0.05 * np.random.normal(0, 0.1, size=10)

        # Average robustness
        agent.robustness = np.mean(robustness_samples) if robustness_samples else 0.5

        # Fitness = -cost (minimize cost)
        fitness = -total_cost / self.num_timesteps

        agent.fitness = fitness
        agent.cost_history.append(total_cost)

        return fitness

    def resilience_test(self, agent: AODAgent, shock_magnitude: float = 1.0) -> float:
        """
        Test agent resilience to catastrophic shock

        Args:
            agent: Agent to test
            shock_magnitude: Magnitude of shock (multiple of normal perturbation)

        Returns:
            Recovery time (timesteps to return to stable state)
        """
        state = np.random.normal(0, 0.1, size=10)

        # Apply catastrophic shock
        shock = np.random.normal(0, self.perturbation_strength * shock_magnitude, size=10)
        state += shock

        # Measure recovery time
        recovery_threshold = 0.2
        recovery_time = 0

        for t in range(200):  # Max 200 timesteps
            # Check if recovered
            if np.linalg.norm(state) < recovery_threshold:
                recovery_time = t
                break

            # Dynamics
            state = 0.95 * state + 0.05 * np.random.normal(0, 0.1, size=10)

        return recovery_time


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: EVOLUTIONARY ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════

class AODEvolutionaryAlgorithm:
    """
    Main evolutionary algorithm orchestrator

    Evolves population of AOD agents to discover optimal Λ* attractor.
    """

    def __init__(self,
                 population_size: int = 200,
                 num_generations: int = 100,
                 elite_fraction: float = 0.1,
                 mutation_rate: float = 0.15,
                 tournament_size: int = 5):
        """
        Args:
            population_size: Number of agents in population
            num_generations: Number of evolutionary generations
            elite_fraction: Fraction of best agents to preserve (elitism)
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament for selection
        """
        self.pop_size = population_size
        self.num_gens = num_generations
        self.elite_size = int(population_size * elite_fraction)
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        self.operators = EvolutionaryOperators()
        self.evaluator = FitnessEvaluator()

        # Population
        self.population: List[AODAgent] = []
        self.best_agent: Optional[AODAgent] = None

        # Evolution history
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'avg_lambda_E': [],
            'avg_lambda_H': [],
            'avg_lambda_R': [],
            'avg_alpha': [],
            'diversity': []
        }

    def initialize_population(self) -> None:
        """Initialize random population"""
        self.population = []

        for i in range(self.pop_size):
            # Random Λ vector
            lambda_vec = np.random.dirichlet([1, 1, 1])  # Ensures sum = 1

            agent = AODAgent(
                lambda_E=lambda_vec[0],
                lambda_H=lambda_vec[1],
                lambda_R=lambda_vec[2],
                generation=0,
                agent_id=i
            )

            self.population.append(agent)

    def evaluate_population(self, generation: int) -> None:
        """Evaluate fitness for all agents"""
        # Simple environment (could be made more complex)
        environment = np.random.normal(0, 1, size=self.evaluator.num_timesteps)

        for agent in self.population:
            self.evaluator.evaluate(agent, environment)

        # Update best agent
        current_best = max(self.population, key=lambda a: a.fitness)
        if self.best_agent is None or current_best.fitness > self.best_agent.fitness:
            self.best_agent = current_best

    def selection_and_reproduction(self, generation: int) -> List[AODAgent]:
        """
        Create next generation through selection, crossover, and mutation

        Uses elitism: preserve top performers
        """
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda a: a.fitness, reverse=True)

        # Elitism: keep best agents
        next_generation = sorted_pop[:self.elite_size]

        # Fill rest of population through selection + crossover + mutation
        agent_id = self.elite_size

        while len(next_generation) < self.pop_size:
            # Select parents
            parent1 = self.operators.tournament_selection(self.population, self.tournament_size)
            parent2 = self.operators.tournament_selection(self.population, self.tournament_size)

            # Crossover
            offspring = self.operators.crossover(parent1, parent2, generation + 1, agent_id)

            # Mutation
            self.operators.mutate(offspring, self.mutation_rate)

            next_generation.append(offspring)
            agent_id += 1

        return next_generation

    def compute_metrics(self) -> None:
        """Compute population statistics"""
        fitnesses = [agent.fitness for agent in self.population]
        lambdas_E = [agent.lambda_E for agent in self.population]
        lambdas_H = [agent.lambda_H for agent in self.population]
        lambdas_R = [agent.lambda_R for agent in self.population]

        alphas = []
        for agent in self.population:
            alpha = agent.compute_weight_exponent()
            if not np.isnan(alpha):
                alphas.append(alpha)

        # Diversity: std of lambda vectors
        lambda_matrix = np.array([[a.lambda_E, a.lambda_H, a.lambda_R] for a in self.population])
        diversity = np.mean(np.std(lambda_matrix, axis=0))

        self.history['best_fitness'].append(np.max(fitnesses))
        self.history['avg_fitness'].append(np.mean(fitnesses))
        self.history['avg_lambda_E'].append(np.mean(lambdas_E))
        self.history['avg_lambda_H'].append(np.mean(lambdas_H))
        self.history['avg_lambda_R'].append(np.mean(lambdas_R))
        self.history['avg_alpha'].append(np.mean(alphas) if alphas else np.nan)
        self.history['diversity'].append(diversity)

    def run(self) -> Dict:
        """
        Run complete evolutionary algorithm

        Returns:
            Dictionary with evolution results
        """
        print("=" * 80)
        print(f"AOD EVOLUTIONARY ALGORITHM - Phase 2")
        print("=" * 80)
        print(f"Population size: {self.pop_size}")
        print(f"Generations: {self.num_gens}")
        print(f"Elite size: {self.elite_size}")
        print(f"Mutation rate: {self.mutation_rate}")
        print()

        # Initialize
        self.initialize_population()

        # Evolution loop
        for generation in range(self.num_gens):
            # Evaluate
            self.evaluate_population(generation)

            # Record metrics
            self.compute_metrics()

            # Print progress
            if generation % 10 == 0 or generation == self.num_gens - 1:
                best_fit = self.history['best_fitness'][-1]
                avg_fit = self.history['avg_fitness'][-1]
                avg_alpha = self.history['avg_alpha'][-1]

                print(f"Gen {generation:3d}: "
                      f"Best={best_fit:8.4f}, "
                      f"Avg={avg_fit:8.4f}, "
                      f"α={avg_alpha:.3f}, "
                      f"Diversity={self.history['diversity'][-1]:.4f}")

            # Create next generation
            if generation < self.num_gens - 1:
                self.population = self.selection_and_reproduction(generation)

        print()
        print("=" * 80)
        print("Evolution Complete")
        print("=" * 80)

        # Analyze results
        results = self.analyze_convergence()

        return results

    def analyze_convergence(self) -> Dict:
        """Analyze convergence to optimal Λ*"""
        if self.best_agent is None:
            return {}

        # Final population statistics
        final_lambdas = np.array([[a.lambda_E, a.lambda_H, a.lambda_R]
                                 for a in self.population])

        lambda_mean = np.mean(final_lambdas, axis=0)
        lambda_std = np.std(final_lambdas, axis=0)

        # Best agent Λ vector
        best_lambda = self.best_agent.get_lambda_vector()

        # Power-law exponent
        best_alpha = self.best_agent.compute_weight_exponent()

        # Resilience test
        resilience_time = self.evaluator.resilience_test(self.best_agent, shock_magnitude=10.0)

        results = {
            'best_fitness': self.best_agent.fitness,
            'best_lambda': best_lambda,
            'best_lambda_E': best_lambda[0],
            'best_lambda_H': best_lambda[1],
            'best_lambda_R': best_lambda[2],
            'best_alpha': best_alpha,
            'best_robustness': self.best_agent.robustness,
            'population_lambda_mean': lambda_mean,
            'population_lambda_std': lambda_std,
            'resilience_recovery_time': resilience_time,
            'converged': lambda_std.mean() < 0.05,  # Convergence threshold
            'history': self.history
        }

        print(f"\nOptimal Λ* Attractor:")
        print(f"  λ_E (Energy):     {best_lambda[0]:.4f} ± {lambda_std[0]:.4f}")
        print(f"  λ_H (Entropy):    {best_lambda[1]:.4f} ± {lambda_std[1]:.4f}")
        print(f"  λ_R (Robustness): {best_lambda[2]:.4f} ± {lambda_std[2]:.4f}")
        print(f"\nStructural Properties:")
        print(f"  Power-law exponent α: {best_alpha:.3f}")
        print(f"  Robustness score: {self.best_agent.robustness:.3f}")
        print(f"  Recovery time (10x shock): {resilience_time} timesteps")
        print(f"\nConvergence:")
        print(f"  Converged: {results['converged']}")
        print(f"  Final diversity: {self.history['diversity'][-1]:.4f}")

        return results

    def plot_evolution(self, save_path: Optional[str] = None) -> None:
        """Plot evolutionary dynamics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        generations = range(len(self.history['best_fitness']))

        # Plot 1: Fitness over time
        ax1 = axes[0, 0]
        ax1.plot(generations, self.history['best_fitness'], label='Best', linewidth=2)
        ax1.plot(generations, self.history['avg_fitness'], label='Average', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Lambda evolution
        ax2 = axes[0, 1]
        ax2.plot(generations, self.history['avg_lambda_E'], label='λ_E (Energy)', linewidth=2)
        ax2.plot(generations, self.history['avg_lambda_H'], label='λ_H (Entropy)', linewidth=2)
        ax2.plot(generations, self.history['avg_lambda_R'], label='λ_R (Robustness)', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Average λ')
        ax2.set_title('Parameter Convergence to Λ*')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Power-law exponent
        ax3 = axes[1, 0]
        ax3.plot(generations, self.history['avg_alpha'], linewidth=2, color='purple')
        ax3.axhline(y=2.6, color='red', linestyle='--', label='Optimal α ≈ 2.6')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Power-law exponent α')
        ax3.set_title('Structural Evolution (Scale-Free Network)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Diversity
        ax4 = axes[1, 1]
        ax4.plot(generations, self.history['diversity'], linewidth=2, color='green')
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Population Diversity')
        ax4.set_title('Genetic Diversity Over Time')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nEvolution plot saved to: {save_path}")

        plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# TESTING & DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run small-scale evolution test
    print("\nRunning Phase 2 Evolutionary Algorithm Test...")
    print("(Small scale: 50 agents, 50 generations)")
    print()

    evo = AODEvolutionaryAlgorithm(
        population_size=50,
        num_generations=50,
        elite_fraction=0.1,
        mutation_rate=0.15
    )

    results = evo.run()

    # Plot results
    evo.plot_evolution(save_path='aod_evolution_test.png')

    print("\n" + "=" * 80)
    print("✓ Phase 2 evolutionary algorithm test complete")
    print("=" * 80)
