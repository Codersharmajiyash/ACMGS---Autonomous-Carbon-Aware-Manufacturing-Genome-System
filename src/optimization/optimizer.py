"""
Phase 5: NSGA-II evolutionary optimizer.

This module is intentionally self-contained so Phase 5 can run independently.
It searches for process settings that jointly optimize:
1) max yield
2) max quality
3) min energy consumption
4) min carbon impact
"""

from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from deap import base, creator, tools

from config.settings import (
    OPT_CROSSOVER_PROB,
    OPT_MUTATION_PROB,
    OPT_NUM_GENERATIONS,
    OPT_POPULATION_SIZE,
    SIM_RANDOM_SEED,
    SIMULATED_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    GENOME_PROCESS_FEATURES,
    GENOME_MATERIAL_FEATURES,
)
from src.utils.logger import get_logger


logger = get_logger("optimization")

# Search-space bounds for the 5 process genes used in the project.
DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "temperature": (150.0, 350.0),
    "pressure": (1.0, 10.0),
    "speed": (500.0, 3000.0),
    "feed_rate": (0.1, 2.0),
    "humidity": (20.0, 80.0),
}


@dataclass
class OptimizationResult:
    """Container for all optimization outputs."""

    pareto_df: pd.DataFrame
    population_df: pd.DataFrame


class NSGA2Optimizer:
    """Phase 5 NSGA-II optimizer for carbon-aware manufacturing trade-offs."""

    def __init__(
        self,
        evaluator: Callable[[Dict[str, float]], Dict[str, float]],
        bounds: Dict[str, Tuple[float, float]] | None = None,
        population_size: int = OPT_POPULATION_SIZE,
        num_generations: int = OPT_NUM_GENERATIONS,
        crossover_prob: float = OPT_CROSSOVER_PROB,
        mutation_prob: float = OPT_MUTATION_PROB,
        random_seed: int = SIM_RANDOM_SEED,
    ) -> None:
        """
        Parameters
        ----------
        evaluator:
            Function that maps decision variables to objective predictions.
            Expected keys in return dict: yield, quality, energy_consumption, carbon_intensity
        bounds:
            Variable bounds by feature name. Defaults to the project process parameters.
        """
        self.evaluator = evaluator
        self.bounds = bounds or DEFAULT_BOUNDS
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.random_seed = random_seed

        self.feature_names = list(self.bounds.keys())
        self.low_bounds = [self.bounds[name][0] for name in self.feature_names]
        self.high_bounds = [self.bounds[name][1] for name in self.feature_names]

        random.seed(self.random_seed)
        self._toolbox = self._build_toolbox()

    def _build_toolbox(self) -> base.Toolbox:
        """Create DEAP toolbox with NSGA-II operators and bounded mutation."""
        # Avoid duplicate global class registration if this module is reloaded.
        if not hasattr(creator, "FitnessManufacturing"):
            # Maximize yield + quality, minimize energy + carbon.
            creator.create("FitnessManufacturing", base.Fitness, weights=(1.0, 1.0, -1.0, -1.0))
        if not hasattr(creator, "IndividualManufacturing"):
            creator.create("IndividualManufacturing", list, fitness=creator.FitnessManufacturing)

        toolbox = base.Toolbox()

        toolbox.register("individual", self._create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_individual)

        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=self.low_bounds, up=self.high_bounds, eta=20.0)
        toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            low=self.low_bounds,
            up=self.high_bounds,
            eta=20.0,
            indpb=1.0 / max(1, len(self.feature_names)),
        )
        toolbox.register("select", tools.selNSGA2)

        return toolbox

    def _create_individual(self):
        values = [
            random.uniform(self.bounds[name][0], self.bounds[name][1])
            for name in self.feature_names
        ]
        return creator.IndividualManufacturing(values)

    def _evaluate_individual(self, individual: Sequence[float]) -> Tuple[float, float, float, float]:
        params = {
            name: float(value)
            for name, value in zip(self.feature_names, individual)
        }
        prediction = self.evaluator(params)

        return (
            float(prediction["yield"]),
            float(prediction["quality"]),
            float(prediction["energy_consumption"]),
            float(prediction["carbon_intensity"]),
        )

    def optimize(self) -> OptimizationResult:
        """Run NSGA-II and return full population + Pareto front as DataFrames."""
        logger.info("=" * 60)
        logger.info("PHASE 5: NSGA-II OPTIMIZATION STARTING")
        logger.info("Population=%s | Generations=%s", self.population_size, self.num_generations)

        population = self._toolbox.population(n=self.population_size)

        # Evaluate initial population
        invalid = [ind for ind in population if not ind.fitness.valid]
        fitnesses = list(map(self._toolbox.evaluate, invalid))
        for individual, fit in zip(invalid, fitnesses):
            individual.fitness.values = fit

        # Assign crowding distance in generation 0
        population = self._toolbox.select(population, len(population))

        for generation in range(1, self.num_generations + 1):
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = [self._toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.crossover_prob:
                    self._toolbox.mate(ind1, ind2)
                    del ind1.fitness.values
                    del ind2.fitness.values

            for individual in offspring:
                if random.random() <= self.mutation_prob:
                    self._toolbox.mutate(individual)
                    del individual.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self._toolbox.evaluate, invalid))
            for individual, fit in zip(invalid, fitnesses):
                individual.fitness.values = fit

            population = self._toolbox.select(population + offspring, self.population_size)

            if generation == 1 or generation % 10 == 0 or generation == self.num_generations:
                logger.info("Generation %s/%s complete", generation, self.num_generations)

        pareto_front = tools.sortNondominated(population, k=len(population), first_front_only=True)[0]

        population_df = self._individuals_to_dataframe(population)
        pareto_df = self._individuals_to_dataframe(pareto_front)

        logger.info("Pareto solutions found: %s", len(pareto_df))
        logger.info("PHASE 5: COMPLETE")
        logger.info("=" * 60)

        return OptimizationResult(pareto_df=pareto_df, population_df=population_df)

    def _individuals_to_dataframe(self, individuals: List[Sequence[float]]) -> pd.DataFrame:
        rows = []
        for ind in individuals:
            row = {
                name: float(value)
                for name, value in zip(self.feature_names, ind)
            }
            row["yield"] = float(ind.fitness.values[0])
            row["quality"] = float(ind.fitness.values[1])
            row["energy_consumption"] = float(ind.fitness.values[2])
            row["carbon_intensity"] = float(ind.fitness.values[3])
            rows.append(row)
        return pd.DataFrame(rows)


def demo_evaluator(params: Dict[str, float]) -> Dict[str, float]:
    """
    Simple deterministic evaluator for standalone Phase 5 testing.

    Replace this with Phase 4 model inference once predictor.py is ready.
    """
    temperature = params["temperature"]
    pressure = params["pressure"]
    speed = params["speed"]
    feed_rate = params["feed_rate"]
    humidity = params["humidity"]

    yield_pred = 0.55 + 0.0012 * temperature + 0.018 * pressure - 0.00007 * speed
    quality_pred = 0.35 + 0.0045 * (100.0 - abs(humidity - 50.0)) + 0.06 * (pressure / 10.0)
    energy_pred = 40.0 + 0.11 * speed + 18.0 * pressure + 70.0 * feed_rate
    carbon_pred = energy_pred * (0.18 + 0.0006 * humidity)

    return {
        "yield": max(0.5, min(1.0, yield_pred)),
        "quality": max(0.3, min(1.0, quality_pred)),
        "energy_consumption": max(50.0, energy_pred),
        "carbon_intensity": max(10.0, carbon_pred),
    }


def build_predictor_evaluator() -> Callable[[Dict[str, float]], Dict[str, float]]:
    """Load Phase 4 predictor and return an evaluator function for NSGA-II.

    Falls back to demo_evaluator if Phase 4 model is not yet available.
    """
    model_path = os.path.join(MODELS_DIR, "predictor.pkl")
    genome_path = os.path.join(PROCESSED_DIR, "genome_vectors.npy")

    if not os.path.exists(model_path) or not os.path.exists(genome_path):
        logger.warning("Phase 4 predictor or genome not found — using demo evaluator")
        return demo_evaluator

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    genome_mean = np.load(genome_path).mean(axis=0)  # shape (25,) — normalized

    # Load normalization params so we can properly normalize process features
    norm_path = os.path.join(PROCESSED_DIR, "genome_normalization.npz")
    if os.path.exists(norm_path):
        norm = np.load(norm_path)
        norm_mean = norm["mean"]   # shape (25,)
        norm_std = norm["std"]     # shape (25,)
        carbon_intensity_raw = float(norm_mean[24])
    else:
        norm_mean = None
        norm_std = None
        carbon_intensity_raw = 300.0

    logger.info("Phase 4 predictor loaded for optimization")
    logger.info("Carbon intensity reference: %.2f gCO2/kWh", carbon_intensity_raw)

    def predictor_evaluator(params: Dict[str, float]) -> Dict[str, float]:
        genome_vec = genome_mean.copy()
        # Inject process parameters, properly normalized to z-score space
        for i, name in enumerate(GENOME_PROCESS_FEATURES):
            raw_val = params[name]
            if norm_mean is not None and norm_std is not None and norm_std[i] > 0:
                genome_vec[i] = (raw_val - norm_mean[i]) / norm_std[i]
            else:
                genome_vec[i] = raw_val
        prediction = model.predict(genome_vec.reshape(1, -1))[0]  # shape (3,)
        # Carbon footprint = predicted energy (kWh) * grid carbon intensity (gCO2/kWh) / 1000 -> kg CO2
        carbon_footprint = float(prediction[2]) * (carbon_intensity_raw / 1000.0)
        return {
            "yield": float(prediction[0]),
            "quality": float(prediction[1]),
            "energy_consumption": float(prediction[2]),
            "carbon_intensity": carbon_footprint,
        }

    return predictor_evaluator


def run_optimization_phase() -> OptimizationResult:
    """Convenience entrypoint for Phase 5 execution."""
    evaluator = build_predictor_evaluator()
    optimizer = NSGA2Optimizer(evaluator=evaluator)
    result = optimizer.optimize()

    # Save Pareto solutions to disk — Phase 6 depends on this file
    os.makedirs(SIMULATED_DIR, exist_ok=True)
    pareto_df = result.pareto_df.copy()

    # Add material feature mean values and carbon_intensity from genome reference
    genome_path = os.path.join(PROCESSED_DIR, "genome_vectors.npy")
    if os.path.exists(genome_path):
        genome_mean = np.load(genome_path).mean(axis=0)  # normalized mean
        # Denormalize material features using normalization params
        norm_path = os.path.join(PROCESSED_DIR, "genome_normalization.npz")
        if os.path.exists(norm_path):
            norm = np.load(norm_path)
            for i, col in enumerate(GENOME_MATERIAL_FEATURES):
                pareto_df[col] = float(norm["mean"][5 + i])
        else:
            for i, col in enumerate(GENOME_MATERIAL_FEATURES):
                pareto_df[col] = genome_mean[5 + i]
    else:
        logger.warning("genome_vectors.npy not found — material cols won't be in pareto output")

    # Rename columns to match Phase 6 scheduler expectations
    pareto_df = pareto_df.rename(columns={
        "yield": "pred_yield",
        "quality": "pred_quality",
        "energy_consumption": "pred_energy",
        "carbon_intensity": "pred_carbon",
    })

    pareto_path = os.path.join(SIMULATED_DIR, "pareto_solutions.csv")
    pareto_df.to_csv(pareto_path, index=False)
    logger.info("Pareto solutions saved → %s (%s solutions)", pareto_path, len(pareto_df))

    return result


if __name__ == "__main__":
    result = run_optimization_phase()
    logger.info("Top Pareto solutions (first 5):")
    logger.info("\n%s", result.pareto_df.head().to_string())
