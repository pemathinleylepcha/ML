from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class GAResult:
    best_genome: list[int]
    best_score: float
    history: list[float]


def run_binary_ga(
    n_genes: int,
    scorer: Callable[[list[int]], float],
    population_size: int = 8,
    generations: int = 3,
    mutation_rate: float = 0.15,
    crossover_rate: float = 0.50,
    seed: int = 7,
) -> GAResult:
    rng = random.Random(seed)

    def _mutate(genome: list[int]) -> list[int]:
        child = genome[:]
        for idx in range(len(child)):
            if rng.random() < mutation_rate:
                child[idx] = 1 - child[idx]
        if sum(child) == 0:
            child[rng.randrange(len(child))] = 1
        return child

    def _crossover(left: list[int], right: list[int]) -> tuple[list[int], list[int]]:
        if len(left) != len(right) or len(left) == 0 or rng.random() >= crossover_rate:
            return left[:], right[:]
        cut = rng.randrange(1, len(left))
        return left[:cut] + right[cut:], right[:cut] + left[cut:]

    population = []
    for _ in range(population_size):
        genome = [rng.randint(0, 1) for _ in range(n_genes)]
        if sum(genome) == 0:
            genome[rng.randrange(n_genes)] = 1
        population.append(genome)

    best_genome = population[0][:]
    best_score = float("-inf")
    history: list[float] = []
    for _ in range(generations):
        scored = [(scorer(genome), genome) for genome in population]
        scored.sort(key=lambda item: item[0], reverse=True)
        history.append(float(scored[0][0]))
        if scored[0][0] > best_score:
            best_score = float(scored[0][0])
            best_genome = scored[0][1][:]
        elites = [genome[:] for _, genome in scored[: max(2, population_size // 3)]]
        next_population = elites[:]
        while len(next_population) < population_size:
            left = rng.choice(elites)
            right = rng.choice(elites)
            child_a, child_b = _crossover(left, right)
            next_population.append(_mutate(child_a))
            if len(next_population) < population_size:
                next_population.append(_mutate(child_b))
        population = next_population[:population_size]

    return GAResult(best_genome=best_genome, best_score=best_score, history=history)


@dataclass(slots=True)
class ContinuousGAResult:
    best_genome: list[float]
    best_score: float
    history: list[float]


def run_continuous_ga(
    n_genes: int,
    scorer: Callable[[list[float]], float],
    population_size: int = 30,
    generations: int = 10,
    mutation_rate: float = 0.15,
    crossover_rate: float = 0.50,
    mutation_scale: float = 0.15,
    seed: int = 7,
) -> ContinuousGAResult:
    rng = random.Random(seed)

    def _mutate(genome: list[float]) -> list[float]:
        child = genome[:]
        for idx in range(len(child)):
            if rng.random() < mutation_rate:
                child[idx] = max(0.0, min(1.0, child[idx] + rng.gauss(0.0, mutation_scale)))
        return child

    def _crossover(left: list[float], right: list[float]) -> tuple[list[float], list[float]]:
        if len(left) != len(right) or len(left) == 0 or rng.random() >= crossover_rate:
            return left[:], right[:]
        cut = rng.randrange(1, len(left))
        return left[:cut] + right[cut:], right[:cut] + left[cut:]

    population: list[list[float]] = []
    for _ in range(population_size):
        population.append([rng.random() for _ in range(n_genes)])

    best_genome = population[0][:]
    best_score = float("-inf")
    history: list[float] = []

    for _ in range(generations):
        scored = [(scorer(genome), genome) for genome in population]
        scored.sort(key=lambda item: item[0], reverse=True)
        history.append(float(scored[0][0]))
        if scored[0][0] > best_score:
            best_score = float(scored[0][0])
            best_genome = scored[0][1][:]
        elite_count = max(2, population_size // 3)
        elites = [genome[:] for _, genome in scored[:elite_count]]
        next_population = elites[:]
        while len(next_population) < population_size:
            left = rng.choice(elites)
            right = rng.choice(elites)
            child_a, child_b = _crossover(left, right)
            next_population.append(_mutate(child_a))
            if len(next_population) < population_size:
                next_population.append(_mutate(child_b))
        population = next_population[:population_size]

    return ContinuousGAResult(best_genome=best_genome, best_score=best_score, history=history)
