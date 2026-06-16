"""Pluggable optimizer strategies for layer assignment.

REFACTOR_GOALS.md M-J Stage 1 adds ``ga_pymoo`` (pymoo GA); Stage 2 adds
``pso_pyswarms`` (pyswarms PSO), ``cmaes`` (Hansen CMA-ES) and
``bo_skopt`` (scikit-optimize Bayesian Optimization). Tabu Search is
deferred (§7 Q-f open question).
"""
from __future__ import annotations

from .base import OptimizationResult, Optimizer, get_optimizer, register_optimizer
from .bo_skopt import BOSkoptOptimizer
from .cmaes import CMAESOptimizer
from .differential_evolution import DifferentialEvolutionOptimizer
from .dual_annealing import DualAnnealingOptimizer
from .dual_annealing_with_swap_polish import DualAnnealingWithSwapPolishOptimizer
from .ga_pymoo import GAPymooOptimizer
from .pso_pyswarms import PSOPyswarmsOptimizer

__all__ = [
    "Optimizer",
    "OptimizationResult",
    "get_optimizer",
    "register_optimizer",
    "DualAnnealingOptimizer",
    "DualAnnealingWithSwapPolishOptimizer",
    "DifferentialEvolutionOptimizer",
    "GAPymooOptimizer",
    "PSOPyswarmsOptimizer",
    "CMAESOptimizer",
    "BOSkoptOptimizer",
]
