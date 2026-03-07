"""Phase 5 optimization exports.

Lazy exports are used so importing `src.optimization` does not eagerly import
`optimizer.py`. This avoids duplicate-module warnings when running:
`python -m src.optimization.optimizer`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = ["NSGA2Optimizer", "OptimizationResult", "run_optimization_phase"]


def __getattr__(name: str) -> Any:
	if name in __all__:
		module = import_module("src.optimization.optimizer")
		return getattr(module, name)
	raise AttributeError(f"module 'src.optimization' has no attribute '{name}'")
