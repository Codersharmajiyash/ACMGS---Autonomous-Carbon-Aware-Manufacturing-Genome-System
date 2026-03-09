# Phase 5 Optimizer Notes

This module is intentionally runnable before Phase 4 is ready.

## Run Phase 5 now

```powershell
python -m src.optimization.optimizer
```

It currently uses `demo_evaluator()` in `src/optimization/optimizer.py`.

## Teammate handoff (when Phase 4 is ready)

1. Keep `NSGA2Optimizer` unchanged.
2. Add an evaluator function that calls Phase 4 prediction logic and returns:
   - `yield`
   - `quality`
   - `energy_consumption`
   - `carbon_intensity`
3. Replace the call in `run_optimization_phase()`:

```python
optimizer = NSGA2Optimizer(evaluator=demo_evaluator)
```

with:

```python
optimizer = NSGA2Optimizer(evaluator=phase4_evaluator)
```

That is the only required integration point.
