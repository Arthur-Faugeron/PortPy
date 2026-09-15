# base

Shared plumbing every other `portpy.models` submodule builds on: the `ModelResult` type every builder/optimizer/rebalancer returns, the constraint objects `.optimization` solvers understand, and the multi-start SLSQP solve routine (`solve_weights`) they're all built on. See [Construction, optimization & management](../../guide/models.md) for how these fit together.

## ModelResult

::: portpy.models.base.ModelResult

## Constraint objects

::: portpy.models.base
    options:
      members:
        - WeightBounds
        - GroupCap
        - TurnoverCap
        - NetExposure
        - GrossExposure

## Solver plumbing

::: portpy.models.base.solve_weights
