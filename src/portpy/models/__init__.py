"""
Portfolio construction, optimization, and management.

A namespace of namespaces:
    - .estimators   - inputs to construction/optimization (expected returns,
                      covariance, factor models).
    - .optimization - pure solver functions, (expected_returns, cov_matrix,
                      constraints) -> weights.
    - .construction - constraint objects, plus the optimize/build/efficient_frontier
                      recipe layer over .optimization.
    - .management   - rebalance/monitor/compare, for after a portfolio is built.

See .base for the shared ModelResult type and the constraint objects every one
of the above uses. Research tooling (stochastic simulation, forecasting, ML)
lives under .simulation but is deferred to v0.3.0.

Portfolio.models exposes all of this bound to a portfolio's own data (with
returns/expected_returns/cov_matrix/weights/etc. auto-filled) - see
Portfolio's _AutoFillNamespace in portfolio.py.
"""

from portpy.models import base, construction, estimators, management, optimization
from portpy.models.base import GrossExposure, GroupCap, ModelResult, NetExposure, TurnoverCap, WeightBounds
from portpy.models.construction.builders import build, efficient_frontier, optimize

__all__ = [
    "base",
    "estimators",
    "optimization",
    "construction",
    "management",
    "ModelResult",
    "WeightBounds",
    "GroupCap",
    "TurnoverCap",
    "NetExposure",
    "GrossExposure",
    "optimize",
    "build",
    "efficient_frontier",
]
