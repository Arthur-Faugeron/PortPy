"""
Pure solver functions: (expected_returns, cov_matrix, constraints) -> weights.

Every function here is importable directly, with no dependency on Portfolio -
see portpy.models.construction.optimize for the discoverable, ModelResult-returning
dispatcher over these, and portpy.models.estimators for computing expected_returns/cov_matrix.
"""

from portpy.models.optimization import solvers
from portpy.models.optimization.solvers import *  # noqa: F401,F403

__all__ = [
    "solvers",
    *solvers.__all__,
]
