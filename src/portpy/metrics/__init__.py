"""
Pure, stateless metric functions, grouped by topic.

Every public function here also works standalone (pass a plain pandas Series/
DataFrame) - :class:`~portpy.portfolio.Portfolio` methods are thin wrappers
around these.
"""

from portpy.metrics import (
    benchmarks,
    costs,
    covariance,
    distributions,
    drawdowns,
    performance,
    regressions,
    returns,
    risk,
    rolling,
    summary,
)
from portpy.metrics.benchmarks import *  # noqa: F401,F403
from portpy.metrics.costs import *  # noqa: F401,F403
from portpy.metrics.covariance import *  # noqa: F401,F403
from portpy.metrics.distributions import *  # noqa: F401,F403
from portpy.metrics.drawdowns import *  # noqa: F401,F403
from portpy.metrics.performance import *  # noqa: F401,F403
from portpy.metrics.regressions import *  # noqa: F401,F403
from portpy.metrics.returns import *  # noqa: F401,F403
from portpy.metrics.risk import *  # noqa: F401,F403
from portpy.metrics.rolling import *  # noqa: F401,F403
from portpy.metrics.summary import *  # noqa: F401,F403

__all__ = [
    "benchmarks",
    "costs",
    "covariance",
    "distributions",
    "drawdowns",
    "performance",
    "regressions",
    "returns",
    "risk",
    "rolling",
    "summary",
    *benchmarks.__all__,
    *costs.__all__,
    *covariance.__all__,
    *distributions.__all__,
    *drawdowns.__all__,
    *performance.__all__,
    *regressions.__all__,
    *returns.__all__,
    *risk.__all__,
    *rolling.__all__,
    *summary.__all__,
]
