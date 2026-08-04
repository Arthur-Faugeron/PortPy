"""PortPy: portfolio analysis, optimization, and management.

The main entry point is :class:`Portfolio`. Every metric/chart/model/strategy
can explain itself in plain language - call `.explain()` on a result, or
`portpy.explain("name")` directly.
"""

from portpy import core, metrics
from portpy.explain import explain
from portpy.portfolio import Portfolio

__version__ = "0.1.0"

__all__ = ["Portfolio", "explain", "core", "metrics", "__version__"]
