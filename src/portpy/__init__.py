"""
PortPy: portfolio analysis, optimization, and management.

The main entry point is :class:`Portfolio`. Every metric/chart/model/strategy
can explain itself in plain language - call `.explain()` on a result, or
`portpy.explain("name")` directly.
"""

from __future__ import annotations

from typing import Any

import portpy.explain as explain_module
from portpy import core, metrics
from portpy.explain import Explanation, MetricResult, available, get, register
from portpy.portfolio import Portfolio

__version__ = "0.1.1"


class _ExplainAPI:
    """
    Callable explain facade that also exposes registry helpers.
    """

    def __call__(self, obj: Any, *, value: Any = None, print_it: bool = True) -> str:
        return explain_module.explain(obj, value=value, print_it=print_it)

    def available(self, category: str | None = None) -> list[str]:
        return explain_module.available(category)

    def get(self, name: str) -> Explanation:
        return explain_module.get(name)

    def register(self, explanation: Explanation) -> Explanation:
        return explain_module.register(explanation)


explain = _ExplainAPI()

__all__ = [
    "Portfolio",
    "explain",
    "Explanation",
    "MetricResult",
    "available",
    "get",
    "register",
    "core",
    "metrics",
    "__version__",
]
