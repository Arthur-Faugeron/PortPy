"""PortPy's built-in explainability layer.

This is what turns PortPy from a "numbers and charts out" library into a teaching
tool: every metric, chart, model, and strategy can tell you, in plain language,
what it is, how to read it, and whether the number you got is good or bad.

Three pieces work together:

- :class:`Explanation` - a structured knowledge card (what it is / how to read it /
  good vs. bad / caveats / formula), optionally with a function that turns a
  concrete computed value into a one-line verdict.
- A module-level registry mapping names (``"sharpe_ratio"``, ``"drawdown_chart"``,
  ``"capm"``, ``"buy_and_hold"``, ...) to their :class:`Explanation`.
- :class:`MetricResult` - a ``float`` subclass returned by metric functions when
  ``as_result=True``. It behaves exactly like a float everywhere (math, comparisons,
  formatting, numpy/pandas operations) but also carries a ``.explain()`` method and
  a ``.interpretation`` property.

The same registry backs charts (``fig.explain()`` in :mod:`portpy.visualization`),
models, and strategies (``.describe()`` / ``.diagnose()`` in :mod:`portpy.models`
and :mod:`portpy.strategies`), so ``portpy.explain(x)`` works uniformly on any of
them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Explanation",
    "MetricResult",
    "register",
    "get",
    "explain",
    "available",
]

_CATEGORIES = ("metric", "chart", "model", "strategy")


@dataclass(frozen=True)
class Explanation:
    """A structured, human-readable knowledge card attached to a PortPy object.

    Attributes:
        name: Registry key, matching the function/chart/model/strategy name
            (e.g. ``"sharpe_ratio"``, ``"drawdown_chart"``, ``"capm"``).
        category: One of ``"metric"``, ``"chart"``, ``"model"``, ``"strategy"``.
        summary: One or two sentences on what this *is*.
        how_to_read: How to read the number/axis/output in practice.
        good_vs_bad: Rules of thumb for judging whether a value is good or bad.
        caveats: Known limitations, edge cases, or ways this can mislead.
        formula: Optional plain-text formula for reference.
        interpret: Optional function mapping a concrete value to a one-line verdict,
            used to render a "this result" line when a value is available.
    """

    name: str
    category: str
    summary: str
    how_to_read: str
    good_vs_bad: str
    caveats: str = ""
    formula: str = ""
    interpret: Callable[[Any], str] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.category not in _CATEGORIES:
            raise ValueError(f"category must be one of {_CATEGORIES}, got {self.category!r}")

    def render(self, value: Any = None) -> str:
        """Render this card as plain text, optionally with a value-specific verdict."""
        lines = [f"{self.name} ({self.category})", "=" * len(f"{self.name} ({self.category})")]
        lines += ["", "What it is:", f"  {self.summary}"]
        if self.formula:
            lines += ["", "Formula:", f"  {self.formula}"]
        lines += ["", "How to read it:", f"  {self.how_to_read}"]
        lines += ["", "Good vs. bad:", f"  {self.good_vs_bad}"]
        if self.caveats:
            lines += ["", "Caveats:", f"  {self.caveats}"]
        if value is not None and self.interpret is not None:
            try:
                lines += ["", "This result:", f"  {self.interpret(value)}"]
            except Exception:
                pass
        return "\n".join(lines)


_REGISTRY: dict[str, Explanation] = {}


def register(explanation: Explanation) -> Explanation:
    """Register (or overwrite) an :class:`Explanation` in the global registry."""
    _REGISTRY[explanation.name] = explanation
    return explanation


def get(name: str) -> Explanation:
    """Look up a registered :class:`Explanation` by name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"No explanation registered for {name!r}. "
            f"Call portpy.explain.available() to list all {len(_REGISTRY)} registered names."
        ) from exc


def available(category: str | None = None) -> list[str]:
    """List registered explanation names, optionally filtered by category."""
    if category is None:
        return sorted(_REGISTRY)
    return sorted(k for k, v in _REGISTRY.items() if v.category == category)


def explain(obj: Any, *, value: Any = None, print_it: bool = True) -> str:
    """Explain a metric name, :class:`MetricResult`, chart, model, or strategy.

    Args:
        obj: A registered name (``str``), a :class:`MetricResult`, or any PortPy
            object that exposes a ``_portpy_explain_name`` attribute (charts, fitted
            models, strategies) or a pandas ``.attrs["portpy_explanation"]`` entry.
        value: Optional concrete value to generate a one-line verdict for. Inferred
            automatically for :class:`MetricResult` and objects carrying
            ``_portpy_explain_value``.
        print_it: If True (default), also ``print()`` the rendered text.

    Returns:
        The rendered explanation text.
    """
    name: str | None = None

    if isinstance(obj, str):
        name = obj
    elif isinstance(obj, MetricResult):
        name = obj.name
        if value is None:
            value = float(obj)
    elif hasattr(obj, "_portpy_explain_name"):
        name = obj._portpy_explain_name
        if value is None:
            value = getattr(obj, "_portpy_explain_value", None)
    else:
        attrs = getattr(obj, "attrs", None)
        if isinstance(attrs, dict) and "portpy_explanation" in attrs:
            name = attrs["portpy_explanation"]

    if name is None:
        raise TypeError(
            f"Don't know how to explain an object of type {type(obj).__name__!r}. "
            "Pass a registered name (str), a MetricResult, or a PortPy chart/model/strategy."
        )

    text = get(name).render(value)
    if print_it:
        print(text)
    return text


class MetricResult(float):
    """A ``float`` that also knows what it means.

    Arithmetic, comparisons, ``round()``, string formatting, and numpy/pandas
    interop all work exactly as they would on a plain float (this *is* a float
    subclass). On top of that, it carries the metric's registered ``name`` and can
    render a full explanation on demand.

    Note:
        ``str(result)`` prints the plain number (so ``print(sharpe)`` stays clean);
        ``repr(result)`` - what you see when a bare expression is evaluated in a
        REPL/notebook - includes the name and a one-line interpretation.
    """

    name: str
    unit: str | None
    meta: dict

    def __new__(
        cls,
        value: float,
        name: str,
        unit: str | None = None,
        meta: dict | None = None,
    ) -> MetricResult:
        obj = super().__new__(cls, value)
        obj.name = name
        obj.unit = unit
        obj.meta = meta or {}
        return obj

    @property
    def value(self) -> float:
        return float(self)

    @property
    def interpretation(self) -> str:
        """A one-line, value-specific verdict (empty string if none is registered)."""
        expl = _REGISTRY.get(self.name)
        if expl is None or expl.interpret is None:
            return ""
        try:
            return expl.interpret(float(self))
        except Exception:
            return ""

    def explain(self, print_it: bool = True) -> str:
        """Print (by default) and return the full explanation for this result."""
        return explain(self, print_it=print_it)

    def __str__(self) -> str:
        return str(float(self))

    def __repr__(self) -> str:
        base = f"{float(self):.6g}"
        unit_str = f" {self.unit}" if self.unit else ""
        interp = self.interpretation
        suffix = f" - {interp}" if interp else ""
        return f"{self.name}={base}{unit_str}{suffix}"
