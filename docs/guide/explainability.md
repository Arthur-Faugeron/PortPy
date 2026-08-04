# Explainability

`portpy.explain()` is what turns PortPy from a "numbers out" library into a teaching tool.
Every metric can render a plain-language card describing itself, and every result can carry
its own name so `.explain()` works without you having to remember what produced it.

## Three pieces

**`Explanation`** — a small, frozen dataclass: `summary`, `formula`, `how_to_read`,
`good_vs_bad`, `caveats`, and an optional `interpret` function mapping a live value to a
one-line verdict.

**A registry** — a plain `dict[str, Explanation]`, populated by `register(Explanation(...))`
calls sitting at the bottom of every metrics module. The text is hand-written prose placed 
into the source at import time — there's no dynamic generation, no docstring parsing, 
no LLM in the loop. For example, in
[`returns.py`](https://github.com/Arthur-Faugeron/PortPy/blob/main/src/portpy/metrics/returns.py):

```python
register(
    Explanation(
        name="total_return",
        category="metric",
        summary="The compounded percentage gain or loss over the entire period...",
        formula="prod(1 + r_t) - 1",
        how_to_read="Read directly as a percentage: 0.35 means +35% over the whole period.",
        ...
        interpret=lambda v: f"{v:+.1%} total over the period" + (" (a net loss)" if v < 0 else ""),
    )
)
```

**`MetricResult`** — a `float` subclass returned when you pass `as_result=True`. It behaves
exactly like a plain float everywhere (arithmetic, comparisons, formatting, numpy/pandas
interop), but also carries the metric's registered `name` and an `.explain()` method.

## Using it

```python
sharpe = portfolio.metrics.sharpe_ratio(as_result=True)

float(sharpe)           # 0.74 - just a number
sharpe.interpretation   # "sub-par" - the one-line verdict, computed from the live value
sharpe.explain()        # the full card, printed
```

`portpy.explain(...)` is the single dispatch point, and accepts several kinds of input:

```python
import portpy

portpy.explain("sharpe_ratio")      # look up by registered name directly
portpy.explain(sharpe)              # a MetricResult - uses its .name and its own value
portpy.explain(comparison_df)       # any object with `.attrs["portpy_explanation"]`,
                                    # e.g. the DataFrame returned by compare_to_benchmark()
```

!!! note 

"Importing `portpy.explain`"
`from portpy import Portfolio` also binds the name `explain` on the `portpy` package to
the **function**, not the module (see `portpy/__init__.py`'s
`from portpy.explain import explain`). To reach the registry helpers directly
(`available()`, `get()`, `register()`, `Explanation`, `MetricResult`), import them from
the submodule explicitly:
```python
from portpy.explain import available, get, MetricResult
```

## Listing what's registered

```python
from portpy.explain import available

available("metric")    # every registered metric name
available("chart")     # rolling/other chart-shaped explanations (drawdown_chart, etc.)
```

## Extending it

Any function can register a card — this isn't limited to `portpy.metrics`. Future
`portpy.visualization`/`portpy.models`/`portpy.strategies` code is expected to register
`Explanation`s the same way, tagged `category="chart"`, `"model"`, or `"strategy"`.
