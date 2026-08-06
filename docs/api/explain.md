# explain

See 
[Explainability](https://github.com/Arthur-Faugeron/PortPy/blob/main/docs/guide/explainability.md) 
for how the pieces below fit together, and
where the explanation text itself actually lives (hint: not in this module).

!!! note

from portpy import Portfolio (or any import of the portpy package) binds the package
attribute portpy.explain to the function below, not this module. Import the other
names (available, get, register, Explanation, MetricResult) directly from
portpy.explain as shown in each signature.

::: portpy.explain.Explanation

::: portpy.explain.MetricResult

::: portpy.explain.explain

::: portpy.explain.register

::: portpy.explain.get

::: portpy.explain.available
