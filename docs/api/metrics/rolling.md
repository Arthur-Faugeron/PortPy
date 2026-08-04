# rolling

`rolling_metric` / `expanding_metric` are generic engines — they roll *any* PortPy metric
function (or your own) over a window. The rest are convenience wrappers over the same engine.

::: portpy.metrics.rolling
    options:
      members:
        - rolling_metric
        - rolling_sharpe
        - rolling_volatility
        - rolling_beta
        - rolling_correlation
        - expanding_metric
