# drawdowns

All functions here take a price series (or any cumulative-value series) - not a returns
series. Use Portfolio.price_index() or portpy.metrics.returns.prices_from_returns if you
only have returns.

::: portpy.metrics.drawdowns
    options:
      members:
        - drawdown_series
        - max_drawdown
        - drawdown_duration
        - time_to_recovery
        - top_n_drawdowns
        - average_drawdown
        - drawdown_at_risk
