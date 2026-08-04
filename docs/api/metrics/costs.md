# costs


PortPy assumes frictionless trading by default. These helpers layer transaction costs back
in, either from an explicit weight-history (real turnover) or an assumed constant per-period
turnover.


::: portpy.metrics.costs
    options:
      members:
        - turnover_from_weights
        - net_of_costs_returns
