# costs

PortPy assumes frictionless trading by default. `turnover_from_weights`/`net_of_costs_returns` layer a single blended cost rate back in; `bid_ask_spreads`, `broker_commissions`, `fx_spread_costs`, and `tax_impact` itemize that cost by source instead - see the [Costs & taxes](../../guide/costs-and-taxes.md) guide for how they compose.

::: portpy.metrics.costs
    options:
      members:
        - turnover_from_weights
        - net_of_costs_returns
        - bid_ask_spreads
        - broker_commissions
        - fx_spread_costs
        - tax_impact
