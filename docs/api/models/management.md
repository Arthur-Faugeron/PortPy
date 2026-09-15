# management

Post-construction portfolio management: rebalancing trade lists, drift/limit monitoring, and before/after or portfolio-vs-portfolio comparisons. `Portfolio` itself assumes frictionless, always-rebalanced-to-target trading - this is where the turn-that-assumption-off tools live.

::: portpy.models.management.rebalancing
    options:
      members:
        - rebalance
        - monitor
        - compare
