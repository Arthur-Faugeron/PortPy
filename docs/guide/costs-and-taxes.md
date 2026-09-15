# Costs & Taxes

PortPy's `.returns()` is frictionless by default: no spreads, no commissions, no FX cost, no tax. That's deliberate - it isolates the strategy's raw signal from implementation drag. `metrics.costs` is where you layer the drag back in, once you want to know what a strategy actually keeps.

There are two ways to do that: one blended rate, or an itemized breakdown by source.

## One blended rate

If you just want a quick, single-number estimate of trading friction:

```python
turnover = portfolio.metrics.turnover_from_weights(weights_history=weights_over_time)
net = portfolio.metrics.net_of_costs_returns(turnover=turnover, cost_bps=10)  # 10 bps round-trip, all-in
```

`cost_bps` is meant to be a single number that already blends spread + commission + everything else you expect to pay per unit of turnover. That's fine for a rough estimate, but it can't tell you *which* cost dominates, and it can't handle costs that differ by asset (a small-cap stock's spread is not a Treasury bond's spread) or by currency.

## Itemized, by source

Each of these takes the same `weights_history` (one row per period, one column per asset - not the aggregated turnover number) and returns a per-period cost drag in return units. They compose by plain addition:

```python
from portpy.metrics.costs import bid_ask_spreads, broker_commissions, fx_spread_costs, tax_impact

spread_drag = bid_ask_spreads(weights_history, spread_bps={"AAPL": 2, "SAP": 8, "TLT": 1})
commission_drag = broker_commissions(weights_history, commission_bps=5)
fx_drag = fx_spread_costs(
    weights_history,
    asset_currencies={"SAP": "EUR"},   # everything else defaults to base_currency
    base_currency="USD",
    fx_spread_bps=3,
)
# tax_impact returns net-of-tax returns directly, not a bare drag (it needs
# the actual return to compute a taxable gain) - subtract it from gross_returns
# to get a drag comparable to the other three, so all four can be composed the same way:
tax_drag = gross_returns - tax_impact(
    gross_returns, turnover=turnover_from_weights(weights_history),
    capital_gains_tax_rate=0.15, dividend_tax_rate=0.20, dividend_yield=0.00008,
)

net_returns = gross_returns - spread_drag - commission_drag - fx_drag - tax_drag
```

Each function's `spread_bps`/`commission_bps`/`fx_spread_bps` argument accepts either one number for every asset or a `{asset: bps}` (or, for `fx_spread_costs`, `{currency: bps}`) dict/Series - build the dict from your own per-region or per-asset-class mapping (e.g. `{a: REGION_SPREADS[region_of[a]] for a in assets}`) if you want per-region costs; there's no separate "region" parameter.

## Real bid/ask FX quotes instead of an assumption

`fx_spread_costs` takes a flat `fx_spread_bps` assumption. If you have actual bid/ask FX quotes instead, turn them into a spread first:

```python
from portpy.core import calculate_fx_spread

spread_fraction = calculate_fx_spread(eur_bid_rates, eur_ask_rates)   # (ask-bid)/mid, per date
fx_drag = fx_spread_costs(weights_history, asset_currencies, base_currency, fx_spread_bps=spread_fraction.iloc[-1] * 10_000)
```

## What `tax_impact` actually models

Real capital-gains tax depends on cost basis, holding period per lot, and jurisdiction - none of which PortPy tracks (it works in weights and returns, not share lots). `tax_impact` uses a common simplifying approximation instead: it treats *turnover* as a proxy for what fraction of a period's gain gets realized and taxed, rather than deferred by staying invested, and taxes an assumed dividend yield separately since dividends are taxed whether or not you sell.

```python
net_after_tax = tax_impact(
    returns, turnover=0.15, capital_gains_tax_rate=0.15,   # 15% long-term
    dividend_tax_rate=0.20, dividend_yield=0.00008,         # ~2%/year, spread daily
)
```

This is not a tax-lot accounting engine. It has no cost basis, no short/long-term distinction beyond whichever single rate you pass in, no loss carryforward or tax-loss harvesting, and no wash-sale rule. A zero-turnover, buy-and-hold book pays zero estimated tax here regardless of how much it has appreciated - correct only in the sense that unrealized gains genuinely aren't taxed yet, but you're on your own for taxing the eventual realization when it happens.

## Putting it together

`examples/metrics_debug.ipynb` exercises every function above with a numeric walk-through; `examples/02_alpaca_multiasset_calendar.py` shows `base_currency`/`convert_to_base_currency` in a full multi-currency, multi-calendar portfolio.
