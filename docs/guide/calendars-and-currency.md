# Calendars & Currencies

PortPy never aligns or cleans your data for you. If you combine assets that trade on different schedules or in different currencies, `Portfolio` will happily accept the result, including any `NaN`s from a calendar mismatch - because silently reindexing or FX-converting financial data is exactly the kind of thing that should be a deliberate, visible step, not a side effect of a constructor.

`portpy.core` gives you the tools to make that step deliberate.

## The problem

Crypto trades 24/7. Equities and bonds trade Monday-Friday. Naively concatenating a Bitcoin price series with an AAPL price series produces `NaN`s on every weekend and holiday for AAPL - or, worse, if you don't notice, silently misaligned returns.

```python
from portpy.core import calendar_coverage_report, detect_frequency

detect_frequency(btc_prices.index)   # "daily-7"
detect_frequency(aapl_prices.index)  # "daily-5"

calendar_coverage_report(combined)   # per-asset: frequency, date range, n_missing
```

Run `calendar_coverage_report` first - it tells you which assets are actually mismatched and by how much, before you pick a fix.

## Fixing it: `align_calendars`

```python
from portpy.core import align_calendars

align_calendars(combined, method="intersection")    # business days only; drops crypto's weekend moves
align_calendars(combined, method="ffill_union")     # every day crypto trades; carries Friday's equity close through the weekend
align_calendars(combined, method="business_days")   # standardize on Mon-Fri, forward-filling gaps
```

| Method | Rows kept | Use when |
|---|---|---|
| `"intersection"` | Only dates where *every* asset has a price | You want to analyze on the slowest asset's calendar (e.g. treat crypto as if it also only moved Mon-Fri). |
| `"ffill_union"` | Every date any asset trades on | You want to keep crypto's full 24/7 calendar and are OK carrying equity closes through the weekend. |
| `"business_days"` | A clean Mon-Fri calendar | You want to standardize everything onto the equity/bond calendar regardless of the input's original shape. |

Whichever you pick, remember to set `frequency=365` (not the default `252`) on the resulting `Portfolio` if you kept crypto's 7-day calendar - otherwise every annualized metric (volatility, Sharpe, CAGR, ...) will be scaled as if the portfolio only traded 252 days/year.

## Currency conversion

By default, PortPy assumes every column in your price `DataFrame` is in the same currency and never converts anything. If you have a dated FX-rate series, convert *before* constructing the `Portfolio`:

```python
from portpy.core import convert_to_base_currency

# fx_rates: index = dates, columns = 3-letter currency codes, values = units of
# base_currency per 1 unit of that currency (e.g. an "EUR" column of 1.08 means 1 EUR = 1.08 USD)
usd_prices = convert_to_base_currency(
    prices,
    fx_rates,
    asset_currencies={"SAP": "EUR"},   # only SAP needs converting; everything else is already USD
    base_currency="USD",
)

portfolio = Portfolio(usd_prices, base_currency="USD")  # base_currency is a label only - it doesn't convert anything itself
```

`fx_rates` is reindexed to `prices.index` and forward-filled, so it doesn't need to share `prices`' exact calendar. Assets missing from `asset_currencies` (or already in `base_currency`) are left untouched. That forward-fill has no limit by default - a stale or missing FX feed silently carries the last known rate forward indefinitely. Pass `max_staleness=<n periods>` to cap it: beyond `n` consecutive missing rows, the rate (and therefore the converted price) becomes `NaN` instead - a visible gap rather than a silently stale number.

`Portfolio(..., base_currency="USD")` records the label for reporting only, the same way `asset_classes` does - it never triggers a conversion; you still need to call `convert_to_base_currency` yourself first if your data isn't already in that currency.

`convert_to_base_currency` also has no triangulation: with FX feeds for USD/EUR and USD/JPY but no direct EUR/JPY column, it raises rather than cross-computing the rate - fetch or compute the cross rate yourself first.

### FX spread cost

Converting currency also crosses an FX bid-ask spread, which is a real cost for a portfolio that rebalances foreign-currency holdings. If you have real bid/ask FX quotes, turn them into a spread fraction with `calculate_fx_spread`, then feed that into `metrics.costs.fx_spread_costs` (see the [Costs & taxes](costs-and-taxes.md) guide) to estimate the resulting return drag:

```python
from portpy.core import calculate_fx_spread

spread = calculate_fx_spread(eur_bid_rates, eur_ask_rates)   # fraction of mid, per date
fx_spread_bps = spread * 10_000   # metrics.costs.fx_spread_costs wants basis points
```

## Putting it together

See
`examples/02_alpaca_multiasset_calendar.py` for a full worked example: US equities + gold + bonds + REITs + crypto from Alpaca, a European stock quoted in EUR converted to USD, calendar coverage reported and resolved, and the resulting `Portfolio` tagged with `AssetClass` for each holding.
