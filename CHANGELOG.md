# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-08-18

A correctness pass driven by an internal quant review of every formula, docstring, and
caveat in the library (26 findings). Real bugs were fixed in place. Every fix and every 
remaining methodology difference was re-validated against `empyrical` and `quantstats` 
on real market data (not synthetic/mocked series) both before and after the change.

### Fixed

- **`semi_variance`**: divided by the count of below-MAR observations instead of the
  full sample size N, contradicting Markowitz's definition and this library's own
  `downside_deviation`. `sqrt(semi_variance(...))` now equals
  `downside_deviation(..., annualized=False)` exactly, as the naming always implied.
- **`annualized_return`** / **`cagr`**: the geometric branch raised
  `TypeError: can't convert complex to float` whenever the compounded growth factor was
  `<= 0` - reachable with leveraged/short portfolios, which this library explicitly
  supports. Now returns `-1.0` (a complete loss) instead of crashing.
- **`up_capture_ratio`** / **`down_capture_ratio`**: re-annualized a filtered,
  non-contiguous up-day/down-day subsample via `annualized_return`, which is not a
  valid use of geometric annualization. Now compares compounded sub-period returns
  directly. **Note:** `empyrical`'s own `up_capture`/`down_capture` use the same
  re-annualization this fix removes, so PortPy's values now intentionally diverge from
  `empyrical`'s on this pair of metrics - see the validation report linked in the repo
  for the measured size of that divergence on real data.
- **`DEFAULT_MAR`**'s docstring was attached to the wrong constant
  (`DEFAULT_RISK_FREE_RATE`) in the `Constants` block. Both constants now have their
  own docstring.

### Added

- **`conditional_drawdown_at_risk`**: true Conditional Drawdown at Risk (CDaR) - the
  mean of the worst tail of the drawdown series, the drawdown analogue of
  `conditional_var`. (`drawdown_at_risk` is only the percentile, "DaR"; its docstring
  no longer claims to be CDaR.)
- **`sterling_ratio(..., drawdown_adjustment=0.0)`**: optional parameter for the
  classic Deane Sterling Jones "+10%" drawdown-adjustment convention
  (`drawdown_adjustment=0.10`). Defaults to `0.0`, matching prior behavior exactly.
- **`normalize_weights(..., method="net")`**: optional `"gross"` mode, normalizing by
  gross instead of net exposure - avoids blow-up for near-dollar-neutral long/short
  books. Defaults to `"net"`, matching prior behavior exactly.
- **`time_to_recovery(..., as_result=False)`**: now supports `as_result=True` like
  every other scalar metric, returning a self-explaining `MetricResult`.

### Changed

- Expanded caveats across nearly every metric module, matching the source review:
  i.i.d./no-autocorrelation assumptions behind every `sqrt(periods_per_year)`
  annualization; sample-vs-population estimators in `skewness`/`kurtosis`; missing
  Dimson/Blume adjustments in `beta`; VaR-method reliability limits; calendar-alignment
  and FX-quote-direction pitfalls in `portpy.core`; HAC/Newey-West guidance for
  `portpy.metrics.regressions`; and more - see each function's docstring and
  `.explain()` text.
- Recategorized several return/turnover-transform functions
  (`simple_returns`, `log_returns`, `cumulative_returns`, `prices_from_returns`,
  `rebased_returns`, `excess_returns`, `active_returns`, `turnover_from_weights`,
  `net_of_costs_returns`) from `category="metric"` to `category="function"` in the
  `.explain()` registry, since they return a series/frame rather than a single
  explainable value - consistent with how `portpy.metrics.rolling`'s generic engines
  are already categorized.
- Removed a duplicated block of `.explain()` registrations for `alpha`/`correlation`/
  `r_squared`/capture-ratio/`batting_average` that had been accidentally left in
  `distributions.py` (those functions live in `benchmarks.py`, which already registers
  them).

### Known limitations

- `up_capture_ratio`/`down_capture_ratio` no longer match `empyrical`'s equivalents by
  construction (see Fixed, above) - this is an intentional, documented methodological
  difference, not a regression.

## [Pre-Release] - 2026-08-04

## [0.1.0] - 2026-08-04

First release. Metrics and core are implemented and tested; visualization, models, and
strategies are designed but not yet built — see [docs/roadmap.md](https://github.com/Arthur-Faugeron/PortPy/blob/main/docs/roadmap.md).

### Added

- **`Portfolio`**: the single entry point over price/return data — weights (negative/short
  positions supported), asset-class tags, and a `.metrics` namespace that auto-fills
  `returns`/`prices`/`weights`/`rf`/`periods_per_year` from the portfolio's own state on
  keyword-only calls.
- **`portpy.metrics`**: `returns`, `risk`, `performance`, `drawdowns`, `rolling`, 
  `distributions`, `benchmarks` (alpha/beta/capture ratios), `regressions` 
  (OLS via statsmodels), `covariance`(portfolio-level risk decomposition: variance, 
  diversification ratio, marginal/component contribution to risk), one-shot 
  `summary` tables, and transaction-`costs` helpers. Most scalar metrics accept 
  `as_result=True` to get a self-explaining `MetricResult` back.
- **`portpy.core`**: `calendars` for combining assets that trade on different calendars 
  (e.g. 24/7 crypto with Mon-Fri equities); `currency` for multi-currency portfolios; 
  `weights` for weight validation (including long/short books); and `asset` to represent
  individual assets.
- **`portpy.explain`**: the explainability layer — `Explanation` knowledge cards registered
  per metric (what it is, formula, how to read it, good vs. bad, caveats, a value-specific
  verdict), `MetricResult` (a `float` subclass carrying its own name/unit/interpretation),
  and `portpy.explain(name_or_result)` as the single dispatch point.
- **Examples**: `01_yfinance_getting_started.py`, `02_alpaca_multiasset_calendar.py`, and
  `tutorial.ipynb` — a full walkthrough of every metric function against three real
  long/short, multi-asset portfolios built from live Alpaca (stocks/ETFs/crypto) and FRED
  (risk-free rate) data.
- **Tests**: unit coverage for every metric module, core of portpy, plus
  `tests/validation/test_vs_empyrical_quantstats.py`, cross-checking PortPy's numbers
  against `empyrical` and `quantstats` on synthetic and real (yfinance) market data.
- **Docs**: a full guide + API reference site (`mkdocs` + `mkdocstrings`) covering the
  `Portfolio` object, weights/shorts, calendar & currency alignment, the explainability
  layer, and every function's API docs.
- Project scaffolding: `pyproject.toml`, `src/` layout, CI (lint/type-check/test/build) and
  PyPI-publish GitHub Actions workflows, MIT license.

### Known limitations

- `portpy.visualization`, `portpy.models`, and `portpy.strategies` are designed (see
  [docs/architecture.md](https://github.com/Arthur-Faugeron/PortPy/blob/main/docs/architecture.md)) but not implemented in this release.
