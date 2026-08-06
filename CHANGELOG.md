# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
