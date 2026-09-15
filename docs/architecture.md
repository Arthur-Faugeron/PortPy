# Architecture & Design

> This is PortPy's internal design document: the full package structure, public API 
> surface, and stage-by-stage build plan. If you just want to use PortPy, start with
> [Getting Started](./getting-started.md) 
> instead — this page is for contributors and anyone curious how the pieces fit 
> together, including the parts (`strategies`, `visualization`, `models.simulation`) that don't exist yet.

**PortPy** is a Python package for quantitative portfolio analysis, performance measurement, risk assessment, and strategy backtesting. The design centers on a single `Portfolio` object that encapsulates all data (prices/returns and weights) and provides a unified interface for metrics, visualization, models, and strategies. This document defines the package's structure, core components, public API, and development roadmap.

---

## 1. Design Philosophy

- **Single entry point**: The user interacts primarily with the `Portfolio` class. All functionalities are accessible through its methods (or namespaced attributes like `.metrics`, `.models`, `.visualization`).
- **Pure functions under the hood**: Metrics and models are implemented as standalone functions that take price/return series (or a fitted result) as input and return numeric results, structured results, or `MetricResult`/`ModelResult` objects. The `Portfolio` namespaces are thin wrappers that pass internal data.
- **Data autonomy**: The package does **not** fetch or clean data from external sources. The user provides a well-formed pandas `DataFrame` (or a dictionary of assets) containing **prices** or **returns**. The `Portfolio` constructor validates basic structure and converts to a canonical internal format.
- **Modular but unified**: Internal code is organized into logical subpackages (`metrics`, `visualization`, `models`, `strategies`), but the public API is flattened through the `Portfolio` class for discoverability and ease of use.
- **Extensibility**: New metrics, models, or strategies can be added by implementing the required interfaces (function signatures) and registering them in the appropriate registry.

---

## 2. Package Structure

The package uses the `src/` layout with `pyproject.toml` as the build system (Hatchling). Dependencies are minimal: `numpy`, `pandas`, `scipy`, `statsmodels` among a few others. Visualization extras (`plotly`, `matplotlib`) and ML extras (`scikit-learn`, `arch`, `hmmlearn`) are declared as optional extras (`[viz]`, `[models]`, `[all]`).

```
portpy/
├── src/
│   └── portpy/
│       ├── __init__.py
│       ├── portfolio.py
│       ├── explain.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── asset.py
│       │   ├── calendar.py
│       │   ├── currency.py
│       │   └── weights.py
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── benchmarks.py
│       │   ├── costs.py
│       │   ├── covariance.py
│       │   ├── distributions.py
│       │   ├── drawdowns.py
│       │   ├── performance.py
│       │   ├── returns.py
│       │   ├── risk.py
│       │   ├── rolling.py
│       │   └── summary.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py           # ModelResult, constraint objects, the shared SLSQP solve routine
│       │   ├── estimators/       # expected_returns, covariance, factor_models (CAPM/Fama-French/OLS)
│       │   ├── optimization/     # pure solvers: mean_variance, max_sharpe, HRP, risk parity, Black-Litterman, ...
│       │   ├── construction/     # constraint re-exports + optimize/build/efficient_frontier
│       │   ├── management/       # rebalance, monitor, compare
│       │   └── simulation/       # Not yet implemented - stochastic processes, Monte Carlo, time series, ML
│       ├── strategies/           # Not yet implemented
│       │   ├── base/
│       │   ├── passive/
│       │   ├── momentum/
│       │   ├── mean_reversion/
│       │   ├── trend_following/
│       │   ├── volatility_targeting/
│       │   ├── factor_tilt/
│       │   ├── sector_rotation/
│       │   └── evaluation/
│       ├── visualization/        # Not yet implemented
│       │   ├── charts.py
│       │   ├── dashboards.py
│       │   ├── export.py
│       │   └── themes.py
│       └── utils/
│           ├── constants.py
│           └── validation.py
├── tests/
├── docs/
├── examples/
├── pyproject.toml
```

**Key points**:

- `portfolio.py` defines the `Portfolio` class. It builds every one of `.metrics`/`.models` (and, eventually, `.visualization`) on a shared `_AutoFillNamespace` mixin, auto-filling arguments (`returns`, `prices`, `weights`, `expected_returns`, `cov_matrix`, `current_weights`, `rf`, `periods_per_year`, ...) from the portfolio's own state where possible.
- `metrics/` contains pure functions; each submodule groups related metrics (e.g. `returns.py`). Every function also works standalone against a plain pandas Series/DataFrame.
- `explain.py` is the explainability layer: a registry of `Explanation` cards plus `MetricResult`, a float subclass that knows its own name and can render a full explanation on demand. `models/base.py`'s `ModelResult` mirrors it for structured (non-scalar) results.
- `models/` scopes to *building and running* a portfolio: `estimators` for inputs (expected returns/covariance/factor models), `optimization` for pure solver functions, `construction` for the constraint objects and the `optimize`/`build`/`efficient_frontier` recipe layer, `management` for what happens after (rebalancing, drift monitoring, comparisons). `models/simulation/` (research tooling - stochastic simulation, forecasting, ML) is a later, separate stage.
- `visualization/` will provide plotting functions that accept `Portfolio` data or pre-computed metric/model results.
- `strategies/` will implement trading strategies and a backtesting engine.

---

## 3. Data Layer (User-Provided)

The user is responsible for acquiring and cleaning their data. The `Portfolio` constructor expects a pandas `DataFrame` in one of two formats:

1. **Price format**: Columns = asset symbols, index = datetime (daily or lower frequency). Prices should be adjusted for splits/dividends.
2. **Return format**: Same shape, but values are periodic (daily) returns (simple or log). If returns are provided, the package still reconstructs cumulative prices from an initial value internally.

**Validation** performed inside `Portfolio`:

- Index is `DatetimeIndex`, sorted, and duplicate-free (raises otherwise — PortPy never silently reorders or deduplicates).
- Minimal coverage: at least two observations.
- Missing values are allowed structurally, but each metric function does its own `dropna()` and raises if too few observations remain.
- An optional `weights` vector (Series/dict/array, negative values allowed for shorts) can be provided; defaults to equal weight (1/N).
- **Calendar alignment**: mixed assets (e.g. 24/7 crypto and Mon-Fri equities) must be aligned by the user *before* constructing a `Portfolio` — see `portpy.core.align_calendars`. PortPy never forward-fills or drops rows on your behalf.
- **Currency alignment**: all assets must be denominated in the same base currency. `portpy.core.convert_to_base_currency` is provided as an explicit, opt-in helper (with an optional `max_staleness` to cap how long a forward-filled FX rate may be carried) — PortPy performs no FX conversion implicitly. `Portfolio(..., base_currency=...)` records that currency as a label only, like `asset_classes`. `portpy.core.calculate_fx_spread` turns real bid/ask FX quotes into a spread estimate for `metrics.costs.fx_spread_costs`.

No data fetching, caching, or cleaning is included. `portpy.core` and `portpy.utils.validation` provide helpers the user **may** call before constructing a `Portfolio`, but they are never invoked automatically.

---

## 4. The `Portfolio` Class

The `Portfolio` class is the central object. It holds:

- `prices` (or `returns`) as a DataFrame
- `weights` (Series, normalized to sum to 1, default 1/N, negative values allowed)
- `name` (optional, default `"My Portfolio"`)
- `frequency` (trading periods per year, used to annualize metrics — default 252)
- `risk_free_rate` (annual rate, default 0.0)
- `asset_classes` (optional `{symbol: AssetClass}` tags, purely for reporting/grouping)
- `base_currency` (optional, default `None` — a label only, like `asset_classes`; never used to convert anything)

### 4.1 Basic Accessors

- `.prices`: price DataFrame.
- `.returns(period=1, log=False)`: the portfolio's own aggregated return series (weights applied).
- `.asset_returns(log=False)`: per-asset return DataFrame (not weight-aggregated).
- `.price_index(base=100.0)`: synthetic portfolio value series, rebased.
- `.asset_names`, `.num_assets`, `.weights`.
- `.set_weights(weights)`, `.set_risk_free_rate(rate)`.

### 4.2 Metrics (via `.metrics`)

Every function in `portpy.metrics` is available as `portfolio.metrics.<name>(...)`. Parameters named `returns`, `y`, `prices`, `weights`, or `cov_matrix` are auto-filled from the portfolio when the call uses only keyword arguments; `rf` and `periods_per_year` default to the portfolio's own `risk_free_rate` and `frequency`. See the [API reference](./api/metrics/index.md) for the full list, grouped by submodule (returns, risk, performance, drawdowns, rolling, distributions, benchmarks, covariance, summary, costs).

### 4.3 Visualization (via `.visualization`) — not yet implemented

Planned: methods generating Plotly/Matplotlib figures (`price_chart`, `drawdown_chart`, `correlation_heatmap`, a full `tearsheet()`, etc.), each carrying its own `.explain()`.

### 4.4 Models (via `.models`)

A namespace of namespaces, each reusing the same `_AutoFillNamespace` mixin as `.metrics`:

- `.models.estimators` — `expected_returns(method="mean_historical"|"ewma"|"capm_implied"|"james_stein")`, `covariance(method="sample"|"ewma"|"shrinkage"|"ledoit_wolf"|"robust")`, and `factor_models` (`capm`, `fama_french`, `linear_regression`, `rolling_regression`, `factor_attribution`) built on `statsmodels`.
- `.models.optimization` — pure solver functions, `(expected_returns, cov_matrix, constraints) -> weights`: `mean_variance`, `max_sharpe`, `min_variance`, `target_return`, `target_volatility`, `efficient_frontier`, `black_litterman`, `hierarchical_risk_parity`, `risk_parity`, `risk_budgeting`, `maximum_diversification`. Solved via multi-start SLSQP (`scipy.optimize`, no cvxpy dependency) — see `models/base.py`.
- `.models.construction` — constraint objects (`WeightBounds`, `GroupCap`, `TurnoverCap`, `NetExposure`, `GrossExposure`), plus `optimize`/`build`/`efficient_frontier` (also exposed directly on `.models` for convenience — the recipe layer most users hit).
- `.models.management` — `rebalance` (trade list from current to target weights), `monitor` (drift/exposure/limit checks), `compare` (before/after or portfolio-vs-portfolio).

Every `optimize`/`build`/`efficient_frontier`/`rebalance`/`compare` call returns a `ModelResult` (`models/base.py`), carrying `.explain()`, `.summary()`, `.plot()` (raises `NotImplementedError` until `.visualization` ships), and `.compare()`. See the [models guide](./guide/models.md) and [API reference](./api/models/index.md).

### 4.5 Strategies (via `.strategies`) — not yet implemented

Planned: a `BaseStrategy` interface, a `backtest()` engine, and concrete strategies (buy-and-hold, momentum, mean-reversion, pairs trading, vol targeting, sector rotation, etc.).

---

## 5. Explainability (`portpy.explain`)

Every metric can explain itself. `Explanation` is a structured knowledge card (summary, formula, how-to-read, good-vs-bad, caveats, and an optional `interpret` function that turns a live value into a one-line verdict), registered per-function at the bottom of the module that defines it. `MetricResult` is a `float` subclass carrying its own name, unit, and `.explain()` method. `portpy.explain(name_or_result)` is the single dispatch point — see [The Explainability Layer](./guide/explainability.md).

---

## 6. Stage Plan

### Stage 1 — Metrics & Core — **DONE**

All of `portpy.metrics` (returns, risk, performance, drawdowns, rolling, distributions, benchmarks, covariance, summary, costs) and `portpy.core` (asset tagging, calendar alignment, currency conversion, weight normalization), each with full type hints, docstrings, and a registered `Explanation`.

### Stage 2 — Visualization — not started

`visualization/charts.py`, `dashboards.py`, `export.py`, `themes.py`. Plotly-first, one chart function per metric family, plus a full `tearsheet()` and a `comparison_dashboard()`.

### Stage 3 — Portfolio Construction, Optimization & Management — **DONE**

`portpy.models.estimators` (expected returns, covariance, factor models), `.optimization` (mean-variance, max Sharpe, min variance, target return/volatility, efficient frontier, Black-Litterman, hierarchical risk parity, risk parity/budgeting, maximum diversification — solved via multi-start SLSQP, no cvxpy dependency), `.construction` (constraint objects + the `optimize`/`build`/`efficient_frontier` recipe layer), and `.management` (rebalance/monitor/compare). Cross-validated against PyPortfolioOpt, Riskfolio-Lib, and skfolio — see Testing Strategy below.

### Stage 4 — Simulation & Forecasting Toolkit — not started

`portpy.models.simulation`: stochastic process simulation/calibration (GBM, jump-diffusion, ...), Monte Carlo (VaR/CVaR, bootstrap, path statistics), classical time series (ARIMA/GARCH), and a small ML toolkit (PCA factors, regime detection). Doesn't block Stage 3 and shouldn't hold up shipping it.

### Stage 5 — Strategies — not started

`BaseStrategy`, a `backtest()` engine, position sizing (Kelly, vol targeting), a library of concrete strategies (buy-and-hold, momentum variants, mean-reversion, pairs trading, breakout, vol targeting, factor tilt, sector rotation), and evaluation tooling (walk-forward analysis, Monte Carlo robustness).

See [Roadmap](./roadmap.md) for the current, plain-language version of this table.

---

## 7. Testing Strategy

- **Unit tests**: every metric/model function tested against known-input/known-output fixtures.
- **Fixtures**: synthetic constant-return series (edge cases), seeded Normal and fat-tailed (Student-t) series (approximate checks against known moments), business-day and 7-day calendars.
- **Cross-validation**: `tests/validation/test_vs_empyrical_quantstats.py` checks PortPy's metrics against `empyrical` and `quantstats`; `tests/validation/test_models_vs_libraries.py` checks every optimizer/estimator against `PyPortfolioOpt`, `Riskfolio-Lib`, and/or `skfolio` (a third, independent convex-optimization stack — planning.md's "at least 3 other libraries" bar; plus a plain-numpy OLS closed form for the factor regressions, and explicit checks for negative-weight/short-position solutions, not just long-only) — both on synthetic data (always) and real yfinance data (`@pytest.mark.network`, opt-in).
- **Portfolio integration tests**: `Portfolio` methods correctly delegate to functions and handle multi-asset DataFrames, including negative (short) weights.
- **CI**: GitHub Actions runs lint (`ruff`), type checking (`mypy`), and the non-network test suite on Python 3.10–3.12.

---

## 8. Package Metadata

- Name: `portpy-quant` (import name `portpy`).
- Dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`.
- Optional extras: `viz` (plotly, matplotlib, seaborn, kaleido), `models` (scikit-learn, arch, hmmlearn), `data` (yfinance, alpaca-py, python-dotenv), `all`, `dev` (adds `PyPortfolioOpt`/`Riskfolio-Lib`/`skfolio` alongside `empyrical`/`quantstats`, for cross-validation only — never a runtime dependency of the library itself).
- Build backend: `hatchling`.

See [CHANGELOG](./changelog.md) for release history.
