# Architecture & Design

> This is PortPy's internal design document: the full package structure, public API surface,
> and stage-by-stage build plan. If you just want to *use* PortPy, start with
> [Getting Started](getting-started.md) instead — this page is for contributors and anyone
> curious how the pieces fit together, including the parts (`models`, `strategies`,
> `visualization`) that don't exist yet.

**PortPy** is a Python package for quantitative portfolio analysis, performance measurement, risk assessment, and strategy backtesting. The design centers on a single **`Portfolio`** object that encapsulates all data (prices/returns and weights) and provides a unified interface for metrics, visualization, models, and strategies. This document defines the package's structure, core components, public API, and development roadmap.

---

## 1. Design Philosophy

- **Single entry point**: The user interacts primarily with the `Portfolio` class. All functionalities are accessible through its methods (or namespaced attributes like `.metrics`, `.visualization`).
- **Pure functions under the hood**: Metrics are implemented as standalone functions that take price/return series as input and return numeric results or `MetricResult` objects. The `Portfolio` methods are thin wrappers that pass internal data.
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
│       │   ├── regressions.py
│       │   ├── returns.py
│       │   ├── risk.py
│       │   ├── rolling.py
│       │   └── summary.py
│       ├── models/               # Not yet implemented
│       │   ├── factor_models/
│       │   ├── stochastic_processes/
│       │   ├── monte_carlo/
│       │   ├── time_series/
│       │   ├── covariance_estimation/
│       │   ├── ml/
│       │   └── optimization/
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

- `portfolio.py` defines the `Portfolio` class. It dynamically binds every function in `portpy.metrics` as a method under `.metrics`, auto-filling arguments (`returns`, `prices`, `weights`, `rf`, `periods_per_year`) from the portfolio's own state where possible.
- `metrics/` contains pure functions; each submodule groups related metrics (e.g. `returns.py`). Every function also works standalone against a plain pandas Series/DataFrame.
- `explain.py` is the explainability layer: a registry of `Explanation` cards plus `MetricResult`, a float subclass that knows its own name and can render a full explanation on demand.
- `visualization/` will provide plotting functions that accept `Portfolio` data or pre-computed metric results.
- `models/` will contain financial models and optimization routines.
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
- **Currency alignment**: all assets must be denominated in the same base currency. `portpy.core.convert_to_base_currency` is provided as an explicit, opt-in helper — PortPy performs no FX conversion implicitly.

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

### 4.1 Basic Accessors

- `.prices` → price DataFrame.
- `.returns(period=1, log=False)` → the portfolio's own aggregated return series (weights applied).
- `.asset_returns(log=False)` → per-asset return DataFrame (not weight-aggregated).
- `.price_index(base=100.0)` → synthetic portfolio value series, rebased.
- `.asset_names`, `.num_assets`, `.weights`.
- `.set_weights(weights)`, `.set_risk_free_rate(rate)`.

### 4.2 Metrics (via `.metrics`)

Every function in `portpy.metrics` is available as `portfolio.metrics.<name>(...)`. Parameters named `returns`, `y`, `prices`, `weights`, or `cov_matrix` are auto-filled from the portfolio when the call uses only keyword arguments; `rf` and `periods_per_year` default to the portfolio's own `risk_free_rate` and `frequency`. See the [API reference](api/metrics/index.md) for the full list, grouped by submodule (returns, risk, performance, drawdowns, rolling, distributions, benchmarks, regressions, covariance, summary, costs).

### 4.3 Visualization (via `.visualization`) — not yet implemented

Planned: methods generating Plotly/Matplotlib figures (`price_chart`, `drawdown_chart`, `correlation_heatmap`, a full `tearsheet()`, etc.), each carrying its own `.explain()`.

### 4.4 Models (via `.models`) — not yet implemented

Planned: `factor_models` (CAPM, Fama-French), `stochastic_processes`, `monte_carlo`, `time_series` (ARIMA/GARCH), `covariance_estimation` (Ledoit-Wolf, EWMA), and `optimization` (mean-variance, risk parity, HRP, efficient frontier).

### 4.5 Strategies (via `.strategies`) — not yet implemented

Planned: a `BaseStrategy` interface, a `backtest()` engine, and concrete strategies (buy-and-hold, momentum, mean-reversion, pairs trading, vol targeting, sector rotation, etc.).

---

## 5. Explainability (`portpy.explain`)

Every metric can explain itself. `Explanation` is a structured knowledge card (summary, formula, how-to-read, good-vs-bad, caveats, and an optional `interpret` function that turns a live value into a one-line verdict), registered per-function at the bottom of the module that defines it. `MetricResult` is a `float` subclass carrying its own name, unit, and `.explain()` method. `portpy.explain(name_or_result)` is the single dispatch point — see [The Explainability Layer](guide/explainability.md).

---

## 6. Stage Plan

### Stage 1 — Metrics & Core — **done, this release**

All of `portpy.metrics` (returns, risk, performance, drawdowns, rolling, distributions, benchmarks, regressions, covariance, summary, costs) and `portpy.core` (asset tagging, calendar alignment, currency conversion, weight normalization), each with full type hints, docstrings, and a registered `Explanation`.

### Stage 2 — Visualization — not started

`visualization/charts.py`, `dashboards.py`, `export.py`, `themes.py`. Plotly-first, one chart function per metric family, plus a full `tearsheet()` and a `comparison_dashboard()`.

### Stage 3 — Models & Optimization — not started

Factor models, stochastic process simulation/calibration, Monte Carlo (VaR/CVaR, bootstrap, path statistics), classical time series (ARIMA/GARCH/cointegration), covariance estimation (shrinkage/EWMA/robust), a small ML toolkit, and portfolio optimizers (mean-variance, max Sharpe, risk parity, HRP, Black-Litterman, efficient frontier).

### Stage 4 — Strategies — not started

`BaseStrategy`, a `backtest()` engine, position sizing (Kelly, vol targeting), a library of concrete strategies (buy-and-hold, momentum variants, mean-reversion, pairs trading, breakout, vol targeting, factor tilt, sector rotation), and evaluation tooling (walk-forward analysis, Monte Carlo robustness).

See [Roadmap](roadmap.md) for the current, plain-language version of this table.

---

## 7. Testing Strategy

- **Unit tests**: every metric function tested against known-input/known-output fixtures.
- **Fixtures**: synthetic constant-return series (edge cases), seeded Normal and fat-tailed (Student-t) series (approximate checks against known moments), business-day and 7-day calendars.
- **Cross-validation**: `tests/validation/test_vs_empyrical_quantstats.py` checks PortPy's numbers against `empyrical` and `quantstats` on synthetic data (always) and real yfinance data (`@pytest.mark.network`, opt-in).
- **Portfolio integration tests**: `Portfolio` methods correctly delegate to functions and handle multi-asset DataFrames, including negative (short) weights.
- **CI**: GitHub Actions runs lint (`ruff`), type checking (`mypy`), and the non-network test suite on Python 3.10–3.12.

---

## 8. Package Metadata

- Name: `portpy`.
- Dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`.
- Optional extras: `viz` (plotly, matplotlib, seaborn, kaleido), `models` (scikit-learn, arch, hmmlearn), `data` (yfinance, alpaca-py, python-dotenv), `all`, `dev`.
- Build backend: `hatchling`.

See [CHANGELOG](https://github.com/Arthur-Faugeron/PortPy/blob/main/CHANGELOG.md) for release history.
