# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.2.0] - 2026-09-15

Stage 3: `portpy.models` (portfolio construction, optimization, and management); a
currency/transaction-cost realism pass over Stage 1's `portpy.core`/`portpy.metrics.costs`
that implements what an earlier `planning.md` revision had already (prematurely) documented;
and a theory/documentation correctness audit of the whole `.models` module that found and
fixed one significant numerical bug (`james_stein`'s shrinkage intensity, silently
near-zero at default settings) and several documentation/example inaccuracies, then
extended the model validation to a third independent library (`skfolio`), negative-weight
(short) solutions, and a genuinely mixed-timeframe (equities + 24/7 crypto) universe.
`portpy.visualization` (Stage 2) was originally planned alongside this release but hasn't
started yet and ships separately in a later version - see
[Known limitations](#known-limitations) below.

### Added

- **`portpy.models.estimators`**: `expected_returns` (`mean_historical`, `ewma`, `capm_implied`,
  `james_stein` shrinkage), `covariance` (`sample`, `ewma`, `shrinkage`, `ledoit_wolf`, `robust`),
  and `factor_models` (`capm`, `fama_french` 3/5-factor, `linear_regression`,
  `rolling_regression`, `factor_attribution`) - replaces the old experimental
  `metrics/regressions.py` module.
- **`portpy.models.optimization`**: pure solver functions - `mean_variance`, `max_sharpe`,
  `min_variance`, `target_return`, `target_volatility`, `efficient_frontier`,
  `black_litterman`, `hierarchical_risk_parity`, `risk_parity`, `risk_budgeting`,
  `maximum_diversification`. Solved via multi-start SLSQP (`scipy.optimize`) - no `cvxpy`
  dependency, matching the existing `[models]` extra's footprint.
- **`portpy.models.construction`**: constraint objects (`WeightBounds`, `GroupCap`,
  `TurnoverCap`, `NetExposure`, `GrossExposure`), and the `optimize`/`build`/
  `efficient_frontier` recipe layer over `.optimization` (also exposed directly on
  `Portfolio.models` for convenience).
- **`portpy.models.management`**: `rebalance` (target weights -> trade list, with turnover and
  a flat cost estimate), `monitor` (drift/exposure/limit checks), `compare` (before/after or
  portfolio-vs-portfolio, weight-level always and metric-level whenever both sides are
  `Portfolio`s with return history).
- **`portpy.models.base`**: `ModelResult` (the model-layer counterpart to `MetricResult` -
  `.explain()`, `.summary()`, `.compare()`; `.plot()` raises `NotImplementedError` until
  `.visualization` ships) and the shared multi-start solve routine every optimizer uses.
- **`Portfolio.models`**: a namespace of namespaces (`.estimators`/`.optimization`/
  `.construction`/`.management`, plus `.optimize`/`.build`/`.efficient_frontier` directly),
  auto-filling `expected_returns`/`cov_matrix`/`current_weights`/etc. from the portfolio's own
  data on keyword-only calls - the same pattern `.metrics` already had.
- Every model type has a registered `Explanation` (`category="model"` for optimizers/`capm`/
  `fama_french`/`rebalance`/`compare`; `category="function"` for the estimator/utility functions).
- **Examples**: `examples/models_debug.ipynb` (exhaustive, function-by-function reference) and
  `examples/models_tutorial.ipynb` (narrative walkthrough: estimate inputs, optimize, apply
  constraints, rebalance, monitor, compare) - both run against real `yfinance` data.
- **Tests**: unit coverage for every `models` submodule
  (`tests/models/`), plus `tests/validation/test_models_vs_libraries.py` cross-validating every
  optimizer/estimator against `PyPortfolioOpt` and/or `Riskfolio-Lib` (and a plain-numpy OLS
  closed form for the factor regressions), on synthetic data (always) and real yfinance data
  (`@pytest.mark.network`, opt-in).
- `PyPortfolioOpt` and `Riskfolio-Lib` added to the `dev` extra (cross-validation only, same
  role as `empyrical`/`quantstats` for Stage 1 - never a runtime dependency).
- **Currency/cost realism** (implementing what an earlier revision of `planning.md` had
  promised but the code never actually shipped - see Fixed):
  - **`portpy.metrics.costs`**: `bid_ask_spreads`, `broker_commissions`, `fx_spread_costs`,
    and `tax_impact` - itemized, per-source cost/tax drag estimators that compose by plain
    addition, alongside the existing blended `net_of_costs_returns`. Each `*_bps` argument
    accepts a scalar, a `{asset: bps}` dict/Series (or `{currency: bps}` for
    `fx_spread_costs`), for per-asset/per-currency (and, via your own grouping dict,
    per-region) cost modeling. `tax_impact` is a deliberately first-order approximation
    (turnover as a realization proxy, one blended rate) - not a lot-level tax engine; see its
    docstring and the new `docs/guide/costs-and-taxes.md` guide for exactly
    what it doesn't model (no cost basis, no loss harvesting, no wash-sale rule).
  - **`portpy.core.currency`**: `calculate_fx_spread(bid_rates, ask_rates)` turns real bid/ask
    FX quotes into a spread fraction for `fx_spread_costs`; `convert_to_base_currency` gained
    an optional `max_staleness` to cap how long a forward-filled FX rate may be carried before
    lapsing to `NaN` (unlimited by default, for backward compatibility).
  - **`Portfolio(..., base_currency=...)`**: a new, purely informational constructor argument
    (same status as `asset_classes`) - fulfills `planning.md`'s claim that `Portfolio` "holds"
    a `base_currency`, without coupling `Portfolio` to any FX-conversion logic (conversion
    stays the user's job, done before construction, matching the existing calendar-alignment
    stance).
  - **`examples/metrics_debug.ipynb`**: extended with sections for all five new functions
    (real market data, `SAP.DE`'s EUR pricing exercises the currency-cost path).
- **`skfolio`** added to `tests/validation/test_models_vs_libraries.py` and
  `examples/04_crosstest_models_final.ipynb` as a third, independent optimization-library
  cross-check (`min_variance`, `max_sharpe`, `risk_parity`) - `planning.md` Section 7 asks
  for "at least 3 other libraries"; the models validation previously had only two
  (`PyPortfolioOpt`, `Riskfolio-Lib`). Also added to the `dev` extra.
- **Negative-weight (short-position) cross-validation**: `max_sharpe` under a
  `WeightBounds(low<0)` constraint is now explicitly checked against both PyPortfolioOpt
  and skfolio in both the automated suite and the crosstest notebook - every prior
  weight-based check used the default long-only bounds, so a genuine short position (not
  just an unused negative bound) had never actually been cross-validated.
- **A genuinely mixed-timeframe cross-validation**: `examples/04_crosstest_models_final.ipynb`
  now fetches `BTC/USD` (24/7, Alpaca's public crypto endpoint) alongside the Mon-Fri equity
  universe, aligns them with `core.align_calendars(method="ffill_union")`, and cross-checks
  `min_variance` on the resulting 6-asset, `frequency=365` portfolio against PyPortfolioOpt -
  the models crosstest previously only ever used a single shared Mon-Fri calendar.

### Fixed

- **`hierarchical_risk_parity`**'s post-hoc `WeightBounds` handling used to clip weights to
  bounds and then renormalize by dividing by the clipped sum - which can push an
  already-at-the-cap asset back *over* its cap, since dividing inflates every weight including
  the pinned ones. Replaced with a proper water-filling redistribution that only redistributes
  surplus/deficit through assets not yet pinned at a bound.
- **`TurnoverCap`** built via a plain top-level import
  (`from portpy.models.construction import TurnoverCap`) had no way to pick up
  `current_weights` afterward (it's a frozen dataclass) - only
  `portfolio.models.construction.TurnoverCap(...)` got the auto-fill. Every `.models` entry
  point that accepts `constraints=` now patches any bare `TurnoverCap` before solving, so which
  import path built the object no longer matters.
- Recategorized six `Explanation`s that were registered under `category="chart"` even though
  they're metrics-module functions, not charts (a correction noted since Stage 1 hardening):
  `rolling_sharpe`, `rolling_volatility`, `rolling_beta`, `rolling_correlation` -> `"metric"`;
  `monthly_returns_table`, `return_histogram_data` -> `"function"`. No name is meant to register
  `category="chart"` under this design - a chart recycles its underlying metric's/model's card
  instead of registering its own.
- **`estimators.covariance`**'s `"shrinkage"` method's docstring incorrectly described its
  shrinkage target as "equal-variance" - the code (correctly) shrinks toward
  `diag(diag(sample))`, which keeps each asset's *own* sample variance and only zeroes
  cross-correlations, not a target with equal variance across assets. The description now
  matches the code, and both `"shrinkage"` and `"robust"` explicitly note that they target a
  different matrix than same-named methods in other libraries (PyPortfolioOpt's
  `shrunk_covariance` shrinks to a scaled-identity target instead; `MinCovDet` applies a
  consistency-reweighting step that a raw MCD implementation doesn't) - a theory-documentation
  audit found no bugs in the underlying formulas, only this one inaccurate description.
- **`planning.md`**'s Section 3/4 currency paragraphs and its "Costs & Real Returns" bullet
  described a design that was never implemented: `Portfolio` "tracking" `base_currency` and
  computing FX spread/cost internally, plus `tax_impact`/`broker_commissions`/
  `bid_ask_spreads`/`fx_spread_costs`/`calculate_fx_spread` listed as existing when none of
  them did. Both the implementation gap and the planning-document drift are now closed (see
  Added, and `planning.md` itself for the corrected wording).
- **`net_of_costs_returns`** silently dropped the input `returns` Series' `.name` whenever
  `turnover` was passed as a `pd.Series` (pandas sets a binary op's result name to `None`
  when the two operands' names differ) - harmless for a bare float result, but meant
  `net_of_costs_returns(returns, turnover=some_series, ...).name` wasn't `returns.name` the
  way the scalar-`turnover` branch's result already was. Now renamed back explicitly in both
  branches, and `tax_impact` follows the same convention.
- **`examples/metrics_debug.ipynb`**'s final "not-yet-implemented subpackages" section (20.3)
  still claimed `portpy.models` was an empty placeholder with "only a `# Yet to be made`
  comment" - stale since Stage 3 shipped `.models` in full. Corrected to report only
  `portpy.strategies`/`portpy.visualization` (still stubs) and `portpy.models.simulation`
  (the one remaining stub *within* `.models`) as not yet implemented. Its title cell also
  still read the pre-rename `` `debug.ipynb` `` from before it became `metrics_debug.ipynb` -
  fixed.
- **`expected_returns(method="james_stein")` computed its shrinkage intensity at the wrong
  scale**, a real bug (not just a documentation issue) found while writing an example for
  `models_debug.ipynb`: `phi=(N+2)/((N+2)+T*mahalanobis)` mixed `t_obs` (a count of periodic,
  e.g. daily, observations) with a `mahalanobis` distance computed on already-*annualized*
  `mu`/`cov`. Annualizing first scales the mahalanobis distance up by roughly
  `periods_per_year`, which collapsed `phi` toward 0 for any daily-return dataset regardless
  of the assets' true dispersion - on real 4-year daily data across 5 assets, the old code
  produced `phi~0.002` (shrinkage invisible to 3 decimal places); the same data now produces
  `phi~0.40` (a textbook Bayes-Stein intensity). Fixed by computing `mahalanobis` (and
  therefore `phi`) at the same periodic scale as `t_obs`, then annualizing only the final
  shrunk estimate - `phi` is now provably independent of `periods_per_year`, guarded by a new
  regression test. This silently defeated the entire purpose of the estimator at the default
  `periods_per_year=252` for any realistic daily-data use - the most significant fix in this
  pass.
- **`TurnoverCap`'s docstring called its convention "one-way turnover"**, but
  `max_turnover` is compared against the *raw*, un-halved `sum(abs(w - current_weights))` -
  exactly twice `metrics.costs.turnover_from_weights`'s actual one-way convention (which
  halves the same sum). `models_tutorial.ipynb`'s original constraints example inherited this
  confusion: it labeled a `TurnoverCap(max_turnover=0.15)` as "10% turnover" in prose while
  the constraint itself allowed 15% raw / 7.5% one-way - none of those three numbers agreed
  with each other. `TurnoverCap`'s docstring now states the raw-sum convention explicitly
  and cross-references the halved one; the tutorial's example and prose now agree.

### Documentation

- Filled two missing-caveat gaps found in the same audit: `portpy.models.optimization`'s module
  docstring now states explicitly that no solver checks `expected_returns`/`cov_matrix` for a
  consistent time scale (mixing periodic and annualized inputs is silently accepted and
  financially meaningless); `efficient_frontier`'s docstring and `Explanation` now note that its
  default `max_return` (the single highest per-asset expected return) is a long-only ceiling
  that understates the true frontier whenever `constraints` permit leverage or shorting.
  `black_litterman`'s `views` docstring now states explicitly that view values must be
  annualized, matching `cov_matrix`'s scale.

### Changed

- `Portfolio.metrics`'s auto-fill dispatcher was refactored onto a shared `_AutoFillNamespace`
  mixin (still in `portfolio.py`) so `.models` (and, later, `.visualization`) reuse it verbatim
  instead of re-implementing argument-filling separately. No behavior change for `.metrics` -
  the full Stage 1 test suite passes unmodified against the refactored code.
- Renamed `examples/debug.ipynb` -> `examples/metrics_debug.ipynb` and
  `examples/tutorial.ipynb` -> `examples/metrics_tutorial.ipynb`, for naming consistency with
  the new `examples/models_debug.ipynb`/`examples/models_tutorial.ipynb`.
- **`examples/models_debug.ipynb` and `examples/models_tutorial.ipynb` rebuilt** to match
  `metrics_debug.ipynb`'s structure and rigor rather than their own earlier, weaker pattern:
  `models_debug.ipynb` now tracks call/explain coverage the same way (`COVERED`/`EXPLAINED`
  sets, an assertion-based coverage audit, a brute-force `Explanation.render()` sanity check
  over every registered "model" card) instead of a bare, non-asserting name dump.
  `models_tutorial.ipynb`'s prose was trimmed of dramatic/imperative framing ("you don't get
  to skip this step", "turning an optimizer into something you can actually run") in favor of
  plain section titles, and both notebooks' James-Stein and `TurnoverCap` examples were fixed
  to actually demonstrate what their surrounding prose claims (see Fixed above for why they
  previously didn't).

### Known limitations

- `portpy.visualization` (Stage 2) is not implemented in this release - `ModelResult.plot()`
  raises `NotImplementedError` by design until it ships.
- `portpy.models.simulation` (stochastic processes, Monte Carlo, time series, ML) is a
  separate, later stage (v0.3.0) and is not implemented - only an empty placeholder module
  exists.
- `hierarchical_risk_parity` only supports the `WeightBounds` constraint (applied post-hoc);
  every other constraint type is dropped with a warning, since recursive bisection has no
  natural way to target an exact group cap, turnover budget, or leverage level.
- `target_return` uses an equality constraint (`w'mu == target`), not "at least target" -
  a target below the global minimum-variance portfolio's own return still pins you to that
  lower, needlessly higher-variance point. Some other libraries (e.g. PyPortfolioOpt's
  `efficient_return`) use `>=` instead, which substitutes the min-variance portfolio once its
  own return already clears the bar - the two conventions only agree above the min-variance
  return.

## [0.1.2] - 2026-08-18

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
