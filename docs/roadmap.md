# Roadmap

PortPy is built in stages. This page is the plain-language status; see [Architecture](./architecture.md) for the full design document each stage is built from.

## Stage 1 - Metrics & Core - done (v0.1.0, hardened in v0.1.2, cost/currency realism extended in v0.2.0)

- `portpy.metrics`: returns, risk, performance, drawdowns, rolling/expanding windows, distributions, benchmark-relative, covariance/portfolio-level risk decomposition, one-shot summaries, transaction costs (a blended flat rate, plus itemized spread/commission/FX-spread/tax-drag estimators - see [Costs & taxes](./guide/costs-and-taxes.md)).
- `portpy.core`: asset-class tagging, calendar alignment, currency conversion (with an FX-spread estimator), weight normalization (including long/short books).
- `portpy.explain`: the explainability layer.
- Cross-validated against `empyrical` and `quantstats` on synthetic and real market data.

## Stage 2 - Visualization - not started (deferred out of v0.2.0)

Plotly-first charts, one per metric family (`price_chart`, `drawdown_chart`, `correlation_heatmap`, `monthly_returns_heatmap`, `risk_return_scatter`, ...), plus a full `dashboard()` and a multi-portfolio `comparison_dashboard()`. Each chart recycles its underlying metric's/model's registered `Explanation` (`fig.explain()`) rather than registering its own - no chart ever registers a `category="chart"` explanation. Originally planned alongside Stage 3 for v0.2.0, but hasn't started yet and now ships in a later version.

## Stage 3 - Portfolio Construction, Optimization & Management - done (v0.2.0)

- **Estimators**: expected returns (mean-historical, EWMA, CAPM-implied, James-Stein shrinkage), covariance (sample, EWMA, shrinkage, Ledoit-Wolf, robust/MCD), factor models (CAPM, Fama-French 3/5-factor, linear/rolling regression, factor attribution).
- **Optimization**: mean-variance, max Sharpe, min variance, target return/volatility, efficient frontier, Black-Litterman, hierarchical risk parity, risk parity/budgeting, maximum diversification - solved via multi-start SLSQP (scipy-only, no cvxpy).
- **Construction**: constraint objects (weight bounds, group/sector caps, turnover caps, net/gross exposure), plus the `optimize`/`build`/`efficient_frontier` recipe layer.
- **Management**: `rebalance` (trade lists), `monitor` (drift/limit checks), `compare` (before/after or portfolio-vs-portfolio).
- Cross-validated against `PyPortfolioOpt`, `Riskfolio-Lib`, and `skfolio` (plus a plain-numpy OLS closed form for the factor regressions) on synthetic and real market data, including negative-weight/short-position solutions and a mixed-timeframe (equities + crypto) universe.

## Stage 4 - Simulation & Forecasting Toolkit - not started

- **Stochastic processes**: GBM, Ornstein-Uhlenbeck, Heston, jump-diffusion, CIR - calibration + simulation.
- **Monte Carlo**: bootstrap resampling, parametric/historical MC VaR & CVaR, path statistics.
- **Time series**: ARIMA, GARCH, cointegration, stationarity tests.
- **ML toolkit**: PCA factor extraction, regime detection (HMM).

Doesn't block Stage 3 and shouldn't hold up shipping it - that's why the two were split into separate stages.

## Stage 5 - Strategies - not started

A `BaseStrategy` interface, a `backtest()` engine, position sizing (Kelly, volatility targeting), and a library of concrete strategies: buy-and-hold, equal-weight, minimum variance, risk parity, time-series/cross-sectional/dual momentum, mean-reversion (z-score, Bollinger), pairs trading, moving-average crossover, breakout, volatility targeting, factor tilt, sector rotation, plus more advanced statistical strategies (Random Matrix Theory portfolios, regime-switching, PCA vector-space tracking, econophysical approaches) - and evaluation tooling (walk-forward analysis, Monte Carlo robustness).

---

Contributions and corrections toward any stage are welcome - open an issue on [GitHub](https://github.com/Arthur-Faugeron/PortPy/issues) to discuss scope.
