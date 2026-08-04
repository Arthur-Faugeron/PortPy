# Roadmap

PortPy is built in stages. This page is the plain-language status; see
[Architecture](https://github.com/Arthur-Faugeron/PortPy/blob/main/docs/architecture.md) 
for the full design document each stage is built from.

## Stage 1 — Metrics & Core — done (this release, v0.1.0)

- `portpy.metrics`: returns, risk, performance, drawdowns, rolling/expanding windows,
  distributions, benchmark-relative, regressions, covariance/portfolio-level risk
  decomposition, one-shot summaries, transaction costs.
- `portpy.core`: asset-class tagging, calendar alignment, currency conversion, weight
  normalization (including long/short books).
- `portpy.explain`: the explainability layer.
- Cross-validated against `empyrical` and `quantstats` on synthetic and real market data.

## Stage 2 — Visualization — not started

Plotly-first charts, one per metric family (`price_chart`, `drawdown_chart`,
`correlation_heatmap`, `monthly_returns_heatmap`, `risk_return_scatter`, ...), plus a full
`tearsheet()` and a multi-portfolio `comparison_dashboard()`. Each chart is expected to carry
its own registered `Explanation` (`fig.explain()`), the same pattern as the metrics layer.

## Stage 3 — Models & Optimization — not started

- **Factor models**: CAPM, Fama-French (3/5/9 factor), factor attribution.
- **Stochastic processes**: GBM, Ornstein-Uhlenbeck, Heston, jump-diffusion, CIR —
  calibration + simulation.
- **Monte Carlo**: bootstrap resampling, parametric/historical MC VaR & CVaR, path
  statistics.
- **Time series**: ARIMA, GARCH, cointegration, stationarity tests.
- **Covariance estimation**: Ledoit-Wolf shrinkage, EWMA, robust (MCD) covariance.
- **Optimization**: mean-variance, max Sharpe, min variance, target return/volatility,
  Black-Litterman, hierarchical risk parity, risk parity/budgeting, maximum diversification,
  efficient frontier.

## Stage 4 — Strategies — not started

A `BaseStrategy` interface, a `backtest()` engine, position sizing (Kelly, volatility
targeting), and a library of concrete strategies: buy-and-hold, equal-weight, minimum
variance, risk parity, time-series/cross-sectional/dual momentum, mean-reversion (z-score,
Bollinger), pairs trading, moving-average crossover, breakout, volatility targeting, factor
tilt, sector rotation — plus evaluation tooling (walk-forward analysis, Monte Carlo
robustness).

---

Contributions and corrections toward any stage are welcome — open an issue on
[GitHub](https://github.com/Arthur-Faugeron/PortPy/issues) to discuss scope before a large PR.
