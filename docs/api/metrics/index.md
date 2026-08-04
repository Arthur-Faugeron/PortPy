# metrics

Every function here is a pure, stateless function: pass a plain `pandas.Series`/`DataFrame`
and get back a `float`, `Series`, `DataFrame`, or (with `as_result=True`, where applicable) a
self-explaining `MetricResult`. `Portfolio.metrics.<name>()` is a thin wrapper around these
same functions — see [The Portfolio object](../../guide/portfolio.md) for what it auto-fills.

| Submodule | Covers |
|---|---|
| [returns](returns.md) | Return transformations, total/annualized/average return, CAGR. |
| [risk](risk.md) | Variance, volatility, downside risk, VaR/CVaR, tail ratio, skew/kurtosis, beta, tracking error. |
| [performance](performance.md) | Sharpe, Sortino, Calmar, Omega, information/Treynor ratios, and more. |
| [drawdowns](drawdowns.md) | Drawdown series, max drawdown, duration, recovery, top-N episodes. |
| [rolling](rolling.md) | Generic rolling/expanding-window engine, plus rolling Sharpe/volatility/beta/correlation. |
| [distributions](distributions.md) | Descriptive stats, normality test, win rate, monthly returns table, histogram data. |
| [benchmarks](benchmarks.md) | Alpha, correlation, R², up/down capture ratios, batting average. |
| [regressions](regressions.md) | OLS (single- and multi-factor), rolling regression, tidy summary tables. |
| [covariance](covariance.md) | Covariance/correlation matrices, portfolio variance/volatility, risk decomposition. |
| [summary](summary.md) | One-call `tearsheet_summary()` and `compare_to_benchmark()`. |
| [costs](costs.md) | Turnover from a weight history, net-of-costs returns. |
