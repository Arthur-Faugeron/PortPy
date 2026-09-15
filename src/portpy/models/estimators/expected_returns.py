"""
Estimating the expected-return vector (mu) that feeds portpy.models.optimization.

Every method here answers the same question - "what return should I plug in
for each asset?" - with a different, well-documented set of assumptions and
failure modes. None of them predicts the future; they're different ways of
compressing a noisy history (or a market-implied view) into a single number
per asset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portpy.explain import Explanation, register
from portpy.metrics.returns import annualized_return
from portpy.utils.constants import TRADING_DAYS_PER_YEAR
from portpy.utils.validation import align_pair, ensure_min_observations

__all__ = ["expected_returns"]

_METHODS = ("mean_historical", "ewma", "capm_implied", "james_stein")


def expected_returns(
    y: pd.DataFrame,
    method: str = "mean_historical",
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    geometric: bool = False,
    span: int | None = None,
    benchmark: pd.Series | None = None,
    rf: float = 0.0,
    prior: float | pd.Series | None = None,
) -> pd.Series:
    """
    Estimate an annualized expected-return vector.

    Args:
        y: Per-asset periodic returns - the full multi-asset DataFrame (matches
            `Portfolio.asset_returns()`), not the portfolio's own weighted series.
        method: One of:
            - "mean_historical": annualized historical mean return per asset
              (arithmetic mean * periods_per_year by default; see `geometric`).
            - "ewma": exponentially-weighted mean return, annualized. `span`
              sets the decay window (default 60 periods) - recent returns
              matter more than old ones.
            - "capm_implied": CAPM-implied return, `rf + beta_i * (E[market] - rf)`,
              using each asset's historical beta against `benchmark`.
            - "james_stein": cross-sectional James-Stein shrinkage of the
              historical mean toward the grand (equal-weighted) mean - a
              well-documented fix for plain historical means' large estimation
              error (Jorion, 1986; Efron & Morris, 1975).
        periods_per_year: Periods per year, used to annualize.
        geometric: "mean_historical" only - use CAGR-style geometric
            annualization per asset instead of `mean * periods_per_year`.
        span: EWMA decay in periods, used by "ewma" (default 60 if omitted).
        benchmark: Market/benchmark return series, required by "capm_implied".
        rf: Annual risk-free rate, used by "capm_implied".
        prior: Optional shrinkage target for "james_stein" - a single annual
            rate applied to every asset, or a per-asset Series. Defaults to
            the equal-weighted grand mean of `y`'s own historical means.

    Returns:
        Annualized expected return per asset, indexed by `y.columns`, named "expected_return".

    Raises:
        ValueError: If `method` is unrecognized, or `benchmark` is missing for "capm_implied".
    """
    ensure_min_observations(y, 2, "y")

    if method == "mean_historical":
        if geometric:
            values = {col: float(annualized_return(y[col].dropna(), periods_per_year=periods_per_year, geometric=True)) for col in y.columns}
            return pd.Series(values, name="expected_return")
        return (y.mean() * periods_per_year).rename("expected_return")

    if method == "ewma":
        window = 60 if span is None else span
        ewma_per_period = y.ewm(span=window).mean().iloc[-1]
        return (ewma_per_period * periods_per_year).rename("expected_return")

    if method == "capm_implied":
        if benchmark is None:
            raise ValueError("method='capm_implied' requires benchmark=<market return series>.")
        market_ann = float(benchmark.mean() * periods_per_year)
        betas = {}
        for col in y.columns:
            r, b = align_pair(y[col], benchmark, name_a=col, name_b="benchmark")
            cov = float(np.cov(r, b, ddof=1)[0, 1])
            var_b = float(np.var(b, ddof=1))
            betas[col] = cov / var_b if var_b > 1e-30 else np.nan
        beta_s = pd.Series(betas)
        return (rf + beta_s * (market_ann - rf)).rename("expected_return")

    if method == "james_stein":
        mu_periodic = y.mean()
        cov_periodic = y.cov().to_numpy()
        mu = mu_periodic * periods_per_year
        n_assets = len(mu)
        t_obs = len(y)

        if prior is None:
            target_periodic = pd.Series(float(mu_periodic.mean()), index=mu_periodic.index)
        elif isinstance(prior, (int, float)):
            target_periodic = pd.Series(float(prior) / periods_per_year, index=mu_periodic.index)
        else:
            target_periodic = pd.Series(prior).reindex(mu_periodic.index) / periods_per_year

        # Mahalanobis distance (and therefore phi) must be computed at the SAME periodic
        # scale as t_obs, the observation count - Jorion's (1986) shrinkage factor pairs a
        # sample size with a distance measured at that sample's own periodicity. Computing
        # it on annualized mu/cov instead (as an earlier version of this function did) mixes
        # an annualized distance with a periodic T: since mahalanobis computed on annualized
        # inputs scales up by periods_per_year relative to the periodic value, phi collapses
        # toward 0 for any daily-return dataset regardless of the assets' true dispersion -
        # silently defeating the shrinkage entirely at the default periods_per_year=252.
        diff_periodic = (mu_periodic - target_periodic).to_numpy()
        try:
            cov_inv = np.linalg.pinv(cov_periodic)
            mahalanobis = float(diff_periodic @ cov_inv @ diff_periodic)
        except np.linalg.LinAlgError:
            mahalanobis = float(np.sum(diff_periodic**2))

        phi = (n_assets + 2) / ((n_assets + 2) + t_obs * mahalanobis) if mahalanobis > 1e-12 else 1.0
        phi = float(np.clip(phi, 0.0, 1.0))
        target = target_periodic * periods_per_year
        shrunk = phi * target + (1.0 - phi) * mu
        return shrunk.rename("expected_return")

    raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")


register(
    Explanation(
        name="expected_returns",
        category="function",
        summary="Estimates the expected annual return of each asset, the mu vector that every optimizer in .models.optimization treats as its return forecast.",
        formula="mean_historical: mean(y)*periods_per_year | ewma: ewm(span).mean()*periods_per_year | capm_implied: rf + beta*(E[market]-rf) | james_stein: shrink mean(y) toward the grand mean by phi=(N+2)/((N+2)+T*mahalanobis)",
        how_to_read="One annualized return number per asset, in the same units as any other annual return figure (0.08 = 8%/yr). Feed it straight into .models.optimization.* as `expected_returns=`.",
        good_vs_bad="There's no 'good value' here - the question is whether the *method* fits your use case. mean_historical is the simplest and noisiest; james_stein and capm_implied trade some of that noise for model risk (wrong shrinkage target / wrong beta-market relationship, respectively).",
        caveats="Expected-return estimation is famously the weakest link in mean-variance optimization (Michaud, 1989 - 'error maximizers') - small changes in mu can swing optimized weights wildly, far more than equivalent changes in the covariance matrix. mean_historical with a short history mostly measures noise, not a forecast. james_stein here shrinks toward the simple equal-weighted grand mean, not Jorion's original minimum-variance-portfolio target - pass `prior=` explicitly to replicate that variant. capm_implied is only as good as the single-factor CAPM relationship and the chosen benchmark.",
        interpret=lambda v: f"range [{v.min():.1%}, {v.max():.1%}]/yr across {len(v)} assets" if hasattr(v, "min") else f"{v:.1%}/yr",
    )
)
