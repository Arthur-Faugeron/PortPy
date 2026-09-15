"""
Factor regressions: CAPM, Fama-French, and general linear/rolling factor models.

Replaces the old experimental `metrics/regressions.py` module. Everything here
is a thin, opinionated wrapper around `statsmodels` OLS - PortPy adds the
excess-return bookkeeping (rf, RF-column handling) and sensible defaults;
statsmodels does the actual estimation, so every `RegressionResultsWrapper`
returned here carries the full statsmodels API (`.params`, `.rsquared`,
`.pvalues`, `.summary()`, ...).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.regression.rolling import RollingOLS

from portpy.explain import Explanation, register
from portpy.utils.constants import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from portpy.utils.validation import align_pair, ensure_min_observations, periodic_rate_from_annual

__all__ = ["linear_regression", "rolling_regression", "capm", "fama_french", "factor_attribution"]

_CANONICAL_FF_FACTORS = {3: ["Mkt-RF", "SMB", "HML"], 5: ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]}


def linear_regression(y: pd.Series, x: pd.DataFrame) -> RegressionResultsWrapper:
    """
    OLS regression of `y` on `x`, with an intercept - the general building
    block every other function here specializes.

    Args:
        y: Dependent variable (e.g. asset returns).
        x: Independent variable(s) (e.g. one or more factor return series).

    Returns:
        A fitted `statsmodels` `RegressionResultsWrapper` (`.params`,
        `.rsquared`, `.pvalues`, `.summary()`, ...). The intercept is labeled "const".
    """
    if isinstance(x, pd.Series):
        x = x.to_frame(x.name or "x")
    aligned = pd.concat([y.rename("y"), x], axis=1, join="inner").dropna()
    ensure_min_observations(aligned, x.shape[1] + 2, "y/x (aligned)")
    design = sm.add_constant(aligned.drop(columns="y"))
    return sm.OLS(aligned["y"], design).fit()


def rolling_regression(y: pd.Series, x: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling-window OLS of `y` on `x` - a rolling alpha and rolling factor
    loading(s), via `statsmodels.regression.rolling.RollingOLS`.

    Args:
        y: Dependent variable (e.g. asset returns).
        x: Independent variable(s), same length/index as `y`.
        window: Rolling window size, in periods.

    Returns:
        A DataFrame indexed like `y`, with one column "alpha" (the rolling
        intercept) plus one column per column of `x` (its rolling loading).
        The first `window - 1` rows are NaN.
    """
    if isinstance(x, pd.Series):
        x = x.to_frame(x.name or "x")
    aligned = pd.concat([y.rename("y"), x], axis=1, join="inner").dropna()
    ensure_min_observations(aligned, window, "y/x (aligned)")
    design = sm.add_constant(aligned.drop(columns="y"))
    fitted = RollingOLS(aligned["y"], design, window=window, min_nobs=window, missing="drop").fit()
    return fitted.params.rename(columns={"const": "alpha"})


def capm(
    y: pd.Series,
    benchmark: pd.Series,
    rf: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> RegressionResultsWrapper:
    """
    Single-factor CAPM regression: `(y - rf) = alpha + beta * (benchmark - rf) + e`.

    Args:
        y: Asset/portfolio return series.
        benchmark: Market/benchmark return series.
        rf: Annual risk-free rate, de-annualized internally and subtracted
            from both `y` and `benchmark` before regressing.
        periods_per_year: Periods per year, used to de-annualize `rf`.

    Returns:
        A fitted `RegressionResultsWrapper`; `.params["const"]` is the
        (periodic-scale) alpha, `.params["market"]` is beta.
    """
    r, b = align_pair(y, benchmark, name_a="y", name_b="market")
    rf_period = periodic_rate_from_annual(rf, periods_per_year)
    excess_y = r - rf_period
    excess_b = (b - rf_period).rename("market")
    design = sm.add_constant(excess_b)
    return sm.OLS(excess_y, design).fit()


def fama_french(y: pd.Series, factors: pd.DataFrame, version: int = 3) -> RegressionResultsWrapper:
    """
    Multi-factor Fama-French style regression of `y` on `factors`.

    Args:
        y: Asset/portfolio return series (raw, not pre-adjusted for rf - see
            the "RF" handling below).
        factors: Factor return series as columns. If a column named "RF" is
            present, it's treated as the risk-free rate: `y` is converted to
            excess-of-rf returns and "RF" is dropped from the regressors
            (matching how Ken French's data library distributes factors).
        version: 3 or 5 validates that `factors` has the canonical named
            Fama-French 3-factor (`Mkt-RF`, `SMB`, `HML`) or 5-factor
            (+`RMW`, `CMA`) columns before regressing. Any other value skips
            validation and regresses on whatever columns `factors` provides -
            useful for extended factor sets (momentum, quality, ...) that
            don't have one universally agreed column count.

    Returns:
        A fitted `RegressionResultsWrapper` over `factors`' columns (minus "RF").

    Raises:
        ValueError: If `version` is 3 or 5 and `factors` is missing the
            corresponding canonical columns.
    """
    if version in _CANONICAL_FF_FACTORS:
        missing = [c for c in _CANONICAL_FF_FACTORS[version] if c not in factors.columns]
        if missing:
            raise ValueError(
                f"factors is missing canonical Fama-French {version}-factor column(s): {missing}. "
                "Rename your columns to match, or pass a version not in {3, 5} to skip this check "
                "and regress on whatever columns `factors` has."
            )

    factors = factors.copy()
    y_adj = y.rename(y.name or "y")
    if "RF" in factors.columns:
        aligned_rf = pd.concat([y_adj.rename("y"), factors["RF"]], axis=1, join="inner").dropna()
        y_adj = (aligned_rf["y"] - aligned_rf["RF"]).rename(y.name or "y")
        factors = factors.drop(columns="RF")

    aligned = pd.concat([y_adj.rename("y"), factors], axis=1, join="inner").dropna()
    ensure_min_observations(aligned, factors.shape[1] + 2, "y/factors (aligned)")
    design = sm.add_constant(aligned.drop(columns="y"))
    return sm.OLS(aligned["y"], design).fit()


def factor_attribution(y: pd.Series, factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose `y`'s average per-period return into a contribution from each
    factor (`beta_i * mean(factor_i)`) plus an unexplained "alpha" residual.

    Args:
        y: Asset/portfolio return series.
        factor_returns: Factor return series as columns.

    Returns:
        A DataFrame indexed by factor name (plus "alpha"), with columns
        "beta" (NaN for the alpha row), "contribution" (return units), and
        "pct_of_total" (each contribution's share of `y`'s mean return).
    """
    aligned = pd.concat([y.rename("y"), factor_returns], axis=1, join="inner").dropna()
    ensure_min_observations(aligned, factor_returns.shape[1] + 2, "y/factor_returns (aligned)")
    design = sm.add_constant(aligned.drop(columns="y"))
    model = sm.OLS(aligned["y"], design).fit()

    betas = model.params.drop("const")
    factor_means = aligned[betas.index].mean()
    contributions = betas * factor_means
    alpha = float(model.params["const"])
    total = float(contributions.sum()) + alpha

    out = pd.DataFrame({"beta": betas, "contribution": contributions})
    out.loc["alpha"] = {"beta": np.nan, "contribution": alpha}
    out["pct_of_total"] = out["contribution"] / total if abs(total) > 1e-12 else np.nan
    return out


register(
    Explanation(
        name="linear_regression",
        category="function",
        summary="General-purpose OLS regression with an intercept - the shared engine behind capm, fama_french, and factor_attribution.",
        formula="y = const + beta_1*x_1 + ... + beta_k*x_k + e, fit by ordinary least squares",
        how_to_read="`.params` holds the intercept (\"const\") and each regressor's coefficient; `.rsquared` is the fraction of y's variance explained; `.pvalues` flags which coefficients are statistically distinguishable from zero.",
        good_vs_bad="Higher R-squared means the regressors explain more of y's variation - not necessarily a 'better' model for decision-making, since a high R-squared factor exposure you don't want is still a risk, not a feature.",
        caveats="Assumes linearity, i.i.d. Normal-ish errors, and no perfect multicollinearity among regressors (statsmodels will warn or fail on the latter). Standard errors are the OLS defaults (not Newey-West/HAC) - serially correlated residuals (common in overlapping or illiquid-asset returns) mean reported p-values/standard errors are too optimistic.",
    )
)

register(
    Explanation(
        name="rolling_regression",
        category="function",
        summary="Runs linear_regression repeatedly over a sliding window, showing how alpha and factor loadings evolve through time instead of assuming they're constant.",
        formula="Same as linear_regression, refit independently on each window of `window` consecutive observations.",
        how_to_read="Each row is a snapshot regression using only the trailing `window` periods - a rolling beta drifting from 0.8 to 1.3 means the asset's market sensitivity genuinely changed, not that the single-window estimate was wrong.",
        good_vs_bad="Stable rolling coefficients support treating the single full-sample regression as reliable; large swings mean the 'true' beta/alpha is time-varying and a single static regression is misleading.",
        caveats="Smaller windows react faster to real regime changes but are noisier; larger windows are smoother but lag genuine shifts. The first `window - 1` rows are NaN (not enough history yet). No overlap-induced serial correlation adjustment, same caveat as linear_regression.",
    )
)

register(
    Explanation(
        name="capm",
        category="model",
        summary="The Capital Asset Pricing Model: explains an asset's excess return purely through its sensitivity (beta) to the market's excess return.",
        formula="(y - rf) = alpha + beta*(benchmark - rf) + e",
        how_to_read="alpha (\"const\") is the average return unexplained by market exposure, per period - a positive, statistically significant alpha is the classic (if often overstated/data-mined) signature of genuine skill. beta is the market sensitivity, matching metrics.risk.beta's Cov/Var formula when rf=0 - see the cross-validation tests for the exact relationship at rf != 0.",
        good_vs_bad="A high R-squared means the single-factor market model explains most of the return variation, so alpha/beta are estimated with more confidence; a low R-squared means most of the return is idiosyncratic and the CAPM lens isn't telling you much.",
        caveats="CAPM is a one-factor simplification - real returns are also driven by size, value, momentum, and other priced factors (see fama_french). A single historical beta assumes a stable linear relationship that can shift in market stress. alpha here is periodic-scale (not annualized) and estimated with OLS standard errors (no HAC/Newey-West correction for serial correlation).",
        interpret=lambda m: f"alpha={m.params['const']:+.4%}/period (p={m.pvalues['const']:.3f}), beta={m.params['market']:.2f}, R2={m.rsquared:.2f}",
    )
)

register(
    Explanation(
        name="fama_french",
        category="model",
        summary="Extends CAPM with additional priced factors (size, value, and optionally profitability/investment) to explain more of an asset's return than the market factor alone.",
        formula="y[- rf if an RF column is supplied] = alpha + beta_Mkt*Mkt-RF + beta_SMB*SMB + beta_HML*HML [+ beta_RMW*RMW + beta_CMA*CMA] + e",
        how_to_read="Each beta is the asset's loading on that factor - e.g. a positive SMB beta means the asset behaves like a small-cap stock, a positive HML beta means it behaves like a value stock. alpha is what's left unexplained by all factors together, usually smaller (in magnitude) than a single-factor CAPM alpha on the same asset.",
        good_vs_bad="A smaller, less significant alpha than the CAPM alpha on the same asset means the extra factors are absorbing return that looked like 'skill' under CAPM but is really a known style exposure. Large, significant factor betas identify which styles actually drive the asset's returns.",
        caveats="version is validated against the canonical named columns only for 3 or 5 - any other value skips validation and regresses on whatever columns `factors` has (useful for momentum/quality-extended sets, but you're responsible for the column meanings then). Factor returns must be in the same frequency/units as `y`. Same OLS standard-error caveat as linear_regression.",
        interpret=lambda m: f"alpha={m.params['const']:+.4%}/period (p={m.pvalues['const']:.3f}), R2={m.rsquared:.2f}, factors={[c for c in m.params.index if c != 'const']}",
    )
)

register(
    Explanation(
        name="factor_attribution",
        category="function",
        summary="Splits an asset's average historical return into how much came from each factor's average level versus how much is unexplained (alpha).",
        formula="contribution_i = beta_i * mean(factor_i); alpha = const; total = sum(contribution_i) + alpha",
        how_to_read="pct_of_total shows each factor's share of the asset's average return - a factor can have a large beta but a small contribution if that factor's own average return was near zero over the sample.",
        good_vs_bad="A return mostly attributed to well-known factors (high combined pct_of_total on Mkt/size/value/etc.) means the strategy is largely a repackaged factor bet; a return mostly attributed to 'alpha' means it isn't explained by the factors you supplied - which could be genuine skill or just an omitted factor.",
        caveats="Attribution is only as complete as the factor set you provide - an important omitted factor shows up as inflated alpha, not as its own line. Uses each factor's simple historical mean over the same sample as the regression; both are subject to the same estimation-risk caveats as expected_returns.",
    )
)
