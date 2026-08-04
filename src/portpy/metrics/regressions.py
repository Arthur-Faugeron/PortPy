"""OLS regression helpers built on statsmodels - used for factor exposure and rolling beta analysis."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.regression.rolling import RollingOLS

from portpy.explain import Explanation, register
from portpy.utils.validation import ensure_min_observations

__all__ = [
    "linear_regression", 
    "rolling_regression", 
    "regression_summary"
]


def _prepare_design(y: pd.Series, x: pd.Series | pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    if isinstance(x, pd.Series):
        x = x.to_frame(x.name or "x")
    joined = pd.concat([y.rename("y"), x], axis=1, join="inner").dropna()
    ensure_min_observations(joined, x.shape[1] + 2, "regression data")
    return joined["y"], sm.add_constant(joined.drop(columns="y"))


def linear_regression(y: pd.Series, x: pd.Series | pd.DataFrame) -> RegressionResultsWrapper:
    """Fit `y ~ const + x` via Ordinary Least Squares and return the full statsmodels result object."""
    y_aligned, design = _prepare_design(y, x)
    return sm.OLS(y_aligned, design).fit()


def rolling_regression(y: pd.Series, x: pd.Series | pd.DataFrame, window: int) -> pd.DataFrame:
    """Refit `y ~ const + x` on a trailing window at every point - a DataFrame of rolling coefficients."""
    y_aligned, design = _prepare_design(y, x)
    model = RollingOLS(y_aligned, design, window=window, min_nobs=window)
    res = model.fit()
    return res.params.dropna(how="all")


def regression_summary(model: RegressionResultsWrapper) -> pd.DataFrame:
    """Tidy summary table (coef, std err, t-stat, p-value, 95% CI) for a fitted `linear_regression` result."""
    conf = model.conf_int()
    conf.columns = ["conf_low", "conf_high"]
    summary = pd.DataFrame(
        {
            "coef": model.params,
            "std_err": model.bse,
            "t_stat": model.tvalues,
            "p_value": model.pvalues,
        }
    ).join(conf)
    summary.attrs["portpy_explanation"] = "regression_summary"
    summary.attrs["r_squared"] = float(model.rsquared)
    summary.attrs["adj_r_squared"] = float(model.rsquared_adj)
    return summary


register(
    Explanation(
        name="regression_summary",
        category="metric",
        summary="A tidy table of an OLS regression's coefficients, significance, and confidence intervals - e.g. exposure to a benchmark or factor set.",
        formula="coef +/- std_err, t = coef/std_err, p_value from a t-distribution, 95% CI = coef +/- 1.96*std_err (approx)",
        how_to_read=(
            "For a coefficient row: `coef` is the estimated sensitivity, `p_value` below 0.05 "
            "conventionally means it's statistically distinguishable from zero. Check `r_squared` "
            "(in `.attrs['r_squared']`) for how much variance the whole model explains."
        ),
        good_vs_bad="A significant (low p-value), stable-signed coefficient with a sensible economic interpretation is 'good' evidence of a real exposure - a high-p-value coefficient close to zero should be treated as noise.",
        caveats="OLS assumes a linear, stable relationship and roughly well-behaved residuals - always sanity-check with a scatter plot, especially with financial return data's fat tails.",
    )
)
