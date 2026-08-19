"""
OLS regression helpers built on statsmodels used for factor exposure and rolling beta analysis.


NOTE:
This is just for experimentation on a first version and is not a critical part of the library. 
I just builded it because I was feeling that I needed to add this to the library 
and experiment with a bit of statsmodels and OLS regression. I don't expect this
to be final as I will explore more regressions and models in the corresponding modules.
"""

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
    """
    Align y and x on their shared index and build an OLS design matrix (x plus a constant column).

    Args:
        y: Dependent variable series.
        x: Explanatory variable(s) - a Series or DataFrame of one or more columns.

    Returns:
        `(y_aligned, design)` - y restricted to overlapping non-NaN rows, and
        the corresponding design matrix with a constant column added.
    """
    if isinstance(x, pd.Series):
        x = x.to_frame(x.name or "x")
    joined = pd.concat([y.rename("y"), x], axis=1, join="inner").dropna()
    ensure_min_observations(joined, x.shape[1] + 2, "regression data")
    return joined["y"], sm.add_constant(joined.drop(columns="y"))


def linear_regression(y: pd.Series, x: pd.Series | pd.DataFrame) -> RegressionResultsWrapper:
    """
    Fit y ~ const + x via Ordinary Least Squares.

    Args:
        y: Dependent variable series.
        x: Explanatory variable(s) - a Series or DataFrame of one or more columns.

    Returns:
        The fitted statsmodels `RegressionResultsWrapper`.
    """
    y_aligned, design = _prepare_design(y, x)
    return sm.OLS(y_aligned, design).fit()


def rolling_regression(y: pd.Series, x: pd.Series | pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Refit y ~ const + x on a trailing window at every point in time.

    Args:
        y: Dependent variable series.
        x: Explanatory variable(s) - a Series or DataFrame of one or more columns.
        window: Number of trailing observations used to fit each window.

    Returns:
        A DataFrame of rolling coefficients, one column per regressor (plus
        the constant), indexed like `y`.
    """
    y_aligned, design = _prepare_design(y, x)
    model = RollingOLS(y_aligned, design, window=window, min_nobs=window)
    res = model.fit()
    return res.params.dropna(how="all")


def regression_summary(model: RegressionResultsWrapper) -> pd.DataFrame:
    """
    Build a tidy summary table for a fitted `linear_regression` result.

    Args:
        model: A fitted statsmodels `RegressionResultsWrapper`, e.g. from
            `linear_regression`.

    Returns:
        A DataFrame indexed by regressor, with columns coef, std_err, t_stat,
        p_value, conf_low, conf_high. Also carries `r_squared` and
        `adj_r_squared` in `.attrs`.
    """
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
        name="linear_regression",
        category="function",
        summary="Fits an OLS regression model of y on one or more explanatory variables using statsmodels.",
        formula="y = beta_0 + beta_1 * x_1 + ... + beta_n * x_n + error",
        how_to_read="Use this when you want a full fitted regression object for diagnostics, coefficient inspection, and further statistical analysis.",
        good_vs_bad="A meaningful regression should have stable coefficients and reasonable explanatory power, but financial relationships can be unstable and do not imply causation.",
    )
)

register(
    Explanation(
        name="rolling_regression",
        category="function",
        summary="Refits an OLS regression over a rolling window so you can inspect how coefficient estimates evolve through time.",
        formula="same as linear_regression, but estimated on a trailing window",
        how_to_read="Use this to see whether the relationship between variables changes over time, which is common in financial data.",
        good_vs_bad="Stable coefficients over time are generally more reliable than coefficients that drift dramatically.",
    )
)

register(
    Explanation(
        name="regression_summary",
        category="function",
        summary="A structured summary table of an OLS regression model containing coefficient estimates, uncertainty measures, statistical tests, and confidence intervals.",
        formula="t_stat = coef / std_err; confidence_interval = coef +/- critical_value * std_err",
        how_to_read=("Each row represents an explanatory variable. coef shows the estimated relationship between the predictor and target variable while controlling for other variables. std_err measures uncertainty around the estimate. t_stat and p_value indicate whether the relationship is statistically distinguishable from zero. The confidence interval shows the plausible range of the coefficient estimate. r_squared and adjusted_r_squared summarize how much variation in the dependent variable is explained by the model."),
        good_vs_bad=("Useful regression results have coefficients that are statistically significant, economically meaningful, and stable across different samples. Large standard errors, unstable coefficients, or poor explanatory power indicate weaker model reliability."),
        caveats=("Regression results show statistical relationships, not causation. OLS assumptions may not hold in financial data due to autocorrelation, heteroskedasticity, non-normal residuals, or changing market regimes. The std_err/t_stat/p_value/conf_low/conf_high columns come from classical OLS standard errors, which are unreliable under the heteroskedasticity and serial correlation common in return data - refit with cov_type=\"HAC\" for robust inference. The minimum observation count enforced at fit time is a technical floor, not a practical one (aim for well over 10x observations per parameter), and multiple regressors are not checked for multicollinearity (e.g. via VIF), which can destabilize individual coefficients even when overall fit looks reasonable."),
        interpret=lambda v: "OLS regression coefficient summary table",
    )
)
