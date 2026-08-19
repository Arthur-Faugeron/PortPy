"""
Shared numeric constants used across PortPy's metrics, models, and strategies.
"""

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365
WEEKS_PER_YEAR = 52
MONTHS_PER_YEAR = 12
QUARTERS_PER_YEAR = 4

"""
Default annual risk-free rate (rf), used as the return hurdle in Sharpe,
Treynor, Alpha, and M2. Independent of DEFAULT_MAR: rf is the riskless
opportunity cost subtracted from *all* returns, while MAR is the shortfall
threshold that only downside-risk metrics (Sortino, semi_variance, ...) care
about. They both default to 0.0 here, but that equality is coincidental -
set them independently when you have a real cash rate and/or a target return
above zero.
"""
DEFAULT_RISK_FREE_RATE = 0.0
DEFAULT_CONFIDENCE_LEVEL = 0.95

"""
Default minimum acceptable return (MAR) for downside-risk metrics (Sortino,
semi_variance, downside_deviation, kappa_three_ratio, upside_potential_ratio).
"""
DEFAULT_MAR = 0.0

"""
Small constant used to guard against division by zero without changing results materially.
"""
EPSILON = 1e-12
