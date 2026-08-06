"""
Shared numeric constants used across PortPy's metrics, models, and strategies.
"""

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365
WEEKS_PER_YEAR = 52
MONTHS_PER_YEAR = 12
QUARTERS_PER_YEAR = 4

"""
Default minimum acceptable return (MAR) for downside-risk metrics.
"""
DEFAULT_RISK_FREE_RATE = 0.0
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_MAR = 0.0

"""
Small constant used to guard against division by zero without changing results materially.
"""
EPSILON = 1e-12
