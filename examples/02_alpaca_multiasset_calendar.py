"""02 - Multi-asset portfolios with Alpaca data: calendars and currencies.

This example tackles the two trickiest data-engineering problems in portfolio
analysis, both called out explicitly in PortPy's design:

    1. **Calendar mismatch**: crypto trades 24/7, equities/commodities/bonds trade ~5 days/week.
       Naively combining them produces NaNs (or, worse, silently misaligned
       returns if you don't notice).
    2. **Currency mismatch**: not every asset is quoted in the same currency.
       PortPy assumes a single base currency by default, but if you *do* have
       dated FX rates, it can convert for you.

Data sources are deliberately mixed to mirror a real institutional multi-asset portfolio over 3 years:
    - US Equities (AAPL, MSFT), Gold (GLD), Long-Term Treasuries (TLT), Real Estate (VNQ),
      and Crypto (BTC/USD, ETH/USD) from Alpaca.
    - Euro-denominated European stock (SAP.DE) and its FX rate (EURUSD=X) from yfinance.

Requires a `.env` file with `ALPACA_KEY` and `ALPACA_SECRET` (see `.env.example`).

Run:
    python examples/02_alpaca_multiasset_calendar.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

from portpy import Portfolio
from portpy.core import AssetClass, align_calendars, calendar_coverage_report, convert_to_base_currency

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

LOOKBACK_DAYS = 3 * 365  # 3-year historical lookback window


def fetch_alpaca_bars(client, request_cls, symbols: list[str], **client_kwargs) -> pd.DataFrame:
    """Fetch daily close prices for `symbols` from an Alpaca historical data client.

    Alpaca returns a (symbol, timestamp) MultiIndex with tz-aware UTC
    timestamps - this reshapes it into the wide, tz-naive, one-column-per-asset
    DataFrame that `Portfolio` (and `align_calendars`) expect.
    """
    request = request_cls(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=LOOKBACK_DAYS),
        **client_kwargs,
    )
    bars = client.get_stock_bars(request) if isinstance(client, StockHistoricalDataClient) else client.get_crypto_bars(request)
    df = bars.df["close"].unstack(level="symbol")
    # Alpaca timestamps carry time-of-day (05:00 UTC for stocks, midnight for
    # crypto) and a UTC tz - normalize both away so dates line up across assets.
    df.index = df.index.tz_localize(None).normalize()
    return df


def main() -> None:
    load_dotenv()
    api_key, api_secret = os.getenv("ALPACA_KEY"), os.getenv("ALPACA_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Set ALPACA_KEY and ALPACA_SECRET in a .env file to run this example.")

    print("=" * 75)
    print("FETCHING 3-YEAR MULTI-ASSET DATA (Equities, Crypto, Gold, Bonds, Real Estate)")
    print("=" * 75)

    # 1. Fetch US Equities, Commodities (Gold), Bonds, and Real Estate from Alpaca (Mon-Fri)
    us_symbols = ["AAPL", "MSFT", "GLD", "TLT", "VNQ"]
    print(f"Fetching US assets from Alpaca: {', '.join(us_symbols)} (Mon-Fri calendar)...")
    stock_client = StockHistoricalDataClient(api_key, api_secret)
    us_assets = fetch_alpaca_bars(stock_client, StockBarsRequest, us_symbols)

    # 2. Fetch Crypto from Alpaca (24/7 continuous calendar)
    crypto_symbols = ["BTC/USD", "ETH/USD"]
    print(f"Fetching Crypto from Alpaca: {', '.join(crypto_symbols)} (7 days/week)...")
    crypto_client = CryptoHistoricalDataClient()
    crypto = fetch_alpaca_bars(crypto_client, CryptoBarsRequest, crypto_symbols)
    crypto.columns = [col.replace("/USD", "") for col in crypto.columns]

    # 3. Fetch European Equity (quoted in EUR) and EUR/USD FX exchange rate from yfinance (3y)
    print("Fetching SAP.DE (quoted in EUR) and EURUSD=X (3y) from yfinance...")
    sap_raw = yf.download("SAP.DE", period="3y", auto_adjust=True, progress=False)
    eurusd_raw = yf.download("EURUSD=X", period="3y", auto_adjust=True, progress=False)

    sap_close = sap_raw["Close"].squeeze()
    eurusd_close = eurusd_raw["Close"].squeeze()

    # Ensure tz-naive and normalized dates for clean alignment
    if hasattr(sap_close.index, "tz") and sap_close.index.tz is not None:
        sap_close.index = sap_close.index.tz_localize(None)
    sap_close.index = sap_close.index.normalize()

    if hasattr(eurusd_close.index, "tz") and eurusd_close.index.tz is not None:
        eurusd_close.index = eurusd_close.index.tz_localize(None)
    eurusd_close.index = eurusd_close.index.normalize()

    # --- Step 1: currency conversion (per-asset, calendar-independent) ---
    # SAP.DE prices are in EUR; convert to USD *before* combining with the other
    # (already-USD) assets, so every column below is apples-to-apples.
    sap_prices = sap_close.to_frame("SAP")
    fx_frame = eurusd_close.to_frame("EUR")
    sap_usd = convert_to_base_currency(
        sap_prices, fx_frame, asset_currencies={"SAP": "EUR"}, base_currency="USD"
    )
    print(
        f"\nSAP.DE converted EUR->USD: last EUR close={sap_prices['SAP'].iloc[-1]:.2f}, "
        f"USD close={sap_usd['SAP'].iloc[-1]:.2f} (EURUSD={eurusd_close.iloc[-1]:.4f})"
    )

    # --- Step 2: see the calendar mismatch before doing anything about it ---
    combined_raw = pd.concat([us_assets, crypto, sap_usd], axis=1, sort=True)
    print("\n" + "=" * 75)
    print("CALENDAR COVERAGE REPORT (before alignment over 3 years)")
    print("=" * 75)
    print(calendar_coverage_report(combined_raw))
    print(f"\nRows with at least one NaN: {combined_raw.isna().any(axis=1).sum()} / {len(combined_raw)}")

    # --- Step 3: two legitimate ways to resolve it, with different tradeoffs ---
    conservative = align_calendars(combined_raw, method="intersection")
    print(
        f"\n'intersection' method: {len(conservative)} rows - business days only, "
        "crypto's weekend moves are dropped entirely."
    )

    crypto_first = align_calendars(combined_raw, method="ffill_union")
    print(
        f"'ffill_union' method:  {len(crypto_first)} rows - keeps every day crypto trades (~365 days/year), "
        "carrying Friday closes through the weekend."
    )

    # --- Step 4: build the actual Portfolio on the calendar we chose ---
    # Broad diversified multi-asset allocation:
    weights = {
        "AAPL": 0.15,  # US Equity (Tech / Large-Cap Growth)
        "MSFT": 0.10,  # US Equity (Tech / Mega-Cap)
        "SAP": 0.10,   # International Equity (Germany / EUR converted)
        "GLD": 0.15,   # Commodity (Gold)
        "TLT": 0.15,   # Fixed Income (Long-Term US Treasury Bonds)
        "VNQ": 0.10,   # Real Estate (US REITs)
        "BTC": 0.15,   # Crypto (Bitcoin)
        "ETH": 0.10,   # Crypto (Ethereum)
    }

    asset_classes = {
        "AAPL": AssetClass.EQUITY,
        "MSFT": AssetClass.EQUITY,
        "SAP": AssetClass.EQUITY,
        "GLD": AssetClass.COMMODITY,
        "TLT": AssetClass.BOND,
        "VNQ": AssetClass.REAL_ESTATE,
        "BTC": AssetClass.CRYPTO,
        "ETH": AssetClass.CRYPTO,
    }

    portfolio = Portfolio(
        crypto_first,
        weights=weights,
        name="3-Year Global Multi-Asset All-Weather Portfolio",
        frequency=365,  # trades every calendar day because of crypto
        asset_classes=asset_classes,
    )
    print(f"\n{portfolio}")
    print(f"Asset classes: {portfolio.asset_classes}")

    print("\n" + "=" * 75)
    print("HEADLINE METRICS (3-Year History, 365 periods/year - this portfolio never sleeps)")
    print("=" * 75)
    for metric_name, result in portfolio.metrics.tearsheet_summary().items():
        print(f"  {metric_name:22s} {float(result):>10.4f}   ({result.interpretation})")

    print("\n" + "=" * 75)
    print("DIVERSIFICATION & CORRELATION ACROSS ASSET CLASSES")
    print("=" * 75)
    cov = portfolio.metrics.covariance_matrix(annualized=True)
    print(cov)

    corr = portfolio.metrics.correlation_matrix()
    print("\nCorrelation Matrix:")
    print(corr)

    div_ratio = portfolio.metrics.diversification_ratio(as_result=True)
    print("\nDiversification Ratio Explanation:")
    div_ratio.explain()


if __name__ == "__main__":
    main()
