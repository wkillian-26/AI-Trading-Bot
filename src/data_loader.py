"""
data_loader.py

Utilities for downloading and preparing market data.
"""

from typing import Optional
import pandas as pd
import yfinance as yf


def load_price_data(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV price data using Yahoo Finance.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. 'AAPL', 'MSFT', 'SPY'.
    period : str, optional
        Lookback period, e.g. '1y', '2y', '5d'. Ignored if `start` is provided.
    interval : str, optional
        Bar interval, e.g. '1d', '1h', '15m'.
    start : str, optional
        Explicit start date in 'YYYY-MM-DD' format. If provided, overrides `period`.
    end : str, optional
        Explicit end date in 'YYYY-MM-DD' format.
    auto_adjust : bool, optional
        Whether to auto-adjust prices for splits/dividends.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by timestamp with columns like:
        ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] (depending on options).
    """
    if start:
        # Use explicit date range
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
        )
    else:
        # Use relative period
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
        )

    # Clean up
    df = df.dropna().copy()

    if df.empty:
        raise ValueError(f"No data returned for {symbol}. Check symbol or parameters.")

    return df


if __name__ == "__main__":
    # Quick manual test when running this file directly:
    test_symbol = "AAPL"
    data = load_price_data(test_symbol, period="6mo", interval="1d")
    print(f"Downloaded {len(data)} rows for {test_symbol}.")
    print(data.tail())
