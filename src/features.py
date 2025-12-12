"""
features.py

Feature engineering functions for the AI trading bot.
Builds SMA, volatility, RSI, returns, and the prediction target.
"""

import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a price DataFrame and returns it with features + target column.
    Adds:
      - Daily return
      - SMA 5
      - SMA 20
      - Volatility (10-bar rolling std)
      - RSI-14
      - Target (1 if next bar up, else 0)
    """

    df = df.copy()

    # Basic returns
    df["Return"] = df["Close"].pct_change()

    # Moving averages
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # Volatility (rolling 10-bar std of returns)
    df["Vol_10"] = df["Return"].rolling(window=10).std()

    # RSI
    df["RSI_14"] = compute_rsi(df["Close"], period=14)

    # Prediction target - direction of next bar
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)

    # Clean missing data
    df = df.dropna()

    return df


if __name__ == "__main__":
    # Quick manual test if run directly
    import yfinance as yf
    test = yf.download("AAPL", period="6mo", interval="1d", auto_adjust=True)
    test = build_feature_set(test)
    print(test.tail())
