"""Realized volatility estimators from daily price bars, annualized with 252 trading days.

* Close-to-close: standard deviation of log returns.
* Parkinson (1980): uses the daily high/low range, about five times as efficient for a driftless process.
* Garman-Klass (1980): uses open, high, low and close.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def close_to_close(close: pd.Series, window: int = 21) -> pd.Series:
    returns = np.log(close).diff()
    return returns.rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS)


def parkinson(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    hl2 = np.log(high / low) ** 2
    return np.sqrt(hl2.rolling(window).mean() / (4.0 * math.log(2.0)) * TRADING_DAYS)


def garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    hl = 0.5 * np.log(high / low) ** 2
    co = (2.0 * math.log(2.0) - 1.0) * np.log(close / open_) ** 2
    return np.sqrt((hl - co).rolling(window).mean() * TRADING_DAYS)
