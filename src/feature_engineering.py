"""
Feature engineering utilities for electricity price forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _encode_binary_yn(series: pd.Series) -> pd.Series:
    """
    Encode Y/N values to 1/0.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map({"Y": 1, "N": 0})
        .astype("Int64")
    )


def _add_lag_features(df: pd.DataFrame, col: str, lags: list[int]) -> None:
    """
    Add lag features for a single column in place.
    """
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)


def _add_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows: list[int],
    stats: list[str],
) -> None:
    """
    Add leakage-safe rolling features in place using shift(1).rolling(...).
    """
    shifted = df[col].shift(1)
    for window in windows:
        roller = shifted.rolling(window=window, min_periods=window)
        for stat in stats:
            if stat == "mean":
                df[f"{col}_roll_mean_{window}"] = roller.mean()
            elif stat == "std":
                df[f"{col}_roll_std_{window}"] = roller.std()
            elif stat == "max":
                df[f"{col}_roll_max_{window}"] = roller.max()
            else:
                raise ValueError(f"Unsupported rolling stat: {stat}")


def build_feature_engineering_dataframe(
    df: pd.DataFrame,
    date_col: str = "date",
    demand_col: str = "demand",
    rrp_col: str = "RRP",
    yearly_fourier_order: int = 2,
) -> pd.DataFrame:
    """
    Build a single dataframe with original columns and engineered features.

    Includes:
    - school_day / holiday Y/N encoding to 1/0
    - time features from date: day, month, year
    - day_of_week feature
    - yearly Fourier terms (sin/cos pairs)
    - log transform of demand (log1p)
    - log transform of RRP (signed log1p to support negative values)
    - temperature squared terms (if temperature columns exist)
    - interaction terms: temperature x month
    - lag features for demand and RRP: 1, 2, 7, 14, 28
    - rolling features for demand:
      mean(7,14,28), std(7,14,28)
    - rolling features for RRP:
      mean(7,14,28), std(7,14,28), max(7,14,28)

    Rolling features are computed with:
    shift first, then roll
    to avoid data leakage.
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in dataframe.")
    if demand_col not in df.columns:
        raise ValueError(f"Column '{demand_col}' not found in dataframe.")
    if rrp_col not in df.columns:
        raise ValueError(f"Column '{rrp_col}' not found in dataframe.")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values(date_col)
    out = out.set_index(date_col)

    if "school_day" in out.columns:
        out["school_day"] = _encode_binary_yn(out["school_day"])
    if "holiday" in out.columns:
        out["holiday"] = _encode_binary_yn(out["holiday"])

    out["day"] = out.index.day
    out["month"] = out.index.month
    out["year"] = out.index.year
    out["day_of_week"] = out.index.dayofweek

    # Yearly seasonality via Fourier terms.
    t = np.arange(len(out), dtype=float)
    yearly_period = 365.25
    for k in range(1, yearly_fourier_order + 1):
        out[f"fourier_year_sin_{k}"] = np.sin(2 * np.pi * k * t / yearly_period)
        out[f"fourier_year_cos_{k}"] = np.cos(2 * np.pi * k * t / yearly_period)

    out[f"{demand_col}_log"] = np.log1p(out[demand_col])

    # Signed log transform handles negative RRP values while reducing skew.
    out[f"{rrp_col}_log"] = np.sign(out[rrp_col]) * np.log1p(np.abs(out[rrp_col]))

    # Temperature nonlinear and interaction effects.
    temp_cols = [col for col in out.columns if "temperature" in col.lower()]
    for col in temp_cols:
        out[f"{col}_sq"] = out[col] ** 2
        out[f"{col}_x_month"] = out[col] * out["month"]

    lags = [1, 2, 7, 14, 28]
    windows = [7, 14, 28]

    _add_lag_features(out, demand_col, lags)
    _add_lag_features(out, rrp_col, lags)

    _add_rolling_features(out, demand_col, windows, stats=["mean", "std"])
    _add_rolling_features(out, rrp_col, windows, stats=["mean", "std", "max"])

    return out
