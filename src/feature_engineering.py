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
    demand_trend: pd.Series | None = None,
    rrp_trend: pd.Series | None = None,
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
    - decomposition trend features: demand_trend, RRP_trend (optional)
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

    if demand_trend is not None:
        out[f"{demand_col}_trend"] = pd.Series(demand_trend).reindex(out.index)
    if rrp_trend is not None:
        out[f"{rrp_col}_trend"] = pd.Series(rrp_trend).reindex(out.index)

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


def _first_mode(series: pd.Series) -> int | float | None:
    """
    Deterministic mode helper for grouped categorical inference.
    """
    clean = series.dropna()
    if clean.empty:
        return None
    mode_values = clean.mode()
    if mode_values.empty:
        return None
    return mode_values.iloc[0]


def build_production_feature_dataframe(
    df_features: pd.DataFrame,
    forecast_horizon_days: int | None = None,
    forecast_until: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Append future daily rows and estimate required production features.

    Historical rows are preserved as-is; only appended future rows are filled.
    """
    if not isinstance(df_features.index, pd.DatetimeIndex):
        raise ValueError("df_features must be indexed by a DatetimeIndex.")
    if forecast_until is not None and forecast_horizon_days is not None:
        raise ValueError(
            "Provide only one of forecast_horizon_days or forecast_until, not both."
        )
    if forecast_until is None and forecast_horizon_days is None:
        raise ValueError(
            "Provide one of forecast_horizon_days or forecast_until."
        )

    base = df_features.sort_index().copy()
    hist_index = base.index.copy()

    last_day = hist_index.max()
    if forecast_until is not None:
        forecast_until_ts = pd.Timestamp(forecast_until)
        if forecast_until_ts <= last_day:
            raise ValueError(
                "forecast_until must be strictly greater than the last historical date "
                f"({last_day.date()})."
            )
        future_index = pd.date_range(
            start=last_day + pd.Timedelta(days=1),
            end=forecast_until_ts,
            freq="D",
        )
    else:
        if forecast_horizon_days < 1:
            raise ValueError("forecast_horizon_days must be >= 1.")
        future_index = pd.date_range(
            start=last_day + pd.Timedelta(days=1),
            periods=forecast_horizon_days,
            freq="D",
        )
    future_df = pd.DataFrame(index=future_index, columns=base.columns, dtype=float)
    out = pd.concat([base, future_df], axis=0)

    # Weather proxies from historical daily climatology (month-day average across years).
    weather_cols = ["min_temperature", "max_temperature", "solar_exposure", "rainfall"]
    hist = out.loc[hist_index]
    hist_day_key = hist.index.strftime("%m-%d")
    future_day_key = future_index.strftime("%m-%d")
    future_month = pd.Index(future_index.month, dtype="int64")

    for col in weather_cols:
        if col not in out.columns:
            continue

        day_means = hist.groupby(hist_day_key)[col].mean()
        month_means = hist.groupby(hist.index.month)[col].mean()
        overall_mean = float(hist[col].mean())

        est = pd.Series(future_day_key).map(day_means)
        est = est.fillna(pd.Series(future_month).map(month_means))
        est = est.fillna(overall_mean)
        out.loc[future_index, col] = est.values

    # Calendar features from future index.
    for cal_col, values in {
        "day": future_index.day,
        "month": future_index.month,
        "year": future_index.year,
        "day_of_week": future_index.dayofweek,
    }.items():
        if cal_col in out.columns:
            out.loc[future_index, cal_col] = values

    # Holiday estimation using Victoria public holidays.
    if "holiday" in out.columns:
        try:
            import holidays
        except ImportError as exc:
            raise ImportError(
                "Package 'holidays' is required. Install it with: pip install holidays"
            ) from exc

        au_holidays = holidays.Australia(subdiv="VIC")
        holiday_values = [1 if d.date() in au_holidays else 0 for d in future_index]
        out.loc[future_index, "holiday"] = holiday_values

    # School-day estimation from historical mode by holiday/day/month/day_of_week.
    if "school_day" in out.columns:
        required_cols = ["holiday", "day", "month", "day_of_week", "school_day"]
        missing = [c for c in required_cols if c not in out.columns]
        if missing:
            raise ValueError(
                "Cannot estimate school_day. Missing required columns: "
                + ", ".join(missing)
            )

        hist_school = out.loc[hist_index, required_cols].dropna()
        if hist_school.empty:
            raise ValueError("Cannot estimate school_day: no historical rows available.")

        map_key4 = (
            hist_school.groupby(["holiday", "day", "month", "day_of_week"])["school_day"]
            .apply(_first_mode)
            .to_dict()
        )
        map_key3 = (
            hist_school.groupby(["holiday", "month", "day_of_week"])["school_day"]
            .apply(_first_mode)
            .to_dict()
        )
        map_key2 = (
            hist_school.groupby(["holiday", "day_of_week"])["school_day"]
            .apply(_first_mode)
            .to_dict()
        )
        map_key1 = (
            hist_school.groupby(["day_of_week"])["school_day"]
            .apply(_first_mode)
            .to_dict()
        )
        global_mode = _first_mode(hist_school["school_day"])
        if global_mode is None:
            global_mode = 0

        for d in future_index:
            hol = out.at[d, "holiday"]
            day = out.at[d, "day"]
            month = out.at[d, "month"]
            dow = out.at[d, "day_of_week"]

            guess = map_key4.get((hol, day, month, dow))
            if guess is None:
                guess = map_key3.get((hol, month, dow))
            if guess is None:
                guess = map_key2.get((hol, dow))
            if guess is None:
                guess = map_key1.get((dow,))
            if guess is None:
                guess = global_mode

            out.at[d, "school_day"] = guess

    # Preserve numeric integer-like flags for appended rows.
    for col in ["holiday", "school_day", "day", "month", "year", "day_of_week"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # If Fourier terms exist, extend them consistently for future dates.
    fourier_k = []
    for col in out.columns:
        if col.startswith("fourier_year_sin_"):
            try:
                fourier_k.append(int(col.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    if fourier_k:
        yearly_period = 365.25
        start_t = len(hist_index)
        t_future = np.arange(start_t, start_t + len(future_index), dtype=float)
        for k in sorted(set(fourier_k)):
            sin_col = f"fourier_year_sin_{k}"
            cos_col = f"fourier_year_cos_{k}"
            if sin_col in out.columns:
                out.loc[future_index, sin_col] = np.sin(
                    2 * np.pi * k * t_future / yearly_period
                )
            if cos_col in out.columns:
                out.loc[future_index, cos_col] = np.cos(
                    2 * np.pi * k * t_future / yearly_period
                )

    return out
