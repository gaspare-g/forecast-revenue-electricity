"""
Modeling functions for time series analysis and decomposition.
"""
from typing import Union, List, Tuple
from statsmodels.tsa.seasonal import MSTL
import pandas as pd
import numpy as np



def decompose_time_series(
    df: pd.DataFrame,
    column: str,
    date_col: str = None,
    seasonal_periods: Union[List[int], Tuple[int, ...]] = (7, 365),
) -> dict:
    """
    Decompose time series into trend, seasonal, and residual components using MSTL.
    """

    # Ensure periods is array-like (MSTL requires array-like)
    periods = np.asarray(seasonal_periods, dtype=int)

    # Prepare the series
    if date_col and date_col in df.columns:
        ts = df.set_index(date_col)[column]
        ts = ts.asfreq("D")
    else:
        ts = df[column]

    # Ensure datetime index if possible
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise ValueError("Time series index must be a DatetimeIndex for MSTL.")

    # Fit MSTL with correct argument name
    model = MSTL(ts, periods=periods)
    result = model.fit()

    return {
        "original": ts,
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
        "model": result,
    }


def get_trend_statistics(trend_component: pd.Series) -> dict:
    """
    Calculate statistics for the trend component.

    Parameters:
    -----------
    trend_component : pd.Series
        Trend component from decomposition

    Returns:
    --------
    dict
        Dictionary with trend statistics
    """
    return {
        "mean": trend_component.mean(),
        "std": trend_component.std(),
        "min": trend_component.min(),
        "max": trend_component.max(),
        "slope": (trend_component.iloc[-1] - trend_component.iloc[0])
        / len(trend_component),
        "direction": "increasing"
        if trend_component.iloc[-1] > trend_component.iloc[0]
        else "decreasing",
    }


def get_seasonality_statistics(seasonal_component: pd.Series) -> dict:
    """
    Calculate statistics for the seasonal component.

    Parameters:
    -----------
    seasonal_component : pd.Series
        Seasonal component from decomposition

    Returns:
    --------
    dict
        Dictionary with seasonality statistics
    """
    return {
        "mean": seasonal_component.mean(),
        "std": seasonal_component.std(),
        "min": seasonal_component.min(),
        "max": seasonal_component.max(),
        "amplitude": seasonal_component.max() - seasonal_component.min(),
        "variability": seasonal_component.std(),
    }


def get_residual_statistics(residual_component: pd.Series) -> dict:
    """
    Calculate statistics for the residual component.

    Parameters:
    -----------
    residual_component : pd.Series
        Residual component from decomposition

    Returns:
    --------
    dict
        Dictionary with residual statistics
    """
    return {
        "mean": residual_component.mean(),
        "std": residual_component.std(),
        "min": residual_component.min(),
        "max": residual_component.max(),
        "rmse": np.sqrt((residual_component**2).mean()),
        "relative_rmse": np.sqrt((residual_component**2).mean())
        / np.abs(residual_component).mean(),
    }


def print_decomposition_summary(decomposition_dict: dict, column_name: str):
    """
    Print a comprehensive summary of the decomposition results.

    Parameters:
    -----------
    decomposition_dict : dict
        Dictionary from decompose_time_series function
    column_name : str
        Name of the column being analyzed
    """
    trend_stats = get_trend_statistics(decomposition_dict["trend"])
    seasonal_stats = get_seasonality_statistics(decomposition_dict["seasonal"])
    residual_stats = get_residual_statistics(decomposition_dict["residual"])

    print(f"\n{'=' * 80}")
    print(f"TIME SERIES DECOMPOSITION ANALYSIS - {column_name}")
    print(f"{'=' * 80}\n")

    print("TREND COMPONENT")
    print("-" * 80)
    print(f"  Mean:           {trend_stats['mean']:.4f}")
    print(f"  Std Dev:        {trend_stats['std']:.4f}")
    print(f"  Min:            {trend_stats['min']:.4f}")
    print(f"  Max:            {trend_stats['max']:.4f}")
    print(f"  Slope (daily):  {trend_stats['slope']:.6f}")
    print(f"  Direction:      {trend_stats['direction']}")
    print()

    print("SEASONAL COMPONENT")
    print("-" * 80)
    print(f"  Mean:           {seasonal_stats['mean']:.4f}")
    print(f"  Std Dev:        {seasonal_stats['std']:.4f}")
    print(f"  Min:            {seasonal_stats['min']:.4f}")
    print(f"  Max:            {seasonal_stats['max']:.4f}")
    print(f"  Amplitude:      {seasonal_stats['amplitude']:.4f}")
    print(f"  Variability:    {seasonal_stats['variability']:.4f}")
    print()

    print("RESIDUAL COMPONENT")
    print("-" * 80)
    print(f"  Mean:           {residual_stats['mean']:.4f}")
    print(f"  Std Dev:        {residual_stats['std']:.4f}")
    print(f"  Min:            {residual_stats['min']:.4f}")
    print(f"  Max:            {residual_stats['max']:.4f}")
    print(f"  RMSE:           {residual_stats['rmse']:.4f}")
    print(f"  Relative RMSE:  {residual_stats['relative_rmse']:.4f}")
    print()
