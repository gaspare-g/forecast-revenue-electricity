"""
Descriptive analysis functions for exploratory data analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats


def get_missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of missing values in the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        Summary with missing count and percentage
    """
    missing_data = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing_Count": df.isnull().sum().values,
            "Missing_Percentage": (df.isnull().sum().values / len(df) * 100).round(2),
        }
    )

    return (
        missing_data[missing_data["Missing_Count"] > 0]
        .sort_values("Missing_Count", ascending=False)
        .reset_index(drop=True)
    )


def get_numerical_descriptive_stats(
    df: pd.DataFrame, columns: list = None
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for numerical columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Specific columns to analyze. If None, analyzes all numerical columns

    Returns:
    --------
    pd.DataFrame
        Descriptive statistics including min, Q1, median, Q3, max, mean, std, and distribution shape
    """
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()

    stats_list = []

    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            stats_dict = {
                "Column": col,
                "Count": df[col].count(),
                "Missing": df[col].isnull().sum(),
                "Mean": df[col].mean(),
                "Std": df[col].std(),
                "Min": df[col].min(),
                "Q1": df[col].quantile(0.25),
                "Median": df[col].median(),
                "Q3": df[col].quantile(0.75),
                "Max": df[col].max(),
                "Skewness": df[col].skew(),
                "Kurtosis": df[col].kurtosis(),
            }
            stats_list.append(stats_dict)

    return pd.DataFrame(stats_list)


def get_categorical_summary(df: pd.DataFrame, columns: list = None) -> dict:
    """
    Get summary of categorical variables.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Specific columns to analyze. If None, analyzes all categorical columns

    Returns:
    --------
    dict
        Dictionary with categorical summaries (unique counts and value counts)
    """
    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()

    categorical_summary = {}

    for col in columns:
        if col in df.columns:
            categorical_summary[col] = {
                "Unique_Count": df[col].nunique(),
                "Missing": df[col].isnull().sum(),
                "Value_Counts": df[col].value_counts().to_dict(),
            }

    return categorical_summary


def print_descriptive_summary(df: pd.DataFrame):
    """
    Print a comprehensive descriptive summary of the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    print("=" * 80)
    print("MISSING VALUES")
    print("=" * 80)
    missing_df = get_missing_values_summary(df)
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("No missing values found!")
    print()

    print("=" * 80)
    print("NUMERICAL VARIABLES STATISTICS")
    print("=" * 80)
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numerical_cols:
        stats_df = get_numerical_descriptive_stats(df, numerical_cols)
        print(stats_df.to_string(index=False))
    print()

    print("=" * 80)
    print("CATEGORICAL VARIABLES SUMMARY")
    print("=" * 80)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        cat_summary = get_categorical_summary(df, categorical_cols)
        for col, summary in cat_summary.items():
            print(f"\n{col}:")
            print(f"  Unique Values: {summary['Unique_Count']}")
            print(f"  Missing: {summary['Missing']}")
            print("  Top Values:")
            for val, count in list(summary["Value_Counts"].items())[:10]:
                print(f"    {val}: {count}")
    print()


def detect_strong_outliers(
    df: pd.DataFrame, columns: list = None, z_score_threshold: float = 3.5
) -> pd.DataFrame:
    """
    Detect and return rows with strong outliers using z-score method.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Specific columns to check for outliers. If None, uses default outlier-prone columns
    z_score_threshold : float, optional
        Z-score threshold for outlier detection (default: 3.5)

    Returns:
    --------
    pd.DataFrame
        DataFrame containing rows with outliers
    """
    if columns is None:
        columns = [
            "RRP",
            "RRP_positive",
            "demand_neg_RRP",
            "RRP_negative",
            "frac_at_neg_RRP",
        ]

    # Filter to only existing columns
    columns = [col for col in columns if col in df.columns]

    if not columns:
        print("No valid columns found for outlier detection")
        return pd.DataFrame()

    # Create a copy for z-score calculation
    df_for_zscore = df[columns].copy()

    # Calculate z-scores
    z_scores = np.abs(stats.zscore(df_for_zscore, nan_policy="omit"))

    # Find rows where any column has z-score > threshold
    outlier_mask = (z_scores > z_score_threshold).any(axis=1)

    outlier_rows = df[outlier_mask].copy()

    if len(outlier_rows) > 0:
        print(f"\n{'=' * 80}")
        print(f"STRONG OUTLIERS DETECTED (z-score > {z_score_threshold})")
        print(f"{'=' * 80}")
        print(f"Found {len(outlier_rows)} rows with outliers\n")
        print(outlier_rows.to_string())
        print()
    else:
        print(f"No strong outliers found (z-score > {z_score_threshold})")

    return outlier_rows


def impute_missing_by_month(
    df: pd.DataFrame, columns: list = None, date_col: str = None
) -> pd.DataFrame:
    """
    Impute missing values using the average of values from the same month.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Specific columns to impute. If None, uses rainfall and solar_exposure
    date_col : str, optional
        Column containing date/time information for extracting month.
        If None, tries to find datetime column automatically

    Returns:
    --------
    pd.DataFrame
        DataFrame with imputed missing values
    """
    if columns is None:
        columns = ["rainfall", "solar_exposure"]

    # Filter to only existing columns
    columns = [col for col in columns if col in df.columns]

    if not columns:
        print("No valid columns found for imputation")
        return df.copy()

    # Find date column if not specified
    if date_col is None:
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        if datetime_cols:
            date_col = datetime_cols[0]
        else:
            print("No datetime column found for month extraction")
            return df.copy()

    df_imputed = df.copy()

    # Extract month from date column
    df_imputed["month"] = pd.to_datetime(df_imputed[date_col]).dt.month

    print(f"\n{'=' * 80}")
    print(f"IMPUTING MISSING VALUES BY MONTH")
    print(f"{'=' * 80}\n")

    for col in columns:
        if col in df_imputed.columns:
            missing_count = df_imputed[col].isnull().sum()

            if missing_count > 0:
                print(f"Column: {col}")
                print(f"Missing values: {missing_count}")

                # For each month, calculate the average and impute
                for month in df_imputed["month"].unique():
                    month_mask = df_imputed["month"] == month
                    month_avg = df_imputed.loc[month_mask, col].mean()

                    # Impute missing values for this month
                    impute_mask = month_mask & df_imputed[col].isnull()
                    imputed_count = impute_mask.sum()

                    if imputed_count > 0:
                        df_imputed.loc[impute_mask, col] = month_avg
                        print(
                            f"  Month {month}: imputed {imputed_count} values (avg: {month_avg:.2f})"
                        )

                print()
            else:
                print(f"Column: {col} - No missing values\n")

    # Drop the temporary month column
    df_imputed = df_imputed.drop("month", axis=1)

    return df_imputed

def check_daily_completeness(
    df: pd.DataFrame,
    date_col: str = None,
    verbose: bool = True
) -> dict:
    """
    Check whether all daily timestamps are present between min and max date.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_col : str, optional
        Column containing dates. If None, uses index.
    verbose : bool
        Whether to print diagnostics.

    Returns
    -------
    dict
        {
            "is_complete": bool,
            "missing_days": DatetimeIndex,
            "duplicate_days": DatetimeIndex,
            "expected_days": int,
            "actual_days": int
        }
    """

    # Extract datetime index
    if date_col:
        idx = pd.to_datetime(df[date_col])
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex or provide date_col.")
        idx = df.index

    idx = pd.DatetimeIndex(idx).sort_values()

    # Check duplicates
    duplicates = idx[idx.duplicated()]

    # Build full expected range
    full_range = pd.date_range(start=idx.min(), end=idx.max(), freq="D")

    # Find missing days
    missing = full_range.difference(idx)

    result = {
        "is_complete": len(missing) == 0 and len(duplicates) == 0,
        "missing_days": missing,
        "duplicate_days": duplicates,
        "expected_days": len(full_range),
        "actual_days": len(idx),
    }

    if verbose:
        print("---- Daily Completeness Check ----")
        print(f"Start date: {idx.min()}")
        print(f"End date:   {idx.max()}")
        print(f"Expected days: {len(full_range)}")
        print(f"Actual days:   {len(idx)}")
        print(f"Missing days:  {len(missing)}")
        print(f"Duplicate days:{len(duplicates)}")
        print(f"Complete:      {result['is_complete']}")

    return result