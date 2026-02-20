"""
Descriptive analysis functions for exploratory data analysis.
"""

import pandas as pd
import numpy as np


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
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    
    return missing_data[missing_data['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    ).reset_index(drop=True)


def get_numerical_descriptive_stats(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
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
        Descriptive statistics including mean, median, std, min, max, distribution shape
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    stats_list = []
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            stats_dict = {
                'Column': col,
                'Count': df[col].count(),
                'Missing': df[col].isnull().sum(),
                'Mean': df[col].mean(),
                'Median': df[col].median(),
                'Std': df[col].std(),
                'Min': df[col].min(),
                'Q1': df[col].quantile(0.25),
                'Q3': df[col].quantile(0.75),
                'Max': df[col].max(),
                'Skewness': df[col].skew(),
                'Kurtosis': df[col].kurtosis()
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
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    categorical_summary = {}
    
    for col in columns:
        if col in df.columns:
            categorical_summary[col] = {
                'Unique_Count': df[col].nunique(),
                'Missing': df[col].isnull().sum(),
                'Value_Counts': df[col].value_counts().to_dict()
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
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numerical_cols:
        stats_df = get_numerical_descriptive_stats(df, numerical_cols)
        print(stats_df.to_string(index=False))
    print()
    
    print("=" * 80)
    print("CATEGORICAL VARIABLES SUMMARY")
    print("=" * 80)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        cat_summary = get_categorical_summary(df, categorical_cols)
        for col, summary in cat_summary.items():
            print(f"\n{col}:")
            print(f"  Unique Values: {summary['Unique_Count']}")
            print(f"  Missing: {summary['Missing']}")
            print(f"  Top Values:")
            for val, count in list(summary['Value_Counts'].items())[:10]:
                print(f"    {val}: {count}")
    print()
