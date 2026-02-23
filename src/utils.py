"""
Utility functions for data loading and preprocessing.
"""

import pandas as pd
import kagglehub
from dotenv import load_dotenv
from pathlib import Path


def load_kaggle_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load dataset from Kaggle using kagglehub.

    Parameters:
    -----------
    dataset_name : str
        The Kaggle dataset name (e.g., "username/dataset-name")

    Returns:
    --------
    pd.DataFrame
        Combined dataframe from all CSV files in the dataset
    """
    load_dotenv()
    local_fallback = Path("complete_dataset.csv")

    try:
        path = kagglehub.dataset_download(dataset_name)
        print("Dataset downloaded.")

        # Load all CSV files from the dataset
        csv_files = list(Path(path).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in downloaded Kaggle dataset.")

        # Load the first CSV file
        return pd.read_csv(csv_files[0])
    except Exception as kaggle_error:
        if local_fallback.exists():
            print("Kaggle download unavailable. Using local complete_dataset.csv.")
            return pd.read_csv(local_fallback)
        raise RuntimeError(
            "Kaggle download failed and local fallback complete_dataset.csv was not found."
        ) from kaggle_error


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe

    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    df = df.copy()

    # Convert data types if needed
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Handle datetime columns
    date_columns = [
        col for col in df.columns if "date" in col.lower() or "year" in col.lower()
    ]
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except:  # noqa: E722
            pass

    return df


def get_numerical_columns(df: pd.DataFrame) -> list:
    """
    Get list of numerical columns from dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    list
        List of numerical column names
    """
    return df.select_dtypes(include=["number"]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list:
    """
    Get list of categorical columns from dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    list
        List of categorical column names
    """
    return df.select_dtypes(include=["object"]).columns.tolist()
