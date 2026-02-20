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
    path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset downloaded to: {path}")
    
    # Load all CSV files from the dataset
    csv_files = list(Path(path).glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    
    # Load the first CSV file
    df = pd.read_csv(csv_files[0])
    return df


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
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Handle datetime columns
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'year' in col.lower()]
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except:
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
    return df.select_dtypes(include=['number']).columns.tolist()


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
    return df.select_dtypes(include=['object']).columns.tolist()
