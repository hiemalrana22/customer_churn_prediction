"""
Data loading and preprocessing utilities for Telco Customer Churn dataset.

This module provides functions to load, validate, and perform initial
preprocessing of the Telco Customer Churn data.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings

from .config import TELCO_DATA_FILE, TELCO_DB_FILE, RANDOM_SEED


def load_telco_data(filepath: Path = TELCO_DATA_FILE) -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset with initial preprocessing.
    
    Parameters
    ----------
    filepath : Path
        Path to the CSV file containing the Telco data.
        
    Returns
    -------
    pd.DataFrame
        Loaded and preprocessed DataFrame.
        
    Raises
    ------
    FileNotFoundError
        If the data file doesn't exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found at {filepath}.\n"
            f"Please download the dataset from:\n"
            f"https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            f"and place it in the data/ directory."
        )
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Initial preprocessing
    # TotalCharges should be numeric but may contain spaces
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Convert SeniorCitizen to categorical for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    return df


# ---------------------------------------------------------------------------
# SQL path
#
# The raw file ships as a CSV, but in practice the data sits in a warehouse and
# you pull it with SQL. I mirror that here: load the CSV into a local SQLite
# database once, then read everything back through a query. Keeps the loading
# code honest about where data really comes from, and makes it easy to add
# filters/joins later without touching pandas.
# ---------------------------------------------------------------------------

# The blank TotalCharges values (new accounts, tenure 0) come through as empty
# strings, so NULLIF lets pandas coerce them to NaN cleanly.
DEFAULT_CHURN_QUERY = """
    SELECT
        customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
        PhoneService, MultipleLines, InternetService, OnlineSecurity,
        OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
        StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges,
        NULLIF(TRIM(TotalCharges), '') AS TotalCharges,
        Churn
    FROM customers
    WHERE tenure >= 0
    ORDER BY customerID
"""


def build_sqlite_warehouse(csv_path: Path = TELCO_DATA_FILE,
                           db_path: Path = TELCO_DB_FILE,
                           table: str = "customers",
                           overwrite: bool = False) -> Path:
    """Load the raw CSV into a local SQLite db (once). Returns the db path."""
    if db_path.exists() and not overwrite:
        return db_path

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {csv_path}. Run ./download_dataset.sh first."
        )

    raw = pd.read_csv(csv_path)
    with sqlite3.connect(db_path) as conn:
        raw.to_sql(table, conn, if_exists="replace", index=False)
    return db_path


def load_from_sqlite(query: str = DEFAULT_CHURN_QUERY,
                     db_path: Path = TELCO_DB_FILE,
                     csv_path: Path = TELCO_DATA_FILE) -> pd.DataFrame:
    """
    Load the Telco data through SQLite instead of reading the CSV directly.

    Builds the database on first call, then runs `query` against it. Post-loading
    cleanup matches load_telco_data so downstream code doesn't care which path
    was used.
    """
    build_sqlite_warehouse(csv_path, db_path)

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if df['SeniorCitizen'].dtype != object:
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

    return df


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive information about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing dataset statistics and metadata.
    """
    info = {
        'shape': df.shape,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'columns': df.columns.tolist(),
    }
    
    return info


def check_data_quality(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Perform comprehensive data quality checks.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to check.
    verbose : bool, default=True
        If True, print detailed quality report.
        
    Returns
    -------
    pd.DataFrame
        Data quality report with metrics for each column.
    """
    quality_report = pd.DataFrame({
        'dtype': df.dtypes,
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique_count': df.nunique(),
        'unique_pct': (df.nunique() / len(df) * 100).round(2),
    })
    
    # Add sample values
    quality_report['sample_values'] = df.apply(
        lambda x: str(x.dropna().unique()[:3].tolist())
    )
    
    if verbose:
        print("=" * 80)
        print("DATA QUALITY REPORT")
        print("=" * 80)
        print(f"\nDataset Shape: {df.shape}")
        print(f"Total Missing Values: {df.isnull().sum().sum()}")
        print(f"Duplicate Rows: {df.duplicated().sum()}")
        print("\nColumn-wise Quality Metrics:")
        print("-" * 80)
        print(quality_report.to_string())
        print("=" * 80)
    
    return quality_report


def get_numerical_summary(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Generate statistical summary for numerical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list, optional
        List of numerical columns to summarize. If None, auto-detect.
        
    Returns
    -------
    pd.DataFrame
        Statistical summary including mean, median, std, skewness, etc.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    summary = df[columns].describe().T
    
    # Add additional statistics
    summary['median'] = df[columns].median()
    summary['skewness'] = df[columns].skew()
    summary['kurtosis'] = df[columns].kurtosis()
    summary['missing'] = df[columns].isnull().sum()
    
    # Reorder columns for better readability
    col_order = ['count', 'missing', 'mean', 'median', 'std', 'min', 
                 '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']
    summary = summary[[col for col in col_order if col in summary.columns]]
    
    return summary.round(2)


def get_categorical_summary(df: pd.DataFrame, columns: list = None) -> Dict[str, pd.DataFrame]:
    """
    Generate summary statistics for categorical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list, optional
        List of categorical columns to summarize. If None, auto-detect.
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping column names to their value count DataFrames.
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    summaries = {}
    
    for col in columns:
        value_counts = df[col].value_counts()
        value_pcts = df[col].value_counts(normalize=True) * 100
        
        summaries[col] = pd.DataFrame({
            'count': value_counts,
            'percentage': value_pcts.round(2)
        })
    
    return summaries


def split_features_by_type(df: pd.DataFrame, 
                           target_col: str = 'Churn') -> Tuple[list, list, str]:
    """
    Automatically split features into numerical and categorical.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str, default='Churn'
        Name of the target column to exclude from features.
        
    Returns
    -------
    Tuple[list, list, str]
        Lists of numerical columns, categorical columns, and target column.
    """
    # Exclude ID columns and target
    exclude_cols = ['customerID', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    numerical = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    return numerical, categorical, target_col


def calculate_churn_rate(df: pd.DataFrame, 
                         churn_col: str = 'Churn') -> Dict[str, float]:
    """
    Calculate overall churn rate and related metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    churn_col : str, default='Churn'
        Name of the churn column.
        
    Returns
    -------
    Dict[str, float]
        Dictionary with churn metrics.
    """
    churn_counts = df[churn_col].value_counts()
    total = len(df)
    
    metrics = {
        'total_customers': total,
        'churned_customers': churn_counts.get('Yes', 0),
        'retained_customers': churn_counts.get('No', 0),
        'churn_rate_pct': (churn_counts.get('Yes', 0) / total * 100),
        'retention_rate_pct': (churn_counts.get('No', 0) / total * 100),
    }
    
    return metrics
