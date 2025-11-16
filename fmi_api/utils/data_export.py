"""
Data export and import utilities
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def save_to_parquet(
    df: pd.DataFrame,
    filename: str,
    output_dir: str = 'data'
) -> str:
    """
    Save DataFrame to Parquet format.
    
    Args:
        df: DataFrame to save
        filename: Output filename (without extension)
        output_dir: Output directory (default: 'data')
        
    Returns:
        Full path to saved file
        
    Example:
        >>> df = get_observations(...)
        >>> filepath = save_to_parquet(df, 'helsinki_weather_2023')
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add extension if not present
    if not filename.endswith('.parquet'):
        filename = f'{filename}.parquet'
    
    # Full path
    filepath = output_path / filename
    
    # Save to parquet
    df.to_parquet(filepath, index=False, engine='pyarrow')
    
    print(f"Saved {len(df)} records to {filepath}")
    
    return str(filepath)


def load_from_parquet(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from Parquet file.
    
    Args:
        filepath: Path to parquet file
        
    Returns:
        Loaded DataFrame
        
    Example:
        >>> df = load_from_parquet('data/helsinki_weather_2023.parquet')
    """
    df = pd.read_parquet(filepath, engine='pyarrow')
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def save_to_csv(
    df: pd.DataFrame,
    filename: str,
    output_dir: str = 'data'
) -> str:
    """
    Save DataFrame to CSV format.
    
    Args:
        df: DataFrame to save
        filename: Output filename (without extension)
        output_dir: Output directory (default: 'data')
        
    Returns:
        Full path to saved file
        
    Example:
        >>> df = get_observations(...)
        >>> filepath = save_to_csv(df, 'helsinki_weather_2023')
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add extension if not present
    if not filename.endswith('.csv'):
        filename = f'{filename}.csv'
    
    # Full path
    filepath = output_path / filename
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    
    print(f"Saved {len(df)} records to {filepath}")
    
    return str(filepath)


def load_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Loaded DataFrame
        
    Example:
        >>> df = load_from_csv('data/helsinki_weather_2023.csv')
    """
    df = pd.read_csv(filepath)
    
    # Try to parse datetime columns
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
    
    print(f"Loaded {len(df)} records from {filepath}")
    return df


