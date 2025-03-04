"""
HAI Dataset Preprocessing Module

This module provides functions for preprocessing HAI (Hardware-in-the-Loop Augmented ICS) 
security datasets. It includes functions for efficient data loading, feature engineering,
and visualization.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import time
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import gc

def get_file_info(file_path):
    """
    Get basic information about a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        dict: Dictionary containing file information
    """
    file_path = Path(file_path)
    file_size_bytes = file_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Read the first line to get column names
    with open(file_path, 'r') as f:
        header = f.readline().strip()
    
    num_columns = len(header.split(','))
    
    # Estimate number of rows based on file size and header size
    # This is a rough estimate
    if num_columns > 0:
        avg_row_size = file_size_bytes / 1000  # Sample size
        with open(file_path, 'r') as f:
            f.readline()  # Skip header
            sample_data = ""
            for _ in range(1000):
                line = f.readline()
                if not line:
                    break
                sample_data += line
        
        if len(sample_data) > 0:
            avg_row_size = len(sample_data) / 1000
            estimated_rows = int(file_size_bytes / avg_row_size)
        else:
            estimated_rows = 0
    else:
        estimated_rows = 0
    
    return {
        'file_name': file_path.name,
        'file_size_bytes': file_size_bytes,
        'file_size_mb': file_size_mb,
        'num_columns': num_columns,
        'estimated_rows': estimated_rows
    }

def lazy_load_csv(file_path, batch_size=100000):
    """
    Lazily load a CSV file using Polars.
    
    Args:
        file_path (str): Path to the CSV file
        batch_size (int): Number of rows to load at once
        
    Returns:
        pl.LazyFrame: Lazy DataFrame
    """
    # Detect if the first column is a timestamp
    with open(file_path, 'r') as f:
        header = f.readline().strip().split(',')
        first_col = header[0].lower()
    
    # If the first column is a timestamp, parse it as datetime
    if 'time' in first_col or 'date' in first_col:
        df_lazy = pl.scan_csv(file_path)
        df_lazy = df_lazy.with_column(
            pl.col(header[0]).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )
    else:
        df_lazy = pl.scan_csv(file_path)
    
    return df_lazy

def process_in_chunks(file_path, process_func, chunk_size=100000, **kwargs):
    """
    Process a large CSV file in chunks.
    
    Args:
        file_path (str): Path to the CSV file
        process_func (function): Function to apply to each chunk
        chunk_size (int): Number of rows in each chunk
        **kwargs: Additional arguments to pass to process_func
        
    Returns:
        list: List of processed chunks
    """
    results = []
    
    # Get total file size for progress tracking
    file_size = os.path.getsize(file_path)
    processed_size = 0
    
    # Create a reader
    reader = pd.read_csv(file_path, chunksize=chunk_size)
    
    # Process each chunk
    for i, chunk in enumerate(tqdm(reader, desc="Processing chunks")):
        # Convert pandas DataFrame to polars DataFrame
        pl_chunk = pl.from_pandas(chunk)
        
        # Process the chunk
        processed_chunk = process_func(pl_chunk, **kwargs)
        results.append(processed_chunk)
        
        # Update progress
        processed_size += chunk.memory_usage(deep=True).sum()
        
        # Clean up to free memory
        del chunk
        gc.collect()
    
    return results

def save_to_efficient_format(df, output_path, format='parquet'):
    """
    Save DataFrame to an efficient format.
    
    Args:
        df (pl.DataFrame): DataFrame to save
        output_path (str): Path to save the file
        format (str): Format to save the file in ('parquet' or 'arrow')
        
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == 'parquet':
        df.write_parquet(output_path)
    elif format == 'arrow':
        df.write_ipc(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Saved to {output_path}")

def add_time_features(df):
    """
    Add time-based features to a DataFrame.
    
    Args:
        df (pl.LazyFrame or pl.DataFrame): DataFrame with a timestamp column
        
    Returns:
        pl.LazyFrame or pl.DataFrame: DataFrame with time features added
    """
    # Identify the timestamp column (assume it's the first column)
    time_col = df.columns[0]
    
    # Check if df is a LazyFrame or DataFrame
    is_lazy = isinstance(df, pl.LazyFrame)
    
    # Convert to LazyFrame if it's not already
    if not is_lazy:
        df = df.lazy()
    
    # Add time features
    df = df.with_columns([
        pl.col(time_col).dt.hour().alias("hour"),
        pl.col(time_col).dt.day_of_week().alias("day_of_week"),
        pl.col(time_col).dt.day().alias("day"),
        pl.col(time_col).dt.month().alias("month"),
        pl.col(time_col).dt.year().alias("year"),
        (pl.col(time_col).dt.day_of_week() >= 5).alias("is_weekend")
    ])
    
    # Add time of day feature (morning, afternoon, evening, night)
    df = df.with_column(
        pl.when(pl.col("hour") < 6).then(pl.lit("night"))
        .when(pl.col("hour") < 12).then(pl.lit("morning"))
        .when(pl.col("hour") < 18).then(pl.lit("afternoon"))
        .otherwise(pl.lit("evening"))
        .alias("time_of_day")
    )
    
    # Convert back to DataFrame if the input was a DataFrame
    if not is_lazy:
        df = df.collect()
    
    return df

def add_lag_features(df, columns, lags=[1, 5, 10]):
    """
    Add lag features to a DataFrame.
    
    Args:
        df (pl.DataFrame): DataFrame to add lag features to
        columns (list): List of column names to create lag features for
        lags (list): List of lag values
        
    Returns:
        pl.DataFrame: DataFrame with lag features added
    """
    for col in columns:
        for lag in lags:
            df = df.with_column(
                pl.col(col).shift(lag).alias(f"{col}_lag_{lag}")
            )
    
    return df

def add_rolling_features(df, columns, windows=[5, 10, 30], functions=["mean", "std"]):
    """
    Add rolling window features to a DataFrame.
    
    Args:
        df (pl.DataFrame): DataFrame to add rolling features to
        columns (list): List of column names to create rolling features for
        windows (list): List of window sizes
        functions (list): List of functions to apply to rolling windows
        
    Returns:
        pl.DataFrame: DataFrame with rolling features added
    """
    for col in columns:
        for window in windows:
            if "mean" in functions:
                df = df.with_column(
                    pl.col(col).rolling_mean(window_size=window).alias(f"{col}_roll_mean_{window}")
                )
            if "std" in functions:
                df = df.with_column(
                    pl.col(col).rolling_std(window_size=window).alias(f"{col}_roll_std_{window}")
                )
            if "min" in functions:
                df = df.with_column(
                    pl.col(col).rolling_min(window_size=window).alias(f"{col}_roll_min_{window}")
                )
            if "max" in functions:
                df = df.with_column(
                    pl.col(col).rolling_max(window_size=window).alias(f"{col}_roll_max_{window}")
                )
    
    return df

def plot_time_series(df, time_column, columns, title=None, figsize=(15, 10), attack_column=None):
    """
    Plot time series data.
    
    Args:
        df (pl.DataFrame): DataFrame containing time series data
        time_column (str): Name of the timestamp column
        columns (list): List of column names to plot
        title (str): Title of the plot
        figsize (tuple): Figure size
        attack_column (str): Name of the attack label column
        
    Returns:
        None
    """
    plt.figure(figsize=figsize)
    
    # Convert to pandas for easier plotting
    pdf = df.to_pandas()
    
    # Plot each column
    for i, col in enumerate(columns):
        plt.subplot(len(columns), 1, i+1)
        plt.plot(pdf[time_column], pdf[col])
        
        # Highlight attack regions if attack_column is provided
        if attack_column and attack_column in pdf.columns:
            attack_regions = pdf[pdf[attack_column] > 0]
            if not attack_regions.empty:
                plt.scatter(attack_regions[time_column], attack_regions[col], 
                           color='red', label='Attack', alpha=0.5)
                plt.legend()
        
        plt.title(col)
        plt.grid(True)
    
    plt.tight_layout()
    if title:
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.95)
    
    plt.show()

def plot_correlation_matrix(df, columns=None, title="Correlation Matrix", figsize=(12, 10)):
    """
    Plot correlation matrix.
    
    Args:
        df (pl.DataFrame): DataFrame containing data
        columns (list): List of column names to include in the correlation matrix
        title (str): Title of the plot
        figsize (tuple): Figure size
        
    Returns:
        None
    """
    # Select columns if provided
    if columns:
        df_subset = df.select(columns)
    else:
        # Exclude non-numeric columns
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        df_subset = df.select(numeric_cols)
    
    # Convert to pandas for correlation calculation
    pdf = df_subset.to_pandas()
    
    # Calculate correlation matrix
    corr = pdf.corr()
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_distribution(df, columns, bins=50, figsize=(15, 10)):
    """
    Plot distributions of columns.
    
    Args:
        df (pl.DataFrame): DataFrame containing data
        columns (list): List of column names to plot
        bins (int): Number of bins for histograms
        figsize (tuple): Figure size
        
    Returns:
        None
    """
    plt.figure(figsize=figsize)
    
    # Convert to pandas for easier plotting
    pdf = df.to_pandas()
    
    # Plot each column
    for i, col in enumerate(columns):
        plt.subplot(len(columns), 1, i+1)
        sns.histplot(pdf[col], bins=bins, kde=True)
        plt.title(f"Distribution of {col}")
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
