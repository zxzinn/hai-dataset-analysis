"""
Visualization utilities for HAI dataset analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Union, Tuple
import polars as pl

def set_style():
    """Set default plotting style"""
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def plot_control_loop(df: pl.DataFrame,
                     loop_name: str,
                     sp_col: str,
                     pv_col: str,
                     cv_cols: List[str],
                     time_col: str = 'timestamp',
                     window: int = 1000) -> None:
    """
    Plot control loop variables (SP, PV, CV)
    
    Args:
        df: Input dataframe
        loop_name: Name of the control loop
        sp_col: Column name for setpoint
        pv_col: Column name for process variable
        cv_cols: List of column names for control variables
        time_col: Column name for timestamp
        window: Number of samples to plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Sample data
    sample_df = df.sample(window)
    
    # Plot SP and PV
    axes[0].plot(sample_df[time_col], sample_df[sp_col], 'r-', label='Setpoint')
    axes[0].plot(sample_df[time_col], sample_df[pv_col], 'b-', label='Process Variable')
    axes[0].set_title(f'{loop_name} Control Loop: SP vs PV')
    axes[0].legend()
    
    # Plot CVs
    for cv in cv_cols:
        axes[1].plot(sample_df[time_col], sample_df[cv], label=cv)
    axes[1].set_title(f'{loop_name} Control Variables')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_error_distribution(df: pl.DataFrame,
                          error_cols: List[str],
                          bins: int = 50) -> None:
    """
    Plot error distributions for multiple control loops
    
    Args:
        df: Input dataframe
        error_cols: List of error column names
        bins: Number of histogram bins
    """
    n_cols = len(error_cols)
    fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4*n_cols))
    
    for i, col in enumerate(error_cols):
        sns.histplot(df[col], bins=bins, kde=True, ax=axes[i])
        axes[i].set_title(f'{col} Distribution')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df: pl.DataFrame,
                          columns: List[str],
                          title: str = 'Correlation Matrix') -> None:
    """
    Plot correlation matrix heatmap
    
    Args:
        df: Input dataframe
        columns: List of columns to include
        title: Plot title
    """
    corr_matrix = df.select(columns).to_pandas().corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.show()

def plot_anomaly_scores(scores: np.ndarray,
                       labels: np.ndarray,
                       title: str = 'Anomaly Detection Results') -> None:
    """
    Plot anomaly scores with attack labels
    
    Args:
        scores: Array of anomaly scores
        labels: Array of attack labels (0/1)
        title: Plot title
    """
    plt.figure(figsize=(15, 6))
    
    # Plot scores
    plt.plot(scores, label='Anomaly Score', alpha=0.7)
    
    # Highlight attack periods
    plt.fill_between(range(len(scores)),
                    min(scores), max(scores),
                    where=labels == 1,
                    color='red', alpha=0.3,
                    label='Attack Period')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.show()

def plot_control_performance(df: pl.DataFrame,
                           loop_vars: Dict[str, Union[str, List[str]]],
                           window: int = 1000) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot control performance metrics
    
    Args:
        df: Input dataframe
        loop_vars: Dictionary containing SP, PV, CV column names
        window: Number of samples to plot
    
    Returns:
        fig: Figure object
        axes: Axes object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sample data
    sample_df = df.sample(window)
    
    # SP vs PV scatter
    axes[0,0].scatter(sample_df[loop_vars['SP']],
                     sample_df[loop_vars['PV']],
                     alpha=0.1)
    axes[0,0].set_title('SP vs PV')
    axes[0,0].set_xlabel('Setpoint')
    axes[0,0].set_ylabel('Process Variable')
    
    # Error vs CV scatter
    cv = loop_vars['CV'][0] if isinstance(loop_vars['CV'], list) else loop_vars['CV']
    axes[0,1].scatter(sample_df[loop_vars['SP']] - sample_df[loop_vars['PV']],
                     sample_df[cv],
                     alpha=0.1)
    axes[0,1].set_title('Error vs CV')
    axes[0,1].set_xlabel('Control Error')
    axes[0,1].set_ylabel('Control Variable')
    
    # Time series
    axes[1,0].plot(sample_df[loop_vars['SP']], label='SP')
    axes[1,0].plot(sample_df[loop_vars['PV']], label='PV')
    axes[1,0].set_title('SP and PV Time Series')
    axes[1,0].legend()
    
    if isinstance(loop_vars['CV'], list):
        for cv in loop_vars['CV']:
            axes[1,1].plot(sample_df[cv], label=cv)
    else:
        axes[1,1].plot(sample_df[loop_vars['CV']], label=loop_vars['CV'])
    axes[1,1].set_title('CV Time Series')
    axes[1,1].legend()
    
    plt.tight_layout()
    return fig, axes
