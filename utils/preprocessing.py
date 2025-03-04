"""
Data preprocessing utilities for HAI dataset
"""

import polars as pl
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

def load_dataset(data_dir: Path,
                version: str = 'hai-22.04') -> Tuple[List[pl.LazyFrame], List[pl.LazyFrame]]:
    """
    Load HAI dataset files using Polars lazy evaluation
    
    Args:
        data_dir: Path to dataset directory
        version: Dataset version to load
        
    Returns:
        tuple: (train_dfs, test_dfs)
    """
    version_dir = data_dir / version
    
    # Load training files
    train_files = list(version_dir.glob('train*.csv'))
    train_dfs = [pl.scan_csv(f) for f in train_files]
    
    # Load test files
    test_files = list(version_dir.glob('test*.csv'))
    test_dfs = [pl.scan_csv(f) for f in test_files]
    
    return train_dfs, test_dfs

def preprocess_dataframe(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Preprocess a single dataframe
    
    Args:
        df: Input dataframe
        
    Returns:
        Preprocessed dataframe
    """
    return df.with_columns([
        # Convert timestamp to datetime
        pl.col('timestamp').str.to_datetime(),
        
        # Handle missing values
        pl.all().fill_null(strategy='forward')
    ])

def extract_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Extract features for each control loop
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with extracted features
    """
    control_loops = {
        'P1-PC': {
            'SP': 'P1_B2016',
            'PV': 'P1_PIT01',
            'CV': ['P1_PCV01D', 'P1_PCV02D']
        },
        'P1-LC': {
            'SP': 'P1_B3004',
            'PV': 'P1_LIT01',
            'CV': ['P1_LCV01D']
        },
        'P1-FC': {
            'SP': 'P1_B3005',
            'PV': 'P1_FT03',
            'CV': ['P1_FCV03D']
        },
        'P1-TC': {
            'SP': 'P1_B4022',
            'PV': 'P1_TIT01',
            'CV': ['P1_FCV01D', 'P1_FCV02D']
        }
    }
    
    features = []
    
    for loop_name, vars in control_loops.items():
        # Calculate control error
        features.append(
            (pl.col(vars['SP']) - pl.col(vars['PV'])).alias(f'{loop_name}_error')
        )
        
        # Calculate moving statistics
        window_sizes = [10, 30, 60]
        for size in window_sizes:
            features.extend([
                pl.col(vars['PV']).rolling_mean(size).alias(f'{loop_name}_PV_mean_{size}'),
                pl.col(vars['PV']).rolling_std(size).alias(f'{loop_name}_PV_std_{size}')
            ])
            
            # Calculate CV statistics
            for cv in vars['CV']:
                features.extend([
                    pl.col(cv).rolling_mean(size).alias(f'{cv}_mean_{size}'),
                    pl.col(cv).rolling_std(size).alias(f'{cv}_std_{size}')
                ])
    
    return df.with_columns(features)

def create_sequences(data: np.ndarray,
                    sequence_length: int,
                    step: int = 1,
                    shuffle: bool = True) -> np.ndarray:
    """
    Create sequences from time series data
    
    Args:
        data: Input data array
        sequence_length: Length of each sequence
        step: Step size between sequences
        shuffle: Whether to shuffle sequences
        
    Returns:
        Array of sequences
    """
    sequences = []
    for i in range(0, len(data) - sequence_length + 1, step):
        sequences.append(data[i:i+sequence_length])
    
    sequences = np.array(sequences)
    
    if shuffle:
        np.random.shuffle(sequences)
    
    return sequences

def split_sequences(sequences: np.ndarray,
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences into train, validation and test sets
    
    Args:
        sequences: Input sequences
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        
    Returns:
        tuple: (train_sequences, val_sequences, test_sequences)
    """
    n = len(sequences)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    train_sequences = sequences[:train_idx]
    val_sequences = sequences[train_idx:val_idx]
    test_sequences = sequences[val_idx:]
    
    return train_sequences, val_sequences, test_sequences

def get_attack_labels(df: pl.DataFrame,
                     control_loops: List[str]) -> Dict[str, np.ndarray]:
    """
    Get attack labels for different control loops
    
    Args:
        df: Input dataframe
        control_loops: List of control loop names
        
    Returns:
        Dictionary mapping control loops to attack labels
    """
    labels = {}
    
    # Get overall attack labels
    labels['all'] = df.select('attack').to_numpy().flatten()
    
    # Get control loop specific labels
    for loop in control_loops:
        col = f'{loop.lower()}_attack'
        if col in df.columns:
            labels[loop] = df.select(col).to_numpy().flatten()
            
    return labels

def save_processed_data(df: pl.LazyFrame,
                       filename: str,
                       save_dir: Path) -> None:
    """
    Save processed dataframe to parquet format
    
    Args:
        df: Dataframe to save
        filename: Output filename
        save_dir: Output directory
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f'{filename}.parquet'
    df.collect().write_parquet(output_path)
    print(f'Saved to {output_path}')

def load_processed_data(filename: str,
                       data_dir: Path) -> pl.DataFrame:
    """
    Load processed data from parquet file
    
    Args:
        filename: Input filename
        data_dir: Input directory
        
    Returns:
        Loaded dataframe
    """
    input_path = data_dir / f'{filename}.parquet'
    return pl.read_parquet(input_path)
