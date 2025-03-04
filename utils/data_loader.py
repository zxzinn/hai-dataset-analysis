import polars as pl
from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np

class HAIDataLoader:
    """Data loader for HAI Security Dataset using polars for efficient processing"""
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize the data loader
        
        Args:
            base_path: Path to the HAI dataset directory
        """
        self.base_path = Path(base_path)
        
    def load_csv(self, file_path: Union[str, Path], lazy: bool = True) -> pl.LazyFrame:
        """
        Load CSV file using polars
        
        Args:
            file_path: Path to the CSV file
            lazy: Whether to load the data lazily (default: True)
            
        Returns:
            Polars LazyFrame containing the data
        """
        df = pl.scan_csv(file_path, separator=';')
        return df
        
    def process_time_column(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Process the time column to datetime format
        
        Args:
            df: Input LazyFrame
            
        Returns:
            LazyFrame with processed time column
        """
        return df.with_column(
            pl.col("time").str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S")
        )
    
    def load_dataset(self, 
                    version: str,
                    dataset_type: str = "train",
                    file_number: Optional[int] = None) -> pl.LazyFrame:
        """
        Load specific HAI dataset version
        
        Args:
            version: Dataset version (e.g., "20.07", "21.03", "22.04", "23.05")
            dataset_type: Type of dataset ("train" or "test")
            file_number: File number to load (e.g., 1 for train1.csv)
            
        Returns:
            LazyFrame containing the requested dataset
        """
        # Construct file path
        if version == "23.05":
            prefix = "hai-"
        else:
            prefix = ""
            
        file_name = f"{prefix}{dataset_type}{file_number}.csv"
        file_path = self.base_path / f"hai-{version}" / file_name
        
        # Load and process data
        df = self.load_csv(file_path)
        df = self.process_time_column(df)
        
        return df
    
    def create_windows(self, 
                      df: pl.LazyFrame,
                      window_size: int,
                      stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from time series data
        
        Args:
            df: Input LazyFrame
            window_size: Size of each window
            stride: Stride between windows
            
        Returns:
            Tuple of (X, y) where X contains the windowed features and y contains the labels
        """
        # Collect data into numpy arrays
        data = df.collect().to_numpy()
        
        # Separate features and labels
        X = data[:, :-4]  # All columns except last 4 (attack labels)
        y = data[:, -4:]  # Last 4 columns (attack labels)
        
        # Create windows
        n_samples = (len(X) - window_size) // stride + 1
        X_windows = np.zeros((n_samples, window_size, X.shape[1]))
        y_windows = np.zeros((n_samples, window_size, y.shape[1]))
        
        for i in range(n_samples):
            start_idx = i * stride
            end_idx = start_idx + window_size
            X_windows[i] = X[start_idx:end_idx]
            y_windows[i] = y[start_idx:end_idx]
            
        return X_windows, y_windows
    
    def split_train_val(self, 
                       X: np.ndarray,
                       y: np.ndarray,
                       val_ratio: float = 0.2,
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets
        
        Args:
            X: Input features
            y: Labels
            val_ratio: Ratio of validation data
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        np.random.seed(random_state)
        
        # Get number of samples for validation
        n_val = int(len(X) * val_ratio)
        
        # Generate random indices
        indices = np.random.permutation(len(X))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        # Split data
        X_train = X[train_indices]
        X_val = X[val_indices]
        y_train = y[train_indices]
        y_val = y[val_indices]
        
        return X_train, X_val, y_train, y_val
