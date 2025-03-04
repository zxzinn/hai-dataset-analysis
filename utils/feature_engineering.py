import polars as pl
import numpy as np
from typing import List, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

class FeatureEngineer:
    """Feature engineering for HAI Security Dataset"""
    
    def __init__(self):
        """Initialize feature engineering components"""
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        
    def fit_standard_scaler(self, df: pl.LazyFrame, exclude_cols: List[str]) -> None:
        """
        Fit StandardScaler on numerical columns
        
        Args:
            df: Input LazyFrame
            exclude_cols: Columns to exclude from scaling
        """
        # Collect data and convert to numpy
        data = df.collect().to_numpy()
        
        # Create and fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        
    def transform_with_scaler(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Transform data using fitted scaler
        
        Args:
            df: Input LazyFrame
            
        Returns:
            Transformed LazyFrame
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_standard_scaler first.")
            
        # Transform data
        data = df.collect().to_numpy()
        scaled_data = self.scaler.transform(data)
        
        # Convert back to LazyFrame
        return pl.LazyFrame(scaled_data, schema=df.columns)
    
    def extract_statistical_features(self, 
                                   df: pl.LazyFrame,
                                   window_size: int) -> pl.LazyFrame:
        """
        Extract statistical features using rolling windows
        
        Args:
            df: Input LazyFrame
            window_size: Size of rolling window
            
        Returns:
            LazyFrame with additional statistical features
        """
        return df.with_columns([
            pl.col("*").rolling_mean(window_size).suffix("_mean"),
            pl.col("*").rolling_std(window_size).suffix("_std"),
            pl.col("*").rolling_min(window_size).suffix("_min"),
            pl.col("*").rolling_max(window_size).suffix("_max"),
            pl.col("*").rolling_skew(window_size).suffix("_skew"),
            pl.col("*").rolling_kurtosis(window_size).suffix("_kurtosis")
        ])
    
    def extract_time_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Extract time-based features from datetime column
        
        Args:
            df: Input LazyFrame
            
        Returns:
            LazyFrame with additional time features
        """
        return df.with_columns([
            pl.col("time").dt.hour().alias("hour"),
            pl.col("time").dt.minute().alias("minute"),
            pl.col("time").dt.second().alias("second"),
            pl.col("time").dt.day().alias("day"),
            pl.col("time").dt.weekday().alias("weekday")
        ])
    
    def select_features(self, 
                       X: np.ndarray,
                       y: np.ndarray,
                       n_features: int = 10) -> np.ndarray:
        """
        Select most important features using mutual information
        
        Args:
            X: Input features
            y: Target labels
            n_features: Number of features to select
            
        Returns:
            Selected features array
        """
        # Initialize and fit feature selector
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Store selected feature indices
        self.selected_features = self.feature_selector.get_support()
        
        return X_selected
    
    def calculate_feature_correlations(self, df: pl.LazyFrame) -> pl.DataFrame:
        """
        Calculate correlation matrix for features
        
        Args:
            df: Input LazyFrame
            
        Returns:
            DataFrame containing correlation matrix
        """
        return df.select([
            pl.all().corr().alias("correlation_matrix")
        ]).collect()
    
    def detect_outliers(self,
                       df: pl.LazyFrame,
                       columns: List[str],
                       threshold: float = 3.0) -> pl.Series:
        """
        Detect outliers using Z-score method
        
        Args:
            df: Input LazyFrame
            columns: Columns to check for outliers
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Boolean series indicating outlier rows
        """
        z_scores = df.select([
            ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).abs()
            for col in columns
        ])
        
        return z_scores.select(
            pl.any_horizontal(pl.all() > threshold).alias("is_outlier")
        ).collect().to_series()
