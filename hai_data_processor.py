#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Processor for HAI-21.03 Dataset
This module handles data loading and basic preprocessing for the HAI-21.03 security dataset.
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm


class HAIDataProcessor:
    """
    HAI-21.03 Dataset Processor
    
    This class handles loading and basic preprocessing for the HAI-21.03 industrial
    control system security dataset, which contains sensor and actuator readings along
    with attack labels for anomaly detection.
    """
    
    def __init__(self, data_dir='hai-security-dataset/hai-21.03', 
                 output_dir='hai-security-dataset/processed'):
        """
        Initialize the HAI data processor
        
        Args:
            data_dir (str): Directory containing HAI dataset files
            output_dir (str): Directory to save processed files
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data containers
        self.train_data = {}
        self.test_data = {}
        
        # Set paths for train and test data
        self.train_files = glob.glob(os.path.join(data_dir, 'train*.csv'))
        self.test_files = glob.glob(os.path.join(data_dir, 'test*.csv'))
        
        print(f"Found {len(self.train_files)} train files and {len(self.test_files)} test files")
    
    def load_data(self):
        """
        Load train and test data from CSV files
        """
        # Load training files
        print("Loading train files: ", end="")
        for file in tqdm(self.train_files, desc="Loading train files"):
            file_name = os.path.basename(file).split('.')[0]
            self.train_data[file_name] = self._load_file(file)
            print(f"{file_name}: {self.train_data[file_name].shape[0]} rows, {self.train_data[file_name].shape[1]} columns")
        
        # Load test files
        print("Loading test files: ", end="")
        for file in tqdm(self.test_files, desc="Loading test files"):
            file_name = os.path.basename(file).split('.')[0]
            self.test_data[file_name] = self._load_file(file)
            print(f"{file_name}: {self.test_data[file_name].shape[0]} rows, {self.test_data[file_name].shape[1]} columns")
    
    def _load_file(self, file_path):
        """
        Load and preprocess a single CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Basic preprocessing
        df = self._preprocess_dataframe(df)
        
        return df
    
    def _preprocess_dataframe(self, df):
        """
        Perform basic preprocessing on a dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Convert column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure attack column exists (it might not in training data)
        if 'attack' not in df.columns:
            df['attack'] = 0
        
        return df
    
    def save_processed_data(self):
        """
        Save processed data to CSV files
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save train data
        for file_name, df in self.train_data.items():
            output_path = os.path.join(self.output_dir, f"{file_name}_processed.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {output_path}")
        
        # Save test data
        for file_name, df in self.test_data.items():
            output_path = os.path.join(self.output_dir, f"{file_name}_processed.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {output_path}")
    
    def merge_train_data(self):
        """
        Merge all training dataframes into one
        
        Returns:
            pd.DataFrame: Merged training dataframe
        """
        if not self.train_data:
            print("No training data loaded. Call load_data() first.")
            return None
        
        # Merge all training dataframes
        merged_df = pd.concat(list(self.train_data.values()), axis=0, ignore_index=True)
        print(f"Merged training data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        
        return merged_df
    
    def merge_test_data(self):
        """
        Merge all test dataframes into one
        
        Returns:
            pd.DataFrame: Merged test dataframe
        """
        if not self.test_data:
            print("No test data loaded. Call load_data() first.")
            return None
        
        # Merge all test dataframes
        merged_df = pd.concat(list(self.test_data.values()), axis=0, ignore_index=True)
        print(f"Merged test data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        
        return merged_df
    
    def get_feature_columns(self, df=None):
        """
        Get the feature columns (excluding time and attack columns)
        
        Args:
            df (pd.DataFrame, optional): Dataframe to extract columns from.
                If None, uses the first training dataframe.
                
        Returns:
            list: List of feature column names
        """
        if df is None:
            if not self.train_data:
                print("No training data loaded. Call load_data() first.")
                return []
            df = list(self.train_data.values())[0]
        
        # Get feature columns excluding time and attack columns
        feature_cols = [col for col in df.columns 
                       if not col.startswith('time') and not col.startswith('attack')]
        
        return feature_cols


if __name__ == "__main__":
    # Example usage
    processor = HAIDataProcessor()
    processor.load_data()
    
    # Get first training and test set
    train_df = list(processor.train_data.values())[0]
    test_df = list(processor.test_data.values())[0]
    
    print("\nTraining data sample:")
    print(train_df.head())
    
    print("\nTest data sample:")
    print(test_df.head())