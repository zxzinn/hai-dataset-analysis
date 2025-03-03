import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

class TimeSeriesDataset(Dataset):
    """
    Custom dataset for time series data with efficient loading and preprocessing
    """
    def __init__(self, file_paths, seq_length=100, chunk_size=100000, transform=None):
        self.file_paths = file_paths
        self.seq_length = seq_length
        self.chunk_size = chunk_size
        self.transform = transform
        self.scaler = None
        
        # Get total number of sequences (approximate)
        self.total_rows = 0
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                self.total_rows += sum(1 for _ in f) - 1  # subtract header
        
        self.total_sequences = max(0, self.total_rows - len(file_paths) * seq_length)
        
        # Process data in chunks and store indices
        self.data_chunks = []
        self._process_data()
    
    def _process_data(self):
        """Process data in chunks to avoid memory issues"""
        # First pass: fit scaler
        self.scaler = StandardScaler()
        for file_path in self.file_paths:
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                features = chunk.drop('timestamp', axis=1).values
                self.scaler.partial_fit(features)
        
        # Second pass: transform and create sequences
        for file_path in self.file_paths:
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                features = chunk.drop('timestamp', axis=1).values
                features_scaled = self.scaler.transform(features)
                
                # Create sequences only if chunk is large enough
                if len(features_scaled) > self.seq_length:
                    sequences = create_sequences(features_scaled, self.seq_length)
                    self.data_chunks.append(torch.tensor(sequences, dtype=torch.float32))
    
    def __len__(self):
        return sum(len(chunk) for chunk in self.data_chunks)
    
    def __getitem__(self, idx):
        # Find which chunk contains this index
        chunk_idx = 0
        while idx >= len(self.data_chunks[chunk_idx]):
            idx -= len(self.data_chunks[chunk_idx])
            chunk_idx += 1
        
        # Get sequence from the appropriate chunk
        sequence = self.data_chunks[chunk_idx][idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        # For unsupervised learning, input = target
        return sequence, sequence

def create_sequences(data, seq_length):
    """Create sliding window sequences from time series data"""
    xs = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        xs.append(x)
    return np.array(xs)

def load_test_data_and_labels(test_file, label_file, seq_length=100, scaler=None):
    """Load test data and align with labels"""
    # Load test data
    test_data = pd.read_csv(test_file)
    timestamps = test_data['timestamp'].values
    features = test_data.drop('timestamp', axis=1).values
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)
    
    # Create sequences
    sequences = create_sequences(features_scaled, seq_length)
    
    # Load labels
    labels = pd.read_csv(label_file)
    
    # Align labels with sequences
    aligned_labels = []
    for i in range(seq_length, len(timestamps)):
        # Get timestamp for this sequence
        ts = timestamps[i]
        # Find corresponding label
        label = labels[labels['timestamp'] == ts]['label'].values
        if len(label) > 0:
            aligned_labels.append(label[0])
        else:
            aligned_labels.append(0)  # Default to normal if no label found
    
    return sequences, np.array(aligned_labels), timestamps[seq_length:], scaler

def get_dataset_info(file_path):
    """Get information about a dataset file"""
    # Get file size
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # in MB
    
    # Read first few rows to get column info
    df_sample = pd.read_csv(file_path, nrows=5)
    num_columns = len(df_sample.columns)
    
    # Count rows efficiently without loading entire file
    with open(file_path, 'r') as f:
        num_rows = sum(1 for _ in f) - 1  # subtract header
    
    return {
        'file_path': file_path,
        'file_size_mb': file_size,
        'num_columns': num_columns,
        'num_rows': num_rows
    }

def load_data_in_chunks(file_path, chunk_size=100000):
    """Load data in chunks to avoid memory issues"""
    chunks = []
    for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size), 
                      desc=f"Loading {os.path.basename(file_path)}"):
        chunks.append(chunk)
    return pd.concat(chunks)