import os
import pandas as pd
import glob
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

def detect_delimiter(file_path):
    """Detect the delimiter used in a CSV file"""
    with open(file_path, 'r') as file:
        first_line = file.readline()
        if ';' in first_line:
            return ';'
        elif ',' in first_line:
            return ','
        else:
            return None

def detect_time_column(df):
    """Detect the name of the time column"""
    possible_names = ['time', 'timestamp', 'Timestamp']
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def convert_csv_to_parquet(csv_path, parquet_path=None, time_column=None):
    """
    Convert a CSV file to Parquet format
    
    Parameters:
    csv_path: Path to the CSV file
    parquet_path: Path for the output Parquet file, if None, uses the same path as CSV but with .parquet extension
    time_column: Name of the time column, if None, tries to detect automatically
    
    Returns:
    parquet_path: Path to the generated Parquet file
    """
    # Detect delimiter
    delimiter = detect_delimiter(csv_path)
    if delimiter is None:
        raise ValueError(f"Could not detect delimiter: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path, delimiter=delimiter)
    
    # Detect time column
    if time_column is None:
        time_column = detect_time_column(df)
        if time_column is None:
            print(f"Warning: Could not detect time column: {csv_path}")
        else:
            # Convert time column to datetime format
            try:
                df[time_column] = pd.to_datetime(df[time_column])
            except:
                print(f"Warning: Could not convert time column to datetime format: {csv_path}")
    
    # Set Parquet output path
    if parquet_path is None:
        parquet_path = os.path.splitext(csv_path)[0] + '.parquet'
    
    # Write DataFrame to Parquet file
    df.to_parquet(parquet_path, engine='pyarrow', index=False)
    
    return parquet_path

def convert_directory(directory_path, recursive=True):
    """
    Convert all CSV files in a directory to Parquet format
    
    Parameters:
    directory_path: Path to the directory
    recursive: Whether to process subdirectories recursively
    
    Returns:
    converted_files: List of converted files
    """
    # Get all CSV files
    if recursive:
        csv_files = glob.glob(os.path.join(directory_path, '**', '*.csv'), recursive=True)
    else:
        csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    
    converted_files = []
    for csv_file in csv_files:
        try:
            parquet_file = convert_csv_to_parquet(csv_file)
            converted_files.append((csv_file, parquet_file))
            print(f"Converted: {csv_file} -> {parquet_file}")
        except Exception as e:
            print(f"Conversion failed: {csv_file}, Error: {str(e)}")
    
    return converted_files

def get_dataset_info(directory_path):
    """
    Get basic information about a dataset
    
    Parameters:
    directory_path: Path to the dataset directory
    
    Returns:
    info: Dictionary containing dataset information
    """
    # Get all Parquet files
    parquet_files = glob.glob(os.path.join(directory_path, '**', '*.parquet'), recursive=True)
    
    info = {
        'dataset_name': os.path.basename(directory_path),
        'num_files': len(parquet_files),
        'files': []
    }
    
    total_rows = 0
    total_columns = 0
    
    for parquet_file in parquet_files:
        try:
            # Read Parquet file metadata
            parquet_metadata = pq.read_metadata(parquet_file)
            num_rows = parquet_metadata.num_rows
            
            # Read Parquet file schema
            table = pq.read_table(parquet_file)
            num_columns = len(table.schema.names)
            
            # Update statistics
            total_rows += num_rows
            total_columns += num_columns
            
            file_info = {
                'file_name': os.path.basename(parquet_file),
                'num_rows': num_rows,
                'num_columns': num_columns
            }
            
            info['files'].append(file_info)
        except Exception as e:
            print(f"Failed to get file info: {parquet_file}, Error: {str(e)}")
    
    info['total_rows'] = total_rows
    info['avg_columns'] = total_columns / len(parquet_files) if parquet_files else 0
    
    return info

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python csv_to_parquet.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    converted_files = convert_directory(directory_path)
    
    print(f"\nConverted {len(converted_files)} files")
