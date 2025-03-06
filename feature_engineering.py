import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore')

def load_parquet_data(file_path):
    """
    Load data from a Parquet file
    
    Parameters:
    file_path: Path to the Parquet file
    
    Returns:
    df: Pandas DataFrame containing the data
    """
    return pd.read_parquet(file_path)

def add_time_features(df, time_col='time'):
    """
    Add time-based features to the DataFrame
    
    Parameters:
    df: DataFrame containing the data
    time_col: Name of the time column
    
    Returns:
    df: DataFrame with added time features
    """
    if time_col in df.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Extract time components
        df['hour'] = df[time_col].dt.hour
        df['minute'] = df[time_col].dt.minute
        df['second'] = df[time_col].dt.second
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Create cyclical features for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    
    return df

def add_statistical_features(df, window_sizes=[5, 10, 20], exclude_cols=None):
    """
    Add statistical features based on rolling windows
    
    Parameters:
    df: DataFrame containing the data
    window_sizes: List of window sizes for rolling calculations
    exclude_cols: List of columns to exclude from calculations
    
    Returns:
    df: DataFrame with added statistical features
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out excluded columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Add rolling statistics for each window size
    for window in window_sizes:
        for col in feature_cols:
            # Rolling mean
            result_df[f'{col}_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            
            # Rolling standard deviation
            result_df[f'{col}_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
            
            # Rolling min and max
            result_df[f'{col}_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
            result_df[f'{col}_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
            
            # Rolling range (max - min)
            result_df[f'{col}_range_{window}'] = result_df[f'{col}_max_{window}'] - result_df[f'{col}_min_{window}']
    
    return result_df

def add_lag_features(df, lag_values=[1, 3, 5], exclude_cols=None):
    """
    Add lagged features
    
    Parameters:
    df: DataFrame containing the data
    lag_values: List of lag values
    exclude_cols: List of columns to exclude from calculations
    
    Returns:
    df: DataFrame with added lag features
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out excluded columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Add lagged features
    for lag in lag_values:
        for col in feature_cols:
            result_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Fill NaN values created by lag
    result_df = result_df.fillna(method='bfill')
    
    return result_df

def add_diff_features(df, diff_values=[1, 3], exclude_cols=None):
    """
    Add difference features
    
    Parameters:
    df: DataFrame containing the data
    diff_values: List of difference periods
    exclude_cols: List of columns to exclude from calculations
    
    Returns:
    df: DataFrame with added difference features
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out excluded columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Add difference features
    for diff in diff_values:
        for col in feature_cols:
            result_df[f'{col}_diff_{diff}'] = df[col].diff(diff)
    
    # Fill NaN values created by diff
    result_df = result_df.fillna(method='bfill')
    
    return result_df

def add_interaction_features(df, n_interactions=10, exclude_cols=None):
    """
    Add interaction features between top correlated columns
    
    Parameters:
    df: DataFrame containing the data
    n_interactions: Number of interaction features to add
    exclude_cols: List of columns to exclude from calculations
    
    Returns:
    df: DataFrame with added interaction features
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out excluded columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Calculate correlation matrix
    corr = df[feature_cols].corr().abs()
    
    # Get pairs of columns with highest correlation
    pairs = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            col1 = feature_cols[i]
            col2 = feature_cols[j]
            correlation = corr.loc[col1, col2]
            pairs.append((col1, col2, correlation))
    
    # Sort pairs by correlation (descending)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Add interaction features for top pairs
    for i, (col1, col2, _) in enumerate(pairs[:n_interactions]):
        # Multiplication
        result_df[f'{col1}_mul_{col2}'] = df[col1] * df[col2]
        
        # Division (with handling for division by zero)
        result_df[f'{col1}_div_{col2}'] = df[col1] / df[col2].replace(0, np.nan)
        result_df[f'{col1}_div_{col2}'] = result_df[f'{col1}_div_{col2}'].fillna(0)
    
    return result_df

def add_pca_features(df, n_components=5, exclude_cols=None):
    """
    Add PCA features
    
    Parameters:
    df: DataFrame containing the data
    n_components: Number of PCA components to add
    exclude_cols: List of columns to exclude from calculations
    
    Returns:
    df: DataFrame with added PCA features
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out excluded columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, len(feature_cols)))
    pca_features = pca.fit_transform(df[feature_cols])
    
    # Add PCA features to DataFrame
    for i in range(pca_features.shape[1]):
        result_df[f'pca_{i+1}'] = pca_features[:, i]
    
    return result_df

def select_features_importance(df, target_col='attack', n_features=50, exclude_cols=None):
    """
    Select features based on importance from a Random Forest model
    
    Parameters:
    df: DataFrame containing the data
    target_col: Name of the target column
    n_features: Number of features to select
    exclude_cols: List of columns to exclude from selection
    
    Returns:
    selected_features: List of selected feature names
    importance_df: DataFrame with feature importance scores
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out excluded columns and target column
    feature_cols = [col for col in numeric_cols if col not in exclude_cols and col != target_col]
    
    # Train a Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df[feature_cols], df[target_col])
    
    # Get feature importance
    importance = rf.feature_importances_
    
    # Create DataFrame with feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Select top features
    selected_features = importance_df['feature'].tolist()[:n_features]
    
    return selected_features, importance_df

def select_features_correlation(df, target_col='attack', n_features=50, exclude_cols=None):
    """
    Select features based on correlation with target
    
    Parameters:
    df: DataFrame containing the data
    target_col: Name of the target column
    n_features: Number of features to select
    exclude_cols: List of columns to exclude from selection
    
    Returns:
    selected_features: List of selected feature names
    correlation_df: DataFrame with correlation scores
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out excluded columns and target column
    feature_cols = [col for col in numeric_cols if col not in exclude_cols and col != target_col]
    
    # Calculate correlation with target
    correlation = []
    for col in feature_cols:
        corr = df[col].corr(df[target_col])
        correlation.append((col, abs(corr)))
    
    # Create DataFrame with correlation
    correlation_df = pd.DataFrame(correlation, columns=['feature', 'correlation'])
    
    # Sort by correlation
    correlation_df = correlation_df.sort_values('correlation', ascending=False)
    
    # Select top features
    selected_features = correlation_df['feature'].tolist()[:n_features]
    
    return selected_features, correlation_df

def select_features_anova(df, target_col='attack', n_features=50, exclude_cols=None):
    """
    Select features based on ANOVA F-value
    
    Parameters:
    df: DataFrame containing the data
    target_col: Name of the target column
    n_features: Number of features to select
    exclude_cols: List of columns to exclude from selection
    
    Returns:
    selected_features: List of selected feature names
    scores_df: DataFrame with ANOVA F-value scores
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out excluded columns and target column
    feature_cols = [col for col in numeric_cols if col not in exclude_cols and col != target_col]
    
    # Apply SelectKBest with f_classif (ANOVA F-value)
    selector = SelectKBest(f_classif, k=min(n_features, len(feature_cols)))
    selector.fit(df[feature_cols], df[target_col])
    
    # Get scores
    scores = selector.scores_
    
    # Create DataFrame with scores
    scores_df = pd.DataFrame({
        'feature': feature_cols,
        'score': scores
    })
    
    # Sort by score
    scores_df = scores_df.sort_values('score', ascending=False)
    
    # Select top features
    selected_features = scores_df['feature'].tolist()[:n_features]
    
    return selected_features, scores_df

def plot_feature_importance(importance_df, n_features=20, figsize=(12, 10)):
    """
    Plot feature importance
    
    Parameters:
    importance_df: DataFrame with feature importance
    n_features: Number of features to display
    figsize: Figure size
    """
    # Select top features
    top_features = importance_df.head(n_features)
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'Top {n_features} Features by Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    return plt.gcf()

def apply_feature_engineering(df, time_col=None, target_col=None, exclude_cols=None, 
                             add_time=True, add_stats=True, add_lag=True, add_diff=True, 
                             add_interaction=True, add_pca=True, window_sizes=[5, 10, 20], 
                             lag_values=[1, 3, 5], diff_values=[1, 3], n_interactions=10, 
                             n_pca_components=5):
    """
    Apply feature engineering pipeline
    
    Parameters:
    df: DataFrame containing the data
    time_col: Name of the time column
    target_col: Name of the target column
    exclude_cols: List of columns to exclude from feature engineering
    add_time: Whether to add time features
    add_stats: Whether to add statistical features
    add_lag: Whether to add lag features
    add_diff: Whether to add difference features
    add_interaction: Whether to add interaction features
    add_pca: Whether to add PCA features
    window_sizes: List of window sizes for rolling calculations
    lag_values: List of lag values
    diff_values: List of difference periods
    n_interactions: Number of interaction features to add
    n_pca_components: Number of PCA components to add
    
    Returns:
    result_df: DataFrame with added features
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Add time column to exclude_cols if it exists
    if time_col is not None and time_col not in exclude_cols:
        exclude_cols.append(time_col)
    
    # Add target column to exclude_cols if it exists
    if target_col is not None and target_col not in exclude_cols:
        exclude_cols.append(target_col)
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Apply feature engineering steps
    if add_time and time_col is not None:
        result_df = add_time_features(result_df, time_col=time_col)
    
    if add_stats:
        result_df = add_statistical_features(result_df, window_sizes=window_sizes, exclude_cols=exclude_cols)
    
    if add_lag:
        result_df = add_lag_features(result_df, lag_values=lag_values, exclude_cols=exclude_cols)
    
    if add_diff:
        result_df = add_diff_features(result_df, diff_values=diff_values, exclude_cols=exclude_cols)
    
    if add_interaction:
        result_df = add_interaction_features(result_df, n_interactions=n_interactions, exclude_cols=exclude_cols)
    
    if add_pca:
        result_df = add_pca_features(result_df, n_components=n_pca_components, exclude_cols=exclude_cols)
    
    return result_df

def select_final_features(df, target_col='attack', n_features=50, method='importance', exclude_cols=None):
    """
    Select final features using specified method
    
    Parameters:
    df: DataFrame containing the data
    target_col: Name of the target column
    n_features: Number of features to select
    method: Feature selection method ('importance', 'correlation', or 'anova')
    exclude_cols: List of columns to exclude from selection
    
    Returns:
    selected_df: DataFrame with selected features
    importance_df: DataFrame with feature importance/scores
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Select features based on specified method
    if method == 'importance':
        selected_features, importance_df = select_features_importance(df, target_col=target_col, n_features=n_features, exclude_cols=exclude_cols)
    elif method == 'correlation':
        selected_features, importance_df = select_features_correlation(df, target_col=target_col, n_features=n_features, exclude_cols=exclude_cols)
    elif method == 'anova':
        selected_features, importance_df = select_features_anova(df, target_col=target_col, n_features=n_features, exclude_cols=exclude_cols)
    else:
        raise ValueError(f"Invalid method: {method}. Must be 'importance', 'correlation', or 'anova'")
    
    # Always include target column
    if target_col not in selected_features and target_col in df.columns:
        selected_features.append(target_col)
    
    # Create DataFrame with selected features
    selected_df = df[selected_features]
    
    return selected_df, importance_df

def process_dataset(input_path, output_path, time_col=None, target_col='attack', 
                   feature_selection_method='importance', n_features=50, 
                   add_time=True, add_stats=True, add_lag=True, add_diff=True, 
                   add_interaction=True, add_pca=True):
    """
    Process a dataset with feature engineering and selection
    
    Parameters:
    input_path: Path to the input Parquet file
    output_path: Path for the output Parquet file
    time_col: Name of the time column
    target_col: Name of the target column
    feature_selection_method: Method for feature selection
    n_features: Number of features to select
    add_time: Whether to add time features
    add_stats: Whether to add statistical features
    add_lag: Whether to add lag features
    add_diff: Whether to add difference features
    add_interaction: Whether to add interaction features
    add_pca: Whether to add PCA features
    
    Returns:
    importance_df: DataFrame with feature importance/scores
    """
    # Load data
    df = load_parquet_data(input_path)
    
    # Apply feature engineering
    engineered_df = apply_feature_engineering(
        df, 
        time_col=time_col, 
        target_col=target_col, 
        add_time=add_time, 
        add_stats=add_stats, 
        add_lag=add_lag, 
        add_diff=add_diff, 
        add_interaction=add_interaction, 
        add_pca=add_pca
    )
    
    # Select final features
    if target_col in engineered_df.columns:
        selected_df, importance_df = select_final_features(
            engineered_df, 
            target_col=target_col, 
            n_features=n_features, 
            method=feature_selection_method
        )
    else:
        # If target column doesn't exist (e.g., for test data without labels),
        # use all features
        selected_df = engineered_df
        importance_df = pd.DataFrame()
    
    # Save to Parquet
    selected_df.to_parquet(output_path, engine='pyarrow', index=False)
    
    return importance_df

def process_dataset_directory(input_dir, output_dir, dataset_name, time_col=None, target_col='attack', 
                             feature_selection_method='importance', n_features=50, 
                             add_time=True, add_stats=True, add_lag=True, add_diff=True, 
                             add_interaction=True, add_pca=True):
    """
    Process all Parquet files in a dataset directory
    
    Parameters:
    input_dir: Base directory containing all datasets
    output_dir: Base directory for output
    dataset_name: Name of the specific dataset (e.g., 'hai-21.03')
    time_col: Name of the time column
    target_col: Name of the target column
    feature_selection_method: Method for feature selection
    n_features: Number of features to select
    add_time: Whether to add time features
    add_stats: Whether to add statistical features
    add_lag: Whether to add lag features
    add_diff: Whether to add difference features
    add_interaction: Whether to add interaction features
    add_pca: Whether to add PCA features
    
    Returns:
    results: Dictionary containing results for each file
    """
    # Create input and output paths
    input_path = os.path.join(input_dir, dataset_name)
    output_path = os.path.join(output_dir, f"{dataset_name}_engineered")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    results = {}
    
    # Process each Parquet file
    for file in os.listdir(input_path):
        if file.endswith('.parquet'):
            file_name = os.path.splitext(file)[0]
            input_file_path = os.path.join(input_path, file)
            output_file_path = os.path.join(output_path, file)
            
            print(f"Processing {file_name}...")
            
            # Process file
            importance_df = process_dataset(
                input_file_path, 
                output_file_path, 
                time_col=time_col, 
                target_col=target_col, 
                feature_selection_method=feature_selection_method, 
                n_features=n_features, 
                add_time=add_time, 
                add_stats=add_stats, 
                add_lag=add_lag, 
                add_diff=add_diff, 
                add_interaction=add_interaction, 
                add_pca=add_pca
            )
            
            results[file_name] = importance_df
            
            print(f"Saved to {output_file_path}")
    
    return results

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python feature_engineering.py <input_dir> <output_dir> <dataset_name>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    dataset_name = sys.argv[3]
    
    results = process_dataset_directory(input_dir, output_dir, dataset_name)
    
    print(f"\nProcessed dataset: {dataset_name}")
