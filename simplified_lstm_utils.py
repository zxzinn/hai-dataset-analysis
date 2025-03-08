#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified LSTM Utilities for HAI-21.03 Dataset Analysis

This module provides utility functions for analyzing the HAI-21.03 dataset
using a simplified LSTM approach with a larger sliding window.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import itertools
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')
sns.set(style="darkgrid")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_graph_from_columns(df, correlation_threshold=0.7):
    """
    Create a graph structure based on column names and their correlations.
    
    Args:
        df (pd.DataFrame): DataFrame with sensor/actuator columns
        correlation_threshold (float): Threshold for adding edges between components in the graph
        
    Returns:
        nx.Graph: NetworkX graph representing the system
    """
    G = nx.Graph()
    
    # Extract columns excluding time and attack columns
    cols = [col for col in df.columns if not col.startswith('time') and not col.startswith('attack')]
    
    # Add nodes for each subsystem
    subsystems = ['P1', 'P2', 'P3', 'P4']
    for subsystem in subsystems:
        G.add_node(subsystem, type='subsystem')
    
    # Add nodes for each sensor/actuator
    for col in cols:
        parts = col.split('_')
        if len(parts) >= 2:
            subsystem = parts[0]
            component = '_'.join(parts[1:])
            
            # Add node for component
            G.add_node(col, type='component', subsystem=subsystem)
            
            # Add edge between component and its subsystem
            G.add_edge(subsystem, col, weight=1.0)
    
    # Add edges between components based on correlation
    corr_matrix = df[cols].corr().abs()
    
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i < j and corr_matrix.loc[col1, col2] > correlation_threshold:
                G.add_edge(col1, col2, weight=corr_matrix.loc[col1, col2])
    
    return G

def visualize_graph(G, max_nodes=50):
    """
    Visualize a graph, limiting to max_nodes if the graph is too large.
    
    Args:
        G (nx.Graph): NetworkX graph
        max_nodes (int): Maximum number of nodes to display
    """
    if G.number_of_nodes() > max_nodes:
        # Get the most connected nodes
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G_sub = G.subgraph(top_nodes)
        print(f"Showing subgraph with {G_sub.number_of_nodes()} nodes and {G_sub.number_of_edges()} edges")
        G_vis = G_sub
    else:
        G_vis = G
    
    plt.figure(figsize=(12, 10))
    
    # Set node colors based on type (safely)
    node_colors = []
    for node in G_vis.nodes():
        node_type = G_vis.nodes[node].get('type', 'unknown')
        if node_type == 'subsystem':
            node_colors.append('red')
        elif node_type == 'component':
            node_colors.append('blue')
        else:
            node_colors.append('gray')
    
    # Set edge widths based on weight
    edge_widths = [G_vis[u][v].get('weight', 1.0) * 2 for u, v in G_vis.edges()]
    
    # Draw the graph
    pos = nx.spring_layout(G_vis, seed=42)
    nx.draw_networkx_nodes(G_vis, pos, node_color=node_colors, node_size=300, alpha=0.8)
    nx.draw_networkx_edges(G_vis, pos, width=edge_widths, alpha=0.5)
    nx.draw_networkx_labels(G_vis, pos, font_size=8)
    
    plt.title("HAI Dataset Component Relationship Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def simplified_graph_based_feature_engineering(df, G):
    """
    Create simplified graph-based features.
    
    Args:
        df (pd.DataFrame): DataFrame with sensor/actuator data
        G (nx.Graph): NetworkX graph
        
    Returns:
        pd.DataFrame: DataFrame with additional graph-based features
    """
    df_processed = df.copy()
    
    # Extract columns excluding time and attack columns
    cols = [col for col in df.columns if not col.startswith('time') and not col.startswith('attack')]
    
    # Add subsystem-level features (simplified)
    for subsystem in ['P1', 'P2', 'P3', 'P4']:
        subsystem_cols = [col for col in cols if col.startswith(f"{subsystem}_")]
        if subsystem_cols:
            # Calculate subsystem-level statistics (reduced set)
            df_processed[f"{subsystem}_graph_mean"] = df[subsystem_cols].mean(axis=1)
            df_processed[f"{subsystem}_graph_std"] = df[subsystem_cols].std(axis=1)
            df_processed[f"{subsystem}_graph_max"] = df[subsystem_cols].max(axis=1)
            df_processed[f"{subsystem}_graph_min"] = df[subsystem_cols].min(axis=1)
    
    # Add cross-subsystem features (simplified)
    subsystems = ['P1', 'P2', 'P3', 'P4']
    for i, s1 in enumerate(subsystems):
        for j, s2 in enumerate(subsystems):
            if i < j:
                # Calculate ratio and difference between subsystem means
                if f"{s1}_graph_mean" in df_processed.columns and f"{s2}_graph_mean" in df_processed.columns:
                    s2_mean = df_processed[f"{s2}_graph_mean"].replace(0, 1e-10)  # Avoid division by zero
                    df_processed[f"{s1}_{s2}_ratio"] = df_processed[f"{s1}_graph_mean"] / s2_mean
                    df_processed[f"{s1}_{s2}_diff"] = df_processed[f"{s1}_graph_mean"] - df_processed[f"{s2}_graph_mean"]
    
    # Add rolling window features (simplified)
    window_sizes = [10]  # Reduced to just one window size
    for window in window_sizes:
        for subsystem in subsystems:
            col = f"{subsystem}_graph_mean"
            if col in df_processed.columns:
                df_processed[f"{col}_rolling_{window}_mean"] = df_processed[col].rolling(window=window).mean()
                df_processed[f"{col}_rolling_{window}_std"] = df_processed[col].rolling(window=window).std()
    
    # Fill NaN values
    df_processed = df_processed.fillna(method='bfill').fillna(method='ffill')
    
    return df_processed

def select_features(train_df, test_df, n_components=20, n_features=50):
    """
    Select features and reduce dimensions.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        n_components (int): Number of PCA components
        n_features (int): Number of top features to select
        
    Returns:
        tuple: (X_train_pca, X_test_pca, y_test, scaler, pca, top_features)
    """
    # Exclude non-numeric and target columns
    exclude_cols = ['time', 'attack', 'attack_P1', 'attack_P2', 'attack_P3']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols and 
                    pd.api.types.is_numeric_dtype(train_df[col])]
    
    # Extract labels if available
    y_test = test_df['attack'].values if 'attack' in test_df.columns else None
    
    # Calculate feature importance (using variance)
    variances = train_df[feature_cols].var()
    feature_importance = {feature: score for feature, score in zip(feature_cols, variances)}
    
    # Select top features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:min(n_features, len(sorted_features))]]
    
    # Extract features
    X_train = train_df[top_features].values
    X_test = test_df[top_features].values
    
    # Scale data using RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    print(f"Selected {len(top_features)} features out of {len(feature_cols)}")
    
    return X_train_pca, X_test_pca, y_test, scaler, pca, top_features

def create_sequences(data, seq_length=192, stride=8):
    """
    Create sequences for time series models with larger window.
    
    Args:
        data (np.array): Input data
        seq_length (int): Sequence length (default 192)
        stride (int): Stride for sliding window
        
    Returns:
        np.array: Sequences
    """
    sequences = []
    for i in range(0, len(data) - seq_length + 1, stride):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    
    return np.array(sequences)

def build_simplified_lstm_autoencoder(seq_length, n_features, learning_rate=0.001):
    """
    Build an LSTM autoencoder with attention mechanism.

    Args:
        seq_length (int): Sequence length
        n_features (int): Number of features
        learning_rate (float): Learning rate for optimizer

    Returns:
        Model: Keras model
    """
    # Input layer
    inputs = Input(shape=(seq_length, n_features), name='input_layer')

    # Encoder with attention
    lstm_encoder = LSTM(16, activation='tanh', return_sequences=True)(inputs)

    # Self-attention mechanism
    attention = Dense(1, activation='tanh')(lstm_encoder)  # 生成注意力權重
    attention = tf.keras.layers.Reshape((seq_length,))(attention)
    attention_weights = tf.keras.layers.Activation('softmax')(attention)  # 注意力權重轉為概率分布

    # 應用注意力權重到LSTM輸出
    attention_weights = tf.keras.layers.Reshape((seq_length, 1))(attention_weights)
    attended_lstm = tf.keras.layers.Multiply()([lstm_encoder, attention_weights])

    # 聚合加權特徵
    context_vector = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended_lstm)

    # 規範化和Dropout
    x = LayerNormalization()(context_vector)
    x = Dropout(0.2)(x)

    # Bottleneck
    encoded = Dense(8, activation='relu')(x)

    # Decoder - 與原模型類似
    x = Dense(16, activation='relu')(encoded)
    x = tf.keras.layers.RepeatVector(seq_length)(x)
    
    x = LSTM(16, activation='tanh', return_sequences=True)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = tf.keras.layers.TimeDistributed(Dense(n_features))(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    return model

def train_model(model, X_train_seq, epochs=10, batch_size=32, validation_split=0.2, patience=5, model_path=None):
    """
    Train the LSTM autoencoder model.
    
    Args:
        model (Model): Keras model
        X_train_seq (np.array): Training sequences
        epochs (int): Maximum number of epochs
        batch_size (int): Batch size
        validation_split (float): Validation split ratio
        patience (int): Patience for early stopping
        model_path (str, optional): Path to save the best model
        
    Returns:
        History: Training history
    """
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)
    ]
    
    if model_path:
        callbacks.append(ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True))
    
    # Train model
    history = model.fit(
        X_train_seq, X_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def determine_threshold(model, X_train_seq, method='dynamic', contamination=0.01, percentile=99):
    """
    Determine threshold for anomaly detection.
    
    Args:
        model (Model): Trained model
        X_train_seq (np.array): Training sequences
        method (str): Method for threshold determination ('dynamic', 'percentile', or 'iqr')
        contamination (float): Expected proportion of anomalies (for dynamic method)
        percentile (float): Percentile for threshold (for percentile method)
        
    Returns:
        float: Threshold value
    """
    # Get predictions from model
    X_train_pred = model.predict(X_train_seq)
    train_mse = np.mean(np.square(X_train_seq - X_train_pred), axis=(1, 2))
    
    if method == 'dynamic':
        # Simulate anomalies using highest reconstruction errors
        simulated_anomalies = np.zeros(len(train_mse))
        anomaly_count = int(contamination * len(train_mse))
        anomaly_indices = np.argsort(train_mse)[-anomaly_count:]
        simulated_anomalies[anomaly_indices] = 1
        
        # Calculate precision-recall curve
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, thresholds = precision_recall_curve(simulated_anomalies, train_mse)
        
        # Find threshold that maximizes F1 score
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
        
    elif method == 'percentile':
        # Use percentile method
        threshold = np.percentile(train_mse, percentile)
        
    elif method == 'iqr':
        # Use IQR method
        q1 = np.percentile(train_mse, 25)
        q3 = np.percentile(train_mse, 75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
        
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    return threshold

def detect_anomalies(model, X_test_seq, seq_length, stride, data_length, threshold, anomaly_score_threshold=0.65, min_anomaly_length=30, gap_threshold=3):
    """
    Detect anomalies using the trained model.
    
    Args:
        model (Model): Trained model
        X_test_seq (np.array): Test sequences
        seq_length (int): Sequence length
        stride (int): Stride used for sequences
        data_length (int): Original data length
        threshold (float): Anomaly detection threshold
        anomaly_score_threshold (float): Anomaly score threshold
        min_anomaly_length (int): Minimum length of anomalies to keep
        gap_threshold (int): Maximum gap between anomalies to merge
        
    Returns:
        tuple: (anomaly_scores, anomaly_labels)
    """
    # Get predictions from model
    X_test_pred = model.predict(X_test_seq)
    mse = np.mean(np.square(X_test_seq - X_test_pred), axis=(1, 2))
    
    # Initialize anomaly scores array
    anomaly_scores = np.zeros(data_length)
    count = np.zeros(data_length)
    
    # For each sequence, if it's anomalous, increment the score for all points in the sequence
    for i, error in enumerate(mse):
        idx = i * stride
        if idx + seq_length <= data_length:
            if error > threshold:
                anomaly_scores[idx:idx+seq_length] += 1
            count[idx:idx+seq_length] += 1
    
    # Normalize scores by count
    anomaly_scores = np.divide(anomaly_scores, count, out=np.zeros_like(anomaly_scores), where=count!=0)
    
    # Apply threshold to get binary labels
    anomaly_labels = (anomaly_scores > anomaly_score_threshold).astype(int)
    
    # Apply post-processing
    anomaly_labels = post_process_anomalies(anomaly_labels, min_anomaly_length, gap_threshold)
    
    return anomaly_scores, anomaly_labels

def post_process_anomalies(anomaly_labels, min_anomaly_length=30, gap_threshold=3):
    """
    Apply post-processing to reduce false positives and false negatives.
    
    Args:
        anomaly_labels (np.array): Binary anomaly labels
        min_anomaly_length (int): Minimum length of anomalies to keep
        gap_threshold (int): Maximum gap between anomalies to merge
        
    Returns:
        np.array: Processed binary anomaly labels
    """
    # Make a copy to avoid modifying the original
    processed_labels = anomaly_labels.copy()
    
    # Remove short anomalies (likely false positives)
    i = 0
    while i < len(processed_labels):
        if processed_labels[i] == 1:
            # Find the end of this anomaly
            j = i
            while j < len(processed_labels) and processed_labels[j] == 1:
                j += 1
            
            # If anomaly is too short, remove it
            if j - i < min_anomaly_length:
                processed_labels[i:j] = 0
            
            i = j
        else:
            i += 1
    
    # Merge nearby anomalies
    i = 0
    while i < len(processed_labels):
        if processed_labels[i] == 1:
            # Find the end of this anomaly
            j = i
            while j < len(processed_labels) and processed_labels[j] == 1:
                j += 1
            
            # Look for another anomaly nearby
            if j < len(processed_labels) - gap_threshold:
                next_start = j
                while next_start < j + gap_threshold and next_start < len(processed_labels) and processed_labels[next_start] == 0:
                    next_start += 1
                
                if next_start < j + gap_threshold and next_start < len(processed_labels) and processed_labels[next_start] == 1:
                    processed_labels[j:next_start] = 1
            
            i = j
        else:
            i += 1
    
    return processed_labels

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance.
    
    Args:
        y_true (np.array): Ground truth labels
        y_pred (np.array): Predicted labels
        
    Returns:
        dict: Evaluation metrics
    """
    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Store results
    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'confusion_matrix': cm,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }
    
    # Print metrics
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")
    print(f"False Negative Rate: {false_negative_rate:.4f}")
    
    return results

def visualize_results(y_test, anomaly_scores, anomaly_labels, test_name, anomaly_score_threshold=0.65):
    """
    Visualize anomaly detection results.
    
    Args:
        y_test (np.array): Ground truth labels
        anomaly_scores (np.array): Anomaly scores
        anomaly_labels (np.array): Predicted anomaly labels
        test_name (str): Name of the test file
        anomaly_score_threshold (float): Threshold used for anomaly scores
    """
    plt.figure(figsize=(20, 10))
    
    # Plot ground truth
    plt.subplot(3, 1, 1)
    plt.plot(y_test, 'b-', label='Ground Truth')
    plt.title(f'Ground Truth - {test_name}')
    plt.ylabel('Anomaly')
    plt.yticks([0, 1])
    plt.grid(True)
    plt.legend()
    
    # Plot anomaly scores
    plt.subplot(3, 1, 2)
    plt.plot(anomaly_scores, 'r-', label='Anomaly Scores')
    plt.axhline(y=anomaly_score_threshold, color='g', linestyle='--', label=f'Score Threshold ({anomaly_score_threshold})')
    plt.title(f'Anomaly Scores - {test_name}')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    
    # Plot predictions
    plt.subplot(3, 1, 3)
    plt.plot(anomaly_labels, 'g-', label='Predictions')
    plt.title(f'Predictions - {test_name}')
    plt.xlabel('Time')
    plt.ylabel('Anomaly')
    plt.yticks([0, 1])
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, anomaly_labels)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {test_name}')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Normal', 'Anomaly'], rotation=45)
    plt.yticks(tick_marks, ['Normal', 'Anomaly'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def visualize_metrics_comparison(results_df, metrics=['Precision', 'Recall', 'F1 Score']):
    """
    Visualize metrics comparison across test files.
    
    Args:
        results_df (pd.DataFrame): DataFrame with results for each test file
        metrics (list): List of metrics to visualize
    """
    plt.figure(figsize=(15, 6))

    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        plt.bar(results_df['Test File'], results_df[metric])
        plt.title(metric)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels
        for j, v in enumerate(results_df[metric]):
            plt.text(j, v + 0.02, f"{v:.4f}", ha='center')

    plt.tight_layout()
    plt.show()

def visualize_error_rates(results_df, error_metrics=['False Positive Rate', 'False Negative Rate']):
    """
    Visualize error rates comparison across test files.
    
    Args:
        results_df (pd.DataFrame): DataFrame with results for each test file
        error_metrics (list): List of error metrics to visualize
    """
    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(error_metrics):
        plt.subplot(1, 2, i+1)
        plt.bar(results_df['Test File'], results_df[metric])
        plt.title(metric)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(results_df[metric].max() * 1.2, 0.1))
        
        # Add value labels
        for j, v in enumerate(results_df[metric]):
            plt.text(j, v + 0.005, f"{v:.4f}", ha='center')

    plt.tight_layout()
    plt.show()

def compare_with_previous_models(results_df):
    """
    Compare with previous models.
    
    Args:
        results_df (pd.DataFrame): DataFrame with results for each test file
        
    Returns:
        pd.DataFrame: Comparison DataFrame
    """
    # Previous models
    previous_models = [
        ('Isolation Forest', 0.000000, 0.000000, 0.015101, 0.014308, 0.000000, 0.000000),
        ('LSTM Autoencoder', 0.075407, 0.589583, 0.119693, 0.744038, 0.122387, 0.763116),
        ('Bidirectional LSTM', 0.082332, 0.586979, 0.137261, 0.729730, 0.142645, 0.763116),
        ('Balanced CNN-LSTM', 0.030966, 0.400000, 0.010210, 0.236884, 0.010210, 0.236884),
        ('Graph-based LSTM', 0.213861, 0.841446, 0.021024, 0.620032, 0.033477, 1.000000),
        ('Improved Graph-based LSTM', 0.157116, 0.862546, 0.244444, 0.699523, 0.316239, 1.000000),
        ('Optimized Ensemble', 0.180000, 0.870000, 0.280000, 0.720000, 0.350000, 1.000000)
    ]

    # Add our model results
    for i, row in results_df.iterrows():
        model_name = f"Simplified LSTM (Window=192) - {row['Test File']}"
        previous_models.append((
            model_name,
            row['Precision'],  # Use precision as eTaP
            row['Recall'],     # Use recall as eTaR
            row['Precision'],
            row['Recall'],
            row['Precision'],  # Use precision as PA Precision
            row['Recall']      # Use recall as PA Recall
        ))

    # Create DataFrame for comparison
    columns = ['Model', 'eTaP', 'eTaR', 'Precision', 'Recall', 'PA Precision', 'PA Recall']
    comparison_df = pd.DataFrame(previous_models, columns=columns)

    # Calculate F1 scores
    comparison_df['F1 (eTaPR)'] = 2 * (comparison_df['eTaP'] * comparison_df['eTaR']) / (comparison_df['eTaP'] + comparison_df['eTaR'] + 1e-10)
    comparison_df['F1 (Standard)'] = 2 * (comparison_df['Precision'] * comparison_df['Recall']) / (comparison_df['Precision'] + comparison_df['Recall'] + 1e-10)
    comparison_df['F1 (PA)'] = 2 * (comparison_df['PA Precision'] * comparison_df['PA Recall']) / (comparison_df['PA Precision'] + comparison_df['PA Recall'] + 1e-10)

    # Replace NaN with 0
    comparison_df = comparison_df.fillna(0)
    
    return comparison_df

def visualize_model_comparison(comparison_df, metrics=['eTaP', 'eTaR', 'F1 (eTaPR)']):
    """
    Visualize model comparison.
    
    Args:
        comparison_df (pd.DataFrame): Comparison DataFrame
        metrics (list): List of metrics to visualize
    """
    plt.figure(figsize=(15, 6))

    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        plt.bar(comparison_df['Model'], comparison_df[metric])
        plt.title(metric)
        plt.xticks(rotation=90, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels for top models only
        top_indices = comparison_df[metric].nlargest(3).index
        for j in top_indices:
            v = comparison_df[metric].iloc[j]
            plt.text(j, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold', color='red')

    plt.tight_layout()
    plt.show()

def train_single_model(train_df, params):
    """
    Train a single model on the training data.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        params (dict): Parameters dictionary
        
    Returns:
        tuple: (model, G, scaler, pca, top_features, threshold)
    """
    print("Training a single model on all training data...")
    
    # Create graph
    G = create_graph_from_columns(train_df, params['correlation_threshold'])
    
    # Apply graph-based feature engineering
    train_df_processed = simplified_graph_based_feature_engineering(train_df, G)
    
    # Select features
    feature_cols = [col for col in train_df_processed.columns 
                   if col not in ['time', 'attack', 'attack_P1', 'attack_P2', 'attack_P3'] 
                   and pd.api.types.is_numeric_dtype(train_df_processed[col])]
    
    # Calculate feature importance (using variance)
    variances = train_df_processed[feature_cols].var()
    feature_importance = {feature: score for feature, score in zip(feature_cols, variances)}
    
    # Select top features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:min(params['n_features'], len(sorted_features))]]
    
    # Extract features
    X_train = train_df_processed[top_features].values
    
    # Scale data using RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=params['n_components'])
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    print(f"Selected {len(top_features)} features out of {len(feature_cols)}")
    
    # Create sequences
    X_train_seq = create_sequences(X_train_pca, params['seq_length'], params['stride'])
    
    # Build model
    model = build_simplified_lstm_autoencoder(params['seq_length'], X_train_pca.shape[1], params['learning_rate'])
    model_path = f"{params['model_dir']}/simplified_lstm_single_model.h5"
    
    # Train model
    history = train_model(
        model, X_train_seq, params['epochs'], params['batch_size'], 
        params['validation_split'], params['patience'], model_path
    )
    
    # Determine threshold
    threshold = determine_threshold(model, X_train_seq, 'dynamic', params['contamination'])
    print(f"Dynamic threshold: {threshold:.6f}")
    
    # Save threshold
    with open(f"{params['model_dir']}/threshold.txt", 'w') as f:
        f.write(str(threshold))
    
    return model, G, scaler, pca, top_features, threshold

def evaluate_test_file(model, G, scaler, pca, top_features, threshold, test_name, test_df, params):
    """
    Evaluate a single test file using the trained model.
    
    Args:
        model (Model): Trained model
        G (nx.Graph): Graph structure
        scaler (RobustScaler): Fitted scaler
        pca (PCA): Fitted PCA
        top_features (list): Selected features
        threshold (float): Anomaly detection threshold
        test_name (str): Name of the test file
        test_df (pd.DataFrame): Test dataframe
        params (dict): Parameters dictionary
        
    Returns:
        dict: Results dictionary
    """
    print(f"\nEvaluating test file: {test_name}")
    
    # Apply graph-based feature engineering
    test_df_processed = simplified_graph_based_feature_engineering(test_df, G)
    
    # Extract features
    X_test = test_df_processed[top_features].values
    
    # Extract labels
    y_test = test_df['attack'].values if 'attack' in test_df.columns else None
    
    # Scale data
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA
    X_test_pca = pca.transform(X_test_scaled)
    
    # Create sequences
    X_test_seq = create_sequences(X_test_pca, params['seq_length'], params['stride'])
    
    # Detect anomalies
    anomaly_scores, anomaly_labels = detect_anomalies(
        model, X_test_seq, params['seq_length'], params['stride'], len(y_test),
        threshold, params['anomaly_score_threshold'], params['min_anomaly_length'], params['gap_threshold']
    )
    
    # Evaluate model
    eval_results = evaluate_model(y_test, anomaly_labels)
    
    # Visualize results
    visualize_results(y_test, anomaly_scores, anomaly_labels, test_name, params['anomaly_score_threshold'])
    
    # Return results
    return {
        'Test File': test_name,
        'Precision': eval_results['precision'],
        'Recall': eval_results['recall'],
        'F1 Score': eval_results['f1_score'],
        'Accuracy': eval_results['accuracy'],
        'False Positive Rate': eval_results['false_positive_rate'],
        'False Negative Rate': eval_results['false_negative_rate'],
        'Anomaly Scores': anomaly_scores,
        'Anomaly Labels': anomaly_labels,
        'Ground Truth': y_test
    }

def evaluate_on_all_test_files(processor, train_df, params):
    """
    Evaluate on all test files using a single trained model.
    
    Args:
        processor (HAIDataProcessor): Data processor
        train_df (pd.DataFrame): Training dataframe
        params (dict): Parameters dictionary
        
    Returns:
        pd.DataFrame: Results DataFrame
    """
    # Train a single model
    model, G, scaler, pca, top_features, threshold = train_single_model(train_df, params)
    
    results = []
    
    # For each test file
    for test_name, test_df in processor.test_data.items():
        result = evaluate_test_file(
            model, G, scaler, pca, top_features, threshold, 
            test_name, test_df, params
        )
        
        results.append({
            'Test File': result['Test File'],
            'Precision': result['Precision'],
            'Recall': result['Recall'],
            'F1 Score': result['F1 Score'],
            'Accuracy': result['Accuracy'],
            'False Positive Rate': result['False Positive Rate'],
            'False Negative Rate': result['False Negative Rate']
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df