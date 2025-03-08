#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved Graph-based Feature Engineering and Anomaly Detection for HAI-21.03 Dataset

This module provides enhanced methods for graph-based feature engineering and anomaly detection
for the HAI-21.03 industrial control system security dataset, focusing on reducing false positives
while maintaining high detection rates.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Attention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

class ImprovedHAIGraphFeatures:
    """
    Enhanced graph-based feature engineering and anomaly detection for HAI-21.03 dataset.
    
    This class provides methods for creating graph structures from the HAI dataset,
    performing graph-based feature engineering, and detecting anomalies using
    improved LSTM autoencoder models.
    """
    
    def __init__(self, correlation_threshold=0.7, min_anomaly_length=20, gap_threshold=3):
        """
        Initialize the ImprovedHAIGraphFeatures class.
        
        Args:
            correlation_threshold (float): Threshold for adding edges between components in the graph
            min_anomaly_length (int): Minimum length of anomalies to keep (to reduce false positives)
            gap_threshold (int): Maximum gap between anomalies to merge
        """
        self.correlation_threshold = correlation_threshold
        self.min_anomaly_length = min_anomaly_length
        self.gap_threshold = gap_threshold
        self.graph = None
        self.scaler = None
        self.pca = None
        self.model = None
        self.threshold = None
        self.feature_importance = None
        
    def create_graph_from_columns(self, df):
        """
        Create a graph structure based on column names and their correlations.
        
        Args:
            df (pd.DataFrame): DataFrame with sensor/actuator columns
            
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
        
        # Add edges between components based on correlation
        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i < j and corr_matrix.loc[col1, col2] > self.correlation_threshold:
                    G.add_edge(col1, col2, weight=corr_matrix.loc[col1, col2])
        
        self.graph = G
        return G
    
    def calculate_graph_centrality(self, G=None):
        """
        Calculate various centrality measures for nodes in the graph.
        
        Args:
            G (nx.Graph, optional): NetworkX graph. If None, uses self.graph
            
        Returns:
            dict: Dictionary of centrality measures for each node
        """
        if G is None:
            G = self.graph
            
        if G is None:
            raise ValueError("Graph not created yet. Call create_graph_from_columns first.")
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # Combine centrality measures
        centrality = {}
        for node in G.nodes():
            centrality[node] = {
                'degree': degree_centrality[node],
                'betweenness': betweenness_centrality[node],
                'closeness': closeness_centrality[node]
            }
        
        return centrality
    
    def get_community_structure(self, G=None):
        """
        Detect communities in the graph using the Louvain method.
        
        Args:
            G (nx.Graph, optional): NetworkX graph. If None, uses self.graph
            
        Returns:
            dict: Dictionary mapping nodes to community IDs
        """
        if G is None:
            G = self.graph
            
        if G is None:
            raise ValueError("Graph not created yet. Call create_graph_from_columns first.")
        
        # Use Louvain method for community detection
        try:
            from community import best_partition
            communities = best_partition(G)
        except ImportError:
            # Fallback to connected components if python-louvain is not installed
            communities = {}
            for i, component in enumerate(nx.connected_components(G)):
                for node in component:
                    communities[node] = i
        
        return communities
    
    def graph_based_feature_engineering(self, df, G=None, include_centrality=True, include_communities=True):
        """
        Create features based on graph structure with improved feature selection.
        
        Args:
            df (pd.DataFrame): DataFrame with sensor/actuator data
            G (nx.Graph, optional): NetworkX graph. If None, uses self.graph
            include_centrality (bool): Whether to include centrality-based features
            include_communities (bool): Whether to include community-based features
            
        Returns:
            pd.DataFrame: DataFrame with additional graph-based features
        """
        if G is None:
            G = self.graph
            
        if G is None:
            raise ValueError("Graph not created yet. Call create_graph_from_columns first.")
        
        df_processed = df.copy()
        
        # Extract columns excluding time and attack columns
        cols = [col for col in df.columns if not col.startswith('time') and not col.startswith('attack')]
        
        # Add subsystem-level features
        for subsystem in ['P1', 'P2', 'P3', 'P4']:
            subsystem_cols = [col for col in cols if col.startswith(f"{subsystem}_")]
            if subsystem_cols:
                # Calculate subsystem-level statistics
                df_processed[f"{subsystem}_graph_mean"] = df[subsystem_cols].mean(axis=1)
                df_processed[f"{subsystem}_graph_std"] = df[subsystem_cols].std(axis=1)
                df_processed[f"{subsystem}_graph_max"] = df[subsystem_cols].max(axis=1)
                df_processed[f"{subsystem}_graph_min"] = df[subsystem_cols].min(axis=1)
                
                # Calculate z-scores within subsystem (more robust)
                for col in subsystem_cols:
                    mean = df_processed[f"{subsystem}_graph_mean"]
                    std = df_processed[f"{subsystem}_graph_std"].replace(0, 1e-10)  # Avoid division by zero
                    df_processed[f"{col}_zscore"] = (df[col] - mean) / std
        
        # Add cross-subsystem features (more selective)
        subsystems = ['P1', 'P2', 'P3', 'P4']
        for i, s1 in enumerate(subsystems):
            for j, s2 in enumerate(subsystems):
                if i < j:
                    # Calculate ratio and difference between subsystem means
                    if f"{s1}_graph_mean" in df_processed.columns and f"{s2}_graph_mean" in df_processed.columns:
                        s2_mean = df_processed[f"{s2}_graph_mean"].replace(0, 1e-10)  # Avoid division by zero
                        df_processed[f"{s1}_{s2}_ratio"] = df_processed[f"{s1}_graph_mean"] / s2_mean
                        df_processed[f"{s1}_{s2}_diff"] = df_processed[f"{s1}_graph_mean"] - df_processed[f"{s2}_graph_mean"]
        
        # Add rolling window features (more selective)
        window_sizes = [5, 10]  # Reduced from [5, 10, 20] to be more selective
        for window in window_sizes:
            for subsystem in subsystems:
                col = f"{subsystem}_graph_mean"
                if col in df_processed.columns:
                    df_processed[f"{col}_rolling_{window}_mean"] = df_processed[col].rolling(window=window).mean()
                    df_processed[f"{col}_rolling_{window}_std"] = df_processed[col].rolling(window=window).std()
                    
                    # Calculate rate of change
                    df_processed[f"{col}_rolling_{window}_diff"] = df_processed[col].diff(window)
        
        # Add centrality-based features if requested
        if include_centrality:
            centrality = self.calculate_graph_centrality(G)
            
            # For each subsystem, calculate weighted average of component values based on centrality
            for subsystem in subsystems:
                subsystem_cols = [col for col in cols if col.startswith(f"{subsystem}_")]
                if subsystem_cols:
                    # Calculate degree centrality weighted average
                    weights = np.array([centrality.get(col, {}).get('degree', 0) for col in subsystem_cols])
                    if weights.sum() > 0:
                        weights = weights / weights.sum()  # Normalize weights
                        df_processed[f"{subsystem}_degree_weighted_mean"] = np.dot(df[subsystem_cols].values, weights)
                    
                    # Calculate betweenness centrality weighted average
                    weights = np.array([centrality.get(col, {}).get('betweenness', 0) for col in subsystem_cols])
                    if weights.sum() > 0:
                        weights = weights / weights.sum()  # Normalize weights
                        df_processed[f"{subsystem}_betweenness_weighted_mean"] = np.dot(df[subsystem_cols].values, weights)
        
        # Add community-based features if requested
        if include_communities:
            communities = self.get_community_structure(G)
            
            # Get unique community IDs
            community_ids = set(communities.values())
            
            # For each community, calculate statistics
            for community_id in community_ids:
                community_cols = [col for col in cols if col in communities and communities[col] == community_id]
                if community_cols:
                    df_processed[f"community_{community_id}_mean"] = df[community_cols].mean(axis=1)
                    df_processed[f"community_{community_id}_std"] = df[community_cols].std(axis=1)
        
        # Fill NaN values
        df_processed = df_processed.fillna(method='bfill').fillna(method='ffill')
        
        return df_processed
    
    def select_features_and_reduce_dimensions(self, train_df, test_df, n_components=40, feature_selection_method='isolation_forest'):
        """
        Select features and reduce dimensions with improved feature selection.
        
        Args:
            train_df (pd.DataFrame): Training dataframe
            test_df (pd.DataFrame): Test dataframe
            n_components (int): Number of PCA components
            feature_selection_method (str): Method for feature selection ('isolation_forest' or 'correlation')
            
        Returns:
            tuple: (X_train_pca, X_test_pca, y_test, scaler, pca, selected_features)
        """
        # Exclude non-numeric and target columns
        exclude_cols = ['time', 'attack', 'attack_P1', 'attack_P2', 'attack_P3']
        feature_cols = [col for col in train_df.columns if col not in exclude_cols and 
                        pd.api.types.is_numeric_dtype(train_df[col])]
        
        # Extract labels if available
        y_test = test_df['attack'].values if 'attack' in test_df.columns else None
        
        # Feature selection
        if feature_selection_method == 'isolation_forest':
            # Use Isolation Forest to identify important features
            iso_forest = IsolationForest(random_state=42, contamination=0.01)
            iso_forest.fit(train_df[feature_cols])
            
            # Calculate feature importance based on average path length
            feature_importance = {}
            for i, col in enumerate(feature_cols):
                X_feature = train_df[feature_cols].copy()
                X_feature_permuted = X_feature.copy()
                X_feature_permuted[col] = np.random.permutation(X_feature_permuted[col].values)
                
                # Calculate scores for original and permuted data
                original_scores = iso_forest.decision_function(X_feature)
                permuted_scores = iso_forest.decision_function(X_feature_permuted)
                
                # Feature importance is the mean absolute difference in scores
                importance = np.mean(np.abs(original_scores - permuted_scores))
                feature_importance[col] = importance
            
            # Select top features
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_features[:min(100, len(sorted_features))]]
            
            self.feature_importance = feature_importance
            
        elif feature_selection_method == 'correlation':
            # Use correlation-based feature selection
            corr_matrix = train_df[feature_cols].corr().abs()
            
            # For each feature, calculate the mean correlation with other features
            mean_corr = {}
            for col in feature_cols:
                mean_corr[col] = corr_matrix[col].mean()
            
            # Select features with lower mean correlation (less redundant)
            sorted_features = sorted(mean_corr.items(), key=lambda x: x[1])
            top_features = [f[0] for f in sorted_features[:min(100, len(sorted_features))]]
            
            self.feature_importance = mean_corr
        else:
            # No feature selection
            top_features = feature_cols
        
        # Extract features
        X_train = train_df[top_features].values
        X_test = test_df[top_features].values
        
        # Scale data using RobustScaler (less sensitive to outliers)
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA for dimensionality reduction
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=n_components)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        print(f"Explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        print(f"Selected {len(top_features)} features out of {len(feature_cols)}")
        
        return X_train_pca, X_test_pca, y_test, self.scaler, self.pca, top_features
    
    def create_sequences(self, data, seq_length, stride=1):
        """
        Create sequences for time series models.
        
        Args:
            data (np.array): Input data
            seq_length (int): Sequence length
            stride (int): Stride for sliding window
            
        Returns:
            np.array: Sequences
        """
        sequences = []
        for i in range(0, len(data) - seq_length + 1, stride):
            seq = data[i:i+seq_length]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def build_improved_lstm_autoencoder(self, seq_length, n_features):
        """
        Build an improved bidirectional LSTM autoencoder model with attention mechanism.
        
        Args:
            seq_length (int): Sequence length
            n_features (int): Number of features
            
        Returns:
            Model: Keras model
        """
        # Input layer
        inputs = Input(shape=(seq_length, n_features))
        
        # Encoder
        x = Bidirectional(LSTM(64, activation='tanh', return_sequences=True, 
                              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5)))(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Bidirectional(LSTM(32, activation='tanh', return_sequences=False, 
                              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5)))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Bottleneck
        encoded = Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5))(x)
        
        # Decoder
        x = Dense(32, activation='relu')(encoded)
        x = tf.keras.layers.RepeatVector(seq_length)(x)
        
        x = Bidirectional(LSTM(32, activation='tanh', return_sequences=True, 
                              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5)))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Bidirectional(LSTM(64, activation='tanh', return_sequences=True, 
                              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5)))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        outputs = tf.keras.layers.TimeDistributed(Dense(n_features))(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
    
    def train_model(self, X_train_seq, epochs=100, batch_size=64, validation_split=0.2, patience=15, model_path=None):
        """
        Train the LSTM autoencoder model with improved training parameters.
        
        Args:
            X_train_seq (np.array): Training sequences
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
            patience (int): Patience for early stopping
            model_path (str, optional): Path to save the best model
            
        Returns:
            History: Training history
        """
        # Define callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        if model_path:
            callbacks.append(ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True))
        
        # Train model
        history = self.model.fit(
            X_train_seq, X_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def determine_threshold(self, X_train_seq, method='dynamic', contamination=0.01):
        """
        Determine the anomaly threshold using improved methods.
        
        Args:
            X_train_seq (np.array): Training sequences
            method (str): Method for threshold determination ('dynamic', 'iqr', or 'percentile')
            contamination (float): Expected proportion of anomalies (for dynamic method)
            
        Returns:
            float: Anomaly threshold
        """
        # Predict on training data
        X_train_pred = self.model.predict(X_train_seq)
        
        # Calculate reconstruction error
        train_mse = np.mean(np.square(X_train_seq - X_train_pred), axis=(1, 2))
        
        if method == 'dynamic':
            # Use precision-recall curve to find optimal threshold
            # This requires a small portion of anomalies in the training data
            # We'll simulate anomalies by taking the highest reconstruction errors
            simulated_anomalies = np.zeros(len(train_mse))
            anomaly_count = int(contamination * len(train_mse))
            anomaly_indices = np.argsort(train_mse)[-anomaly_count:]
            simulated_anomalies[anomaly_indices] = 1
            
            # Calculate precision-recall curve
            precisions, recalls, thresholds = precision_recall_curve(simulated_anomalies, train_mse)
            
            # Find threshold that maximizes F1 score
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            threshold = thresholds[optimal_idx]
            
        elif method == 'iqr':
            # IQR method (more robust to outliers)
            q1 = np.percentile(train_mse, 25)
            q3 = np.percentile(train_mse, 75)
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr
            
        elif method == 'percentile':
            # Percentile method
            threshold = np.percentile(train_mse, 99)
            
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        self.threshold = threshold
        return threshold
    
    def detect_anomalies_with_improved_postprocessing(self, mse, threshold, seq_length, stride, data_length):
        """
        Detect anomalies with improved post-processing to reduce false positives.
        
        Args:
            mse (np.array): Reconstruction errors
            threshold (float): Anomaly threshold
            seq_length (int): Sequence length
            stride (int): Stride used for sequences
            data_length (int): Original data length
            
        Returns:
            tuple: (anomaly_scores, anomaly_labels)
        """
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
        # Increased threshold from 0.5 to 0.6 to reduce false positives
        anomaly_labels = (anomaly_scores > 0.6).astype(int)
        
        # Remove short anomalies (likely false positives)
        i = 0
        while i < len(anomaly_labels):
            if anomaly_labels[i] == 1:
                # Find the end of this anomaly
                j = i
                while j < len(anomaly_labels) and anomaly_labels[j] == 1:
                    j += 1
                
                # If anomaly is too short, remove it
                if j - i < self.min_anomaly_length:
                    anomaly_labels[i:j] = 0
                
                i = j
            else:
                i += 1
        
        # Merge nearby anomalies
        i = 0
        while i < len(anomaly_labels):
            if anomaly_labels[i] == 1:
                # Find the end of this anomaly
                j = i
                while j < len(anomaly_labels) and anomaly_labels[j] == 1:
                    j += 1
                
                # Look for another anomaly nearby
                if j < len(anomaly_labels) - self.gap_threshold:
                    next_start = j
                    while next_start < j + self.gap_threshold and next_start < len(anomaly_labels) and anomaly_labels[next_start] == 0:
                        next_start += 1
                    
                    if next_start < j + self.gap_threshold and next_start < len(anomaly_labels) and anomaly_labels[next_start] == 1:
                        anomaly_labels[j:next_start] = 1
                
                i = j
            else:
                i += 1
        
        return anomaly_scores, anomaly_labels
    
    def run_anomaly_detection_pipeline(self, train_df, test_df, seq_length=50, stride=10, 
                                      n_components=40, feature_selection_method='isolation_forest',
                                      threshold_method='dynamic', contamination=0.01,
                                      epochs=100, batch_size=64, validation_split=0.2, 
                                      patience=15, model_path=None):
        """
        Run the complete anomaly detection pipeline.
        
        Args:
            train_df (pd.DataFrame): Training dataframe
            test_df (pd.DataFrame): Test dataframe
            seq_length (int): Sequence length for time series
            stride (int): Stride for sliding window
            n_components (int): Number of PCA components
            feature_selection_method (str): Method for feature selection
            threshold_method (str): Method for threshold determination
            contamination (float): Expected proportion of anomalies
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
            patience (int): Patience for early stopping
            model_path (str, optional): Path to save the best model
            
        Returns:
            tuple: (anomaly_scores, anomaly_labels, y_test, threshold)
        """
        # Create graph structure
        print("Creating graph structure...")
        G = self.create_graph_from_columns(train_df)
        
        # Perform graph-based feature engineering
        print("Performing graph-based feature engineering...")
        train_df_processed = self.graph_based_feature_engineering(train_df, G)
        test_df_processed = self.graph_based_feature_engineering(test_df, G)
        
        # Select features and reduce dimensions
        print("Selecting features and reducing dimensions...")
        X_train_pca, X_test_pca, y_test, _, _, _ = self.select_features_and_reduce_dimensions(
            train_df_processed, test_df_processed, n_components, feature_selection_method
        )
        
        # Create sequences
        print("Creating sequences...")
        X_train_seq = self.create_sequences(X_train_pca, seq_length, stride)
        X_test_seq = self.create_sequences(X_test_pca, seq_length, stride)
        
        # Build model
        print("Building model...")
        self.model = self.build_improved_lstm_autoencoder(seq_length, X_train_pca.shape[1])
        
        # Train model
        print("Training model...")
        history = self.train_model(X_train_seq, epochs, batch_size, validation_split, patience, model_path)
        
        # Determine threshold
        print("Determining threshold...")
        threshold = self.determine_threshold(X_train_seq, threshold_method, contamination)
        
        # Predict on test data
        print("Predicting on test data...")
        X_test_pred = self.model.predict(X_test_seq)
        
        # Calculate reconstruction error
        mse = np.mean(np.square(X_test_seq - X_test_pred), axis=(1, 2))
        
        # Detect anomalies
        print("Detecting anomalies...")
        anomaly_scores, anomaly_labels = self.detect_anomalies_with_improved_postprocessing(
            mse, threshold, seq_length, stride, len(y_test)
        )
        
        return anomaly_scores, anomaly_labels, y_test, threshold