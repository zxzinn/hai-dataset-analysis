import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, BatchNormalization, Add, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
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

def load_dataset(dataset_dir, dataset_name):
    """
    Load all Parquet files from a dataset directory
    
    Parameters:
    dataset_dir: Base directory containing all datasets
    dataset_name: Name of the specific dataset (e.g., 'hai-20.07')
    
    Returns:
    data_dict: Dictionary containing DataFrames for each file
    """
    dataset_path = os.path.join(dataset_dir, dataset_name)
    data_dict = {}
    
    for file in os.listdir(dataset_path):
        if file.endswith('.parquet'):
            file_path = os.path.join(dataset_path, file)
            file_name = os.path.splitext(file)[0]
            data_dict[file_name] = load_parquet_data(file_path)
            
    return data_dict

def plot_time_series(df, columns, time_col='time', n_cols=3, figsize=(18, 12)):
    """
    Plot time series data for selected columns
    
    Parameters:
    df: DataFrame containing the data
    columns: List of columns to plot
    time_col: Name of the time column
    n_cols: Number of columns in the subplot grid
    figsize: Figure size
    """
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(axes):
            df.plot(x=time_col, y=col, ax=axes[i])
            axes[i].set_title(col)
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(df, n_features=20, figsize=(12, 10)):
    """
    Plot correlation matrix for top features
    
    Parameters:
    df: DataFrame containing the data
    n_features: Number of top features to include
    figsize: Figure size
    """
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Get top correlated features
    if n_features < len(corr.columns):
        # Get the average correlation for each feature
        avg_corr = corr.abs().mean().sort_values(ascending=False)
        top_features = avg_corr.index[:n_features]
        corr = corr.loc[top_features, top_features]
    
    # Plot
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                annot=False, square=True, linewidths=.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    return plt.gcf()

def plot_attack_distribution(df, attack_col='attack'):
    """
    Plot the distribution of attack vs normal samples
    
    Parameters:
    df: DataFrame containing the data
    attack_col: Name of the attack column
    """
    if attack_col in df.columns:
        plt.figure(figsize=(10, 6))
        attack_counts = df[attack_col].value_counts()
        ax = sns.barplot(x=attack_counts.index, y=attack_counts.values)
        
        # Add percentage labels
        total = attack_counts.sum()
        for i, count in enumerate(attack_counts.values):
            percentage = count / total * 100
            ax.text(i, count + 0.1, f'{percentage:.1f}%', ha='center')
        
        plt.title('Distribution of Attack vs Normal Samples')
        plt.xlabel('Attack (1) vs Normal (0)')
        plt.ylabel('Count')
        plt.tight_layout()
        return plt.gcf()
    else:
        print(f"Warning: '{attack_col}' column not found in the DataFrame")
        return None

def create_sequences(X, y, time_steps=100, step=1):
    """
    Create sequences for time series data
    
    Parameters:
    X: Feature data
    y: Target data
    time_steps: Number of time steps in each sequence
    step: Step size between sequences
    
    Returns:
    X_seq: Sequences of feature data
    y_seq: Sequences of target data
    """
    X_seq, y_seq = [], []
    for i in range(0, len(X) - time_steps, step):
        X_seq.append(X[i:i + time_steps])
        # Use the label of the last time step in the sequence
        y_seq.append(y[i + time_steps - 1])
    
    return np.array(X_seq), np.array(y_seq)

def select_features(X_train, y_train, X_test, k=20):
    """
    Select top k features based on ANOVA F-value
    
    Parameters:
    X_train: Training features
    y_train: Training target
    X_test: Test features
    k: Number of features to select
    
    Returns:
    X_train_selected: Selected training features
    X_test_selected: Selected test features
    selected_indices: Indices of selected features
    """
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get indices of selected features
    selected_indices = selector.get_support(indices=True)
    
    return X_train_selected, X_test_selected, selected_indices

def preprocess_data(train_df, test_df, target_col='attack', time_col=None, 
                    feature_selection=True, n_features=20, scaler_type='standard'):
    """
    Preprocess data for model training
    
    Parameters:
    train_df: Training DataFrame
    test_df: Test DataFrame
    target_col: Name of the target column
    time_col: Name of the time column (to be excluded from features)
    feature_selection: Whether to perform feature selection
    n_features: Number of features to select if feature_selection is True
    scaler_type: Type of scaler ('standard' or 'minmax')
    
    Returns:
    X_train: Preprocessed training features
    X_test: Preprocessed test features
    y_train: Training target
    y_test: Test target
    feature_names: Names of selected features
    scaler: Fitted scaler object
    """
    # Drop time column if specified
    if time_col is not None and time_col in train_df.columns:
        train_features = train_df.drop([time_col, target_col], axis=1)
        test_features = test_df.drop([time_col, target_col], axis=1)
    else:
        train_features = train_df.drop([target_col], axis=1)
        test_features = test_df.drop([target_col], axis=1)
    
    # Store original feature names
    feature_names = train_features.columns.tolist()
    
    # Convert to numpy arrays
    X_train = train_features.values
    X_test = test_features.values
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    # Apply scaling
    if scaler_type.lower() == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Apply feature selection if requested
    if feature_selection and n_features < X_train.shape[1]:
        X_train, X_test, selected_indices = select_features(X_train, y_train, X_test, k=n_features)
        feature_names = [feature_names[i] for i in selected_indices]
    
    return X_train, X_test, y_train, y_test, feature_names, scaler

def create_residual_bilstm_model(input_shape, lstm_units=64, dense_units=32, dropout_rate=0.3):
    """
    Create a Residual Bidirectional LSTM model
    
    Parameters:
    input_shape: Shape of input data (time_steps, n_features)
    lstm_units: Number of LSTM units
    dense_units: Number of dense units
    dropout_rate: Dropout rate
    
    Returns:
    model: Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First Bidirectional LSTM layer
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Residual block 1
    residual = x
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Add()([x, residual])  # Add residual connection
    
    # Residual block 2
    residual = x
    x = Bidirectional(LSTM(lstm_units, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Dense layers
    x = Dense(dense_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, patience=10, model_path='best_model.h5'):
    """
    Train the model with early stopping and model checkpointing
    
    Parameters:
    model: Keras model
    X_train: Training features
    y_train: Training target
    X_val: Validation features
    y_val: Validation target
    batch_size: Batch size
    epochs: Maximum number of epochs
    patience: Patience for early stopping
    model_path: Path to save the best model
    
    Returns:
    history: Training history
    model: Trained model
    """
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    
    return history, model

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate the model on test data
    
    Parameters:
    model: Trained Keras model
    X_test: Test features
    y_test: Test target
    threshold: Classification threshold
    
    Returns:
    results: Dictionary containing evaluation metrics
    """
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return results

def plot_training_history(history):
    """
    Plot training history
    
    Parameters:
    history: Training history from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_evaluation_results(results):
    """
    Plot evaluation results
    
    Parameters:
    results: Dictionary containing evaluation metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot confusion matrix
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Plot ROC curve
    ax2.plot(results['fpr'], results['tpr'], label=f'ROC Curve (AUC = {results["auc"]:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, n_top=20):
    """
    Plot feature importance for a trained model
    
    Parameters:
    model: Trained Keras model
    feature_names: List of feature names
    n_top: Number of top features to display
    """
    # For simplicity, we'll use the weights of the first layer as a proxy for feature importance
    # This is a simplification and not always accurate for complex models
    weights = model.layers[1].get_weights()[0]
    
    # Calculate importance as the sum of absolute weights for each feature
    importance = np.sum(np.abs(weights), axis=1)
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    
    # Select top features
    if n_top < len(feature_names):
        indices = indices[:n_top]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    return plt.gcf()