import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, patience=5, model_save_path=None):
    """
    Train a model with mixed precision and early stopping
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Maximum number of epochs to train
        lr: Learning rate
        patience: Number of epochs to wait for improvement before early stopping
        model_save_path: Path to save the best model
        
    Returns:
        model: Trained model
        history: Dictionary containing training and validation losses
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience//2, factor=0.5)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
            
            if model_save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, model_save_path)
                print(f"Model saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model:
        model.load_state_dict(best_model)
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}


def calculate_reconstruction_error(model, data_loader):
    """
    Calculate reconstruction error for anomaly detection
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for test data
        
    Returns:
        errors: Numpy array of reconstruction errors
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    errors = []
    
    with torch.no_grad():
        for inputs in tqdm(data_loader, desc="Calculating reconstruction error"):
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = inputs[0]  # Handle different DataLoader formats
            
            inputs = inputs.to(device)
            
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    # Calculate error for each feature and time step
                    error = criterion(outputs, inputs)
            else:
                outputs = model(inputs)
                error = criterion(outputs, inputs)
            
            # Average over features
            error = error.mean(dim=2)
            # Keep time dimension for anomaly detection
            errors.append(error.cpu().numpy())
    
    return np.concatenate(errors)


def find_optimal_threshold(errors, labels):
    """
    Find the optimal threshold for anomaly detection based on F1 score
    
    Args:
        errors: Numpy array of reconstruction errors
        labels: Numpy array of true labels (0 for normal, 1 for anomaly)
        
    Returns:
        optimal_threshold: Optimal threshold value
        optimal_f1: F1 score at the optimal threshold
    """
    thresholds = np.linspace(np.min(errors), np.max(errors), 100)
    f1_scores = []
    
    for threshold in thresholds:
        predictions = (errors > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    return optimal_threshold, optimal_f1


def plot_training_history(history, title='Model Training History', save_path=None):
    """
    Plot training and validation losses
    
    Args:
        history: Dictionary containing training and validation losses
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def evaluate_model(model, test_loader, test_labels, save_dir=None):
    """
    Evaluate model performance for anomaly detection
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        test_labels: True labels for test data
        save_dir: Directory to save evaluation plots
        
    Returns:
        results: Dictionary containing evaluation metrics
    """
    # Calculate reconstruction errors
    errors = calculate_reconstruction_error(model, test_loader)
    
    # Average over time steps to get a single error value per sequence
    errors_mean = errors.mean(axis=1)
    
    # Find optimal threshold
    threshold, f1 = find_optimal_threshold(errors_mean, test_labels)
    
    # Get predictions
    predictions = (errors_mean > threshold).astype(int)
    
    # Calculate metrics
    precision, recall, _, _ = precision_recall_fscore_support(test_labels, predictions, average='binary', zero_division=0)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(test_labels, errors_mean)
    roc_auc = auc(fpr, tpr)
    
    # Create confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    
    # Save plots if directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
        plt.close()
        
        # Plot reconstruction error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(errors_mean[test_labels == 0], bins=50, alpha=0.5, label='Normal')
        plt.hist(errors_mean[test_labels == 1], bins=50, alpha=0.5, label='Anomaly')
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Count')
        plt.title('Reconstruction Error Distribution')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'error_distribution.png'))
        plt.close()
        
        # Plot reconstruction error over time
        plt.figure(figsize=(12, 6))
        plt.plot(errors_mean)
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
        # Mark anomalies
        anomaly_indices = np.where(test_labels == 1)[0]
        plt.scatter(anomaly_indices, errors_mean[anomaly_indices], color='red', label='True Anomalies', alpha=0.5)
        plt.xlabel('Sequence Index')
        plt.ylabel('Reconstruction Error')
        plt.title('Reconstruction Error Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'error_time_series.png'))
        plt.close()
    
    return {
        'threshold': threshold,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'errors': errors_mean,
        'predictions': predictions
    }