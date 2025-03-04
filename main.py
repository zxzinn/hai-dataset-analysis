import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import argparse
import json
from datetime import datetime

# Import custom modules
from data_loader import TimeSeriesDataset, load_test_data_and_labels, get_dataset_info
from models import LSTMAutoencoder, TCNAutoencoder, TransformerAutoencoder
from train import train_model, evaluate_model, plot_training_history

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def setup_directories():
    """Create necessary directories for outputs"""
    dirs = ['models', 'results', 'plots']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    return dirs

def analyze_dataset(data_dir):
    """Analyze dataset files and print information"""
    # Define paths to dataset files
    train_files = [
        os.path.join(data_dir, 'hai-train1.csv'),
        os.path.join(data_dir, 'hai-train2.csv'),
        os.path.join(data_dir, 'hai-train3.csv'),
        os.path.join(data_dir, 'hai-train4.csv')
    ]
    test_files = [
        os.path.join(data_dir, 'hai-test1.csv'),
        os.path.join(data_dir, 'hai-test2.csv')
    ]
    label_files = [
        os.path.join(data_dir, 'label-test1.csv'),
        os.path.join(data_dir, 'label-test2.csv')
    ]
    
    # Check if files exist
    all_files = train_files + test_files + label_files
    existing_files = [f for f in all_files if os.path.exists(f)]
    
    if not existing_files:
        print(f"No dataset files found in {data_dir}")
        return None
    
    # Get info for existing files
    dataset_info = [get_dataset_info(file) for file in existing_files]
    info_df = pd.DataFrame(dataset_info)
    
    print("Dataset Information:")
    print(info_df)
    
    # Load a sample for visualization
    sample_file = existing_files[0]
    sample_data = pd.read_csv(sample_file, nrows=10000)
    
    # Plot sample data
    os.makedirs('plots/dataset_analysis', exist_ok=True)
    
    # Plot distribution of a few key features
    plt.figure(figsize=(20, 15))
    feature_cols = sample_data.columns[1:10]  # Select first 9 features after timestamp
    
    for i, col in enumerate(feature_cols):
        plt.subplot(3, 3, i+1)
        sns.histplot(sample_data[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
    
    plt.savefig('plots/dataset_analysis/feature_distributions.png')
    plt.close()
    
    # Plot time series of key features
    plt.figure(figsize=(20, 15))
    for i, col in enumerate(feature_cols):
        plt.subplot(3, 3, i+1)
        plt.plot(sample_data['timestamp'][:1000], sample_data[col][:1000])
        plt.title(f'Time Series of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.savefig('plots/dataset_analysis/time_series_plots.png')
    plt.close()
    
    # Plot correlation heatmap
    plt.figure(figsize=(20, 16))
    correlation_matrix = sample_data.iloc[:, 1:30].corr()  # Use first 30 features for clarity
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=False, mask=mask, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/dataset_analysis/correlation_heatmap.png')
    plt.close()
    
    return {
        'train_files': [f for f in train_files if os.path.exists(f)],
        'test_files': [f for f in test_files if os.path.exists(f)],
        'label_files': [f for f in label_files if os.path.exists(f)]
    }

def train_and_evaluate_models(dataset_files, seq_length=100, batch_size=128, epochs=20):
    """Train and evaluate multiple models on the dataset"""
    # Create datasets
    train_dataset = TimeSeriesDataset(dataset_files['train_files'], seq_length=seq_length)
    
    # Split into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train dataset size: {len(train_subset)}")
    print(f"Validation dataset size: {len(val_subset)}")
    
    # Get input dimensions from a sample batch
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[2]  # Number of features
    seq_len = sample_batch.shape[1]    # Sequence length
    
    print(f"Input dimensions: {input_dim} features, {seq_len} time steps")
    
    # Define models
    models = {
        'lstm': LSTMAutoencoder(input_dim=input_dim, hidden_dim=64, num_layers=2),
        'tcn': TCNAutoencoder(input_dim=input_dim, hidden_dims=[64, 32, 16]),
        'transformer': TransformerAutoencoder(input_dim=input_dim, d_model=128, nhead=8, num_layers=3)
    }
    
    # Print model summaries
    for name, model in models.items():
        print(f"{name.upper()} model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train models
    trained_models = {}
    histories = {}
    
    for name, model in models.items():
        print(f"\nTraining {name.upper()} model...")
        model_save_path = f"models/{name}_autoencoder.pth"
        
        trained_model, history = train_model(
            model, train_loader, val_loader, 
            epochs=epochs, lr=0.001 if name != 'transformer' else 0.0005, 
            patience=5, model_save_path=model_save_path
        )
        
        trained_models[name] = trained_model
        histories[name] = history
        
        # Plot training history
        plot_training_history(
            history, 
            title=f'{name.upper()} Training History',
            save_path=f'plots/{name}_training_history.png'
        )
    
    # Load test data
    test_results = {}
    
    for i, (test_file, label_file) in enumerate(zip(dataset_files['test_files'], dataset_files['label_files'])):
        print(f"\nEvaluating on test set {i+1}...")
        
        # Load test data
        test_sequences, test_labels, test_timestamps, _ = load_test_data_and_labels(
            test_file, label_file, seq_length=seq_length, scaler=train_dataset.scaler
        )
        
        print(f"Test sequences shape: {test_sequences.shape}")
        print(f"Test labels shape: {test_labels.shape}")
        print(f"Number of anomalies: {np.sum(test_labels)}")
        
        # Create test dataset and dataloader
        test_tensor_dataset = TensorDataset(torch.tensor(test_sequences, dtype=torch.float32))
        test_loader = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Evaluate each model
        for name, model in trained_models.items():
            print(f"Evaluating {name.upper()} model...")
            results = evaluate_model(
                model, test_loader, test_labels, 
                save_dir=f'plots/test{i+1}/{name}'
            )
            
            # Save metrics
            metrics = {
                'threshold': float(results['threshold']),
                'f1_score': float(results['f1_score']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'roc_auc': float(results['roc_auc']),
                'confusion_matrix': results['confusion_matrix'].tolist()
            }
            
            os.makedirs(f'results/test{i+1}', exist_ok=True)
            with open(f'results/test{i+1}/{name}_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Store results for comparison
            if i not in test_results:
                test_results[i] = {}
            test_results[i][name] = results
    
    # Compare models
    for i in test_results:
        print(f"\nTest Set {i+1} Results:")
        metrics_df = pd.DataFrame({
            'Model': list(test_results[i].keys()),
            'F1 Score': [test_results[i][model]['f1_score'] for model in test_results[i]],
            'Precision': [test_results[i][model]['precision'] for model in test_results[i]],
            'Recall': [test_results[i][model]['recall'] for model in test_results[i]],
            'ROC AUC': [test_results[i][model]['roc_auc'] for model in test_results[i]]
        })
        print(metrics_df)
        
        # Save comparison
        metrics_df.to_csv(f'results/test{i+1}/model_comparison.csv', index=False)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        metrics = ['F1 Score', 'Precision', 'Recall', 'ROC AUC']
        
        for j, metric in enumerate(metrics):
            plt.subplot(2, 2, j+1)
            sns.barplot(x='Model', y=metric, data=metrics_df)
            plt.title(f'Model Comparison - {metric}')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        plt.savefig(f'plots/test{i+1}/model_comparison.png')
        plt.close()
    
    return trained_models, test_results

def create_ensemble_model(models, test_results):
    """Create and evaluate an ensemble model"""
    ensemble_results = {}
    
    for i in test_results:
        print(f"\nCreating ensemble for Test Set {i+1}...")
        
        # Average errors from all models
        all_errors = np.stack([test_results[i][model]['errors'] for model in models])
        ensemble_errors = np.mean(all_errors, axis=0)
        
        # Get labels
        test_labels = next(iter(test_results[i].values()))['predictions']
        true_labels = next(iter(test_results[i].values()))['predictions']
        
        # Find optimal threshold
        from train import find_optimal_threshold
        threshold, f1 = find_optimal_threshold(ensemble_errors, true_labels)
        
        # Get predictions
        predictions = (ensemble_errors > threshold).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix
        precision, recall, _, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', zero_division=0)
        fpr, tpr, _ = roc_curve(true_labels, ensemble_errors)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(true_labels, predictions)
        
        # Save plots
        os.makedirs(f'plots/test{i+1}/ensemble', exist_ok=True)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Ensemble Model - ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(f'plots/test{i+1}/ensemble/roc_curve.png')
        plt.close()
        
        # Plot error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(ensemble_errors[true_labels == 0], bins=50, alpha=0.5, label='Normal')
        plt.hist(ensemble_errors[true_labels == 1], bins=50, alpha=0.5, label='Anomaly')
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Count')
        plt.title('Ensemble Model - Error Distribution')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/test{i+1}/ensemble/error_distribution.png')
        plt.close()
        
        # Save metrics
        metrics = {
            'threshold': float(threshold),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist()
        }
        
        os.makedirs(f'results/test{i+1}', exist_ok=True)
        with open(f'results/test{i+1}/ensemble_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Add to comparison
        metrics_df = pd.read_csv(f'results/test{i+1}/model_comparison.csv')
        ensemble_df = pd.DataFrame({
            'Model': ['Ensemble'],
            'F1 Score': [f1],
            'Precision': [precision],
            'Recall': [recall],
            'ROC AUC': [roc_auc]
        })
        metrics_df = pd.concat([metrics_df, ensemble_df], ignore_index=True)
        metrics_df.to_csv(f'results/test{i+1}/model_comparison.csv', index=False)
        
        # Update plot
        plt.figure(figsize=(12, 8))
        metrics = ['F1 Score', 'Precision', 'Recall', 'ROC AUC']
        
        for j, metric in enumerate(metrics):
            plt.subplot(2, 2, j+1)
            sns.barplot(x='Model', y=metric, data=metrics_df)
            plt.title(f'Model Comparison - {metric}')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        plt.savefig(f'plots/test{i+1}/model_comparison.png')
        plt.close()
        
        # Store results
        ensemble_results[i] = {
            'threshold': threshold,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'errors': ensemble_errors,
            'predictions': predictions
        }
    
    return ensemble_results

def main():
    parser = argparse.ArgumentParser(description='HAI Dataset Analysis and Anomaly Detection')
    parser.add_argument('--data_dir', type=str, default='hai-security-dataset/hai-23.05/', help='Path to dataset directory')
    parser.add_argument('--seq_length', type=int, default=100, help='Sequence length for time series')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Print system information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Available GPU memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Analyze dataset
    print("\nAnalyzing dataset...")
    dataset_files = analyze_dataset(args.data_dir)
    
    if not dataset_files:
        print("No dataset files found. Exiting.")
        return
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    trained_models, test_results = train_and_evaluate_models(
        dataset_files, 
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Create ensemble model
    print("\nCreating ensemble model...")
    ensemble_results = create_ensemble_model(trained_models, test_results)
    
    print("\nAnalysis complete. Results saved to 'results/' directory.")
    print("Plots saved to 'plots/' directory.")
    print("Models saved to 'models/' directory.")

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Starting analysis at {start_time}")
    
    main()
    
    end_time = datetime.now()
    print(f"Analysis completed at {end_time}")
    print(f"Total runtime: {end_time - start_time}")