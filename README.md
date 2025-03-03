# HAI Security Dataset Analysis

This project analyzes the HAI (HIL-based Augmented ICS) Security Dataset for anomaly detection in industrial control systems. The project implements multiple deep learning models to detect anomalies in time series data collected from industrial control systems.

## Dataset Overview

The HAI Security Dataset contains data from industrial control systems with normal operations and attack scenarios:

- HAI-23.05: 86 features representing various sensors and control points
- Time-series data collected at 1-second intervals
- Contains both normal operation data and attack scenarios
- Attacks represent a small percentage of the total data (imbalanced classification problem)

## Project Structure

```
├── data_loader.py     # Data loading and preprocessing utilities
├── models.py          # Model architectures (LSTM, TCN, Transformer)
├── train.py           # Training and evaluation functions
├── main.py            # Main script to run the analysis
├── models/            # Saved model weights
├── results/           # Evaluation metrics and results
└── plots/             # Visualizations and plots
```

## Models Implemented

1. **LSTM Autoencoder**: Long Short-Term Memory based autoencoder for sequence modeling
2. **TCN Autoencoder**: Temporal Convolutional Network based autoencoder for capturing local patterns
3. **Transformer Autoencoder**: Transformer-based autoencoder for capturing long-range dependencies
4. **Ensemble Model**: Combination of the above models for improved performance

## GPU Optimization

The code is optimized for GPU acceleration using:

- Mixed precision training (FP16) to reduce memory usage and increase speed
- Efficient data loading with chunking to avoid memory bottlenecks
- Parallel data loading with multiple workers
- Batch processing to maximize GPU utilization

## Usage

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hai-dataset-analysis.git
cd hai-dataset-analysis

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Run with default parameters
python main.py

# Run with custom parameters
python main.py --data_dir hai-security-dataset/hai-23.05/ --seq_length 100 --batch_size 128 --epochs 20
```

### Command Line Arguments

- `--data_dir`: Path to dataset directory (default: 'hai-security-dataset/hai-23.05/')
- `--seq_length`: Sequence length for time series (default: 100)
- `--batch_size`: Batch size for training (default: 128)
- `--epochs`: Number of epochs for training (default: 20)

## Results

The analysis produces several outputs:

1. **Dataset Analysis**:
   - Feature distributions
   - Time series plots
   - Correlation heatmaps

2. **Model Training**:
   - Training and validation loss curves
   - Saved model weights

3. **Anomaly Detection**:
   - ROC curves
   - Precision-Recall curves
   - Error distributions
   - Confusion matrices
   - Performance metrics (F1, Precision, Recall, AUC)

4. **Model Comparison**:
   - Comparative analysis of all models
   - Ensemble model performance

## Example Visualizations

### Feature Distributions
![Feature Distributions](plots/dataset_analysis/feature_distributions.png)

### Training History
![Training History](plots/lstm_training_history.png)

### Reconstruction Error
![Reconstruction Error](plots/test1/lstm/error_time_series.png)

### Model Comparison
![Model Comparison](plots/test1/model_comparison.png)

## Performance Considerations

- The code is optimized for A100 80GB GPUs
- For large datasets, adjust batch size and sequence length based on available GPU memory
- Use the chunking mechanism to process datasets larger than available RAM

## References

1. HAI Security Dataset: HIL-based Augmented ICS Security Dataset
2. "HAI 1.0: HIL-based Augmented ICS Security Dataset" by Hyeok-Ki Shin, Woomyo Lee, Jeong-Han Yun and Hyoungchun Kim