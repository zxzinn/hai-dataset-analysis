# HAI Security Dataset Analysis

This repository contains a comprehensive analysis of the HIL-based Augmented ICS (HAI) Security Dataset, which contains data collected from a realistic industrial control system (ICS) testbed augmented with a hardware-in-the-loop (HIL) simulator that emulates steam-turbine power generation and pumped-storage hydropower generation.

## Project Overview

The HAI security dataset includes multiple versions (HAI-20.07, HAI-21.03, HAI-22.04, HAI-23.05, HAIEnd-23.05), each containing normal and abnormal behaviors for ICS anomaly detection research. This project focuses on:

1. Preprocessing the dataset for efficient analysis
2. Implementing various anomaly detection models
3. Comparing model performance
4. Providing insights and recommendations for ICS anomaly detection

## Repository Structure

- `preprocessing.ipynb`: Data preprocessing and exploration
- `lstm_model.ipynb`: LSTM autoencoder for anomaly detection
- `random_forest_model.ipynb`: Random Forest classifier for anomaly detection
- `autoencoder_model.ipynb`: Deep autoencoder for anomaly detection
- `model_comparison.ipynb`: Comparison of model performance and ensemble approaches

## Dataset Structure

The HAI dataset is organized into multiple versions:

- **HAI-20.07**: The initial version with semicolon-separated CSV files
- **HAI-21.03**: Updated version with comma-separated CSV files
- **HAI-22.04**: Further updated version with additional features
- **HAI-23.05**: Latest version with enhanced data collection
- **HAIEnd-23.05**: Extended version with additional sensors

Each version contains training files (normal behavior) and test files (including anomalies).

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for LSTM and Autoencoder models)
- Required Python packages:
  ```
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  tensorflow
  cudf (for GPU acceleration)
  cupy (for GPU acceleration)
  dask
  joblib
  ```

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/hai-dataset-analysis.git
   cd hai-dataset-analysis
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the HAI dataset and place it in the `hai-security-dataset` directory.

### Running the Notebooks

Execute the notebooks in the following order:

1. `preprocessing.ipynb`: Preprocess the dataset and create processed data files
2. Model notebooks (can be run in parallel):
   - `lstm_model.ipynb`: Train and evaluate LSTM autoencoder
   - `random_forest_model.ipynb`: Train and evaluate Random Forest
   - `autoencoder_model.ipynb`: Train and evaluate deep autoencoder
3. `model_comparison.ipynb`: Compare model performance and create ensemble

## Model Descriptions

### LSTM Autoencoder

- **Approach**: Uses a sequence-to-sequence LSTM autoencoder to learn temporal patterns in normal data
- **Anomaly Detection**: Measures reconstruction error to identify anomalies
- **Strengths**: Captures temporal dependencies, effective for sequential anomalies
- **File**: `lstm_model.ipynb`

### Random Forest

- **Approach**: Uses a Random Forest classifier with feature selection
- **Anomaly Detection**: Measures anomaly probability based on learned patterns
- **Strengths**: Fast prediction, interpretable feature importance, robust to noise
- **File**: `random_forest_model.ipynb`

### Deep Autoencoder

- **Approach**: Uses a deep neural network autoencoder to compress and reconstruct data
- **Anomaly Detection**: Measures reconstruction error to identify anomalies
- **Strengths**: Captures complex non-linear relationships, effective dimensionality reduction
- **File**: `autoencoder_model.ipynb`

### Ensemble Approach

- **Approach**: Combines predictions from multiple models
- **Methods**: Majority voting and average of normalized anomaly scores
- **Strengths**: More robust detection, reduces false positives/negatives
- **File**: `model_comparison.ipynb`

## Results and Findings

The detailed results and findings are available in the `model_comparison.ipynb` notebook. Key insights include:

- Performance comparison of different models on the HAI dataset
- Analysis of model strengths and weaknesses for ICS anomaly detection
- Recommendations for model selection based on specific requirements
- Ensemble approach evaluation

## Future Work

- Extend analysis to other HAI dataset versions
- Implement more sophisticated ensemble methods
- Develop online learning approaches for adapting to evolving normal behavior
- Investigate explainable AI techniques for better understanding of detected anomalies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The HAI dataset creators for providing this valuable resource for ICS security research
- The open-source community for the tools and libraries used in this project