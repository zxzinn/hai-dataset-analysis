# HAI Dataset Analysis and ResBiLSTM Model

This project provides tools for analyzing HAI (HIL-based Augmented ICS) security datasets and implementing a Residual Bidirectional LSTM (ResBiLSTM) model for anomaly detection.

## Files

- `hai_utils.py`: Utility functions for data loading, preprocessing, model creation, training, and evaluation
- `hai_20_07_analysis.ipynb`: Jupyter notebook for analyzing the HAI-20.07 dataset
- `hai_dataset_analysis_template.ipynb`: Template notebook that can be adapted for any HAI dataset

## Requirements

The following packages are required:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
pyarrow
joblib
```

You can install these packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pyarrow joblib
```

## Dataset Structure

The HAI datasets are organized as follows:

```
hai-security-dataset/
├── hai-20.07/
│   ├── test1.csv
│   ├── test2.csv
│   ├── train1.csv
│   └── train2.csv
├── hai-21.03/
│   ├── test1.csv
│   ├── test2.csv
│   └── ...
└── ...
```

The datasets are converted to Parquet format for more efficient processing:

```
parquet_data/
├── hai-20.07/
│   ├── test1.parquet
│   ├── test2.parquet
│   ├── train1.parquet
│   └── train2.parquet
└── ...
```

## Usage

### Converting CSV to Parquet

If you need to convert CSV files to Parquet format, use the `csv_to_parquet.py` script:

```bash
python csv_to_parquet.py hai-security-dataset/hai-20.07
```

### Analyzing HAI-20.07 Dataset

To analyze the HAI-20.07 dataset:

1. Open the `hai_20_07_analysis.ipynb` notebook in Jupyter
2. Run the cells sequentially to:
   - Load and explore the data
   - Apply feature engineering
   - Create and train the ResBiLSTM model
   - Evaluate model performance
   - Visualize results

### Analyzing Other HAI Datasets

To analyze other HAI datasets:

1. Open the `hai_dataset_analysis_template.ipynb` notebook in Jupyter
2. Modify the configuration parameters at the beginning of the notebook:
   ```python
   DATASET_NAME = 'hai-21.03'  # Change to the dataset you want to analyze
   TRAIN_FILE = 'train2'        # File containing training data with attack labels
   TEST_FILE = 'test2'          # File containing test data with attack labels
   ```
3. Run the cells sequentially as in the HAI-20.07 analysis

## Model Architecture

The ResBiLSTM model architecture includes:

- Input layer
- Multiple Bidirectional LSTM layers with residual connections
- Batch normalization and dropout for regularization
- Dense layers for classification
- Sigmoid activation for binary output

## Feature Engineering

The feature engineering process includes:

1. Feature selection using ANOVA F-value
2. Standardization of features
3. Sequence creation for time series data

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

## Notes on HAI Datasets

- In HAI-20.07, only train2.csv contains attack patterns for training
- Test datasets contain attack patterns for evaluation
- Different HAI versions may have different column structures and naming conventions