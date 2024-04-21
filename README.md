
# Naive Bayes Classifier for IAM Anomaly Detection

This repository contains code for training and predicting anomalies in Identity and Access Management (IAM) using a Naive Bayes Classifier. It includes scripts for data preprocessing, hyperparameter tuning, k-fold cross-validation, and feature scaling. Additionally, it provides a module for simulating Single Sign-On (SSO) events to generate datasets for training and testing.

## Getting Started

### Prerequisites

- Python 3.6+
- pandas
- numpy
- joblib
- Ensure you have Python installed on your system, and the required libraries can be installed using pip:

```bash
pip install pandas numpy joblib
```

### Dataset Generation

The `simulate_sso_events.py` script generates synthetic data that simulates SSO events based on predefined patterns and randomness to reflect typical and anomalous user behaviors. This script needs to be run first to generate the data needed for training and testing the Naive Bayes classifier.

To generate the dataset, run:

```bash
python simulate_sso_events.py
```

This will create a file named `sso_events.csv` in the current directory, containing the simulated SSO logs.

### Configuration Variables

- `NUM_RECORDS`: Defines the total number of records to generate.
- `START_DATE` and `END_DATE`: Define the period over which the events are simulated.
- `USERS` and `APPLICATIONS`: Lists that define the range of users and applications to simulate activity for.

### Using the Classifier

The main script for the classifier is `naive_bayes_classifier.py`. This script can operate in two modes:
1. **Training Mode**: Reads the training dataset, preprocesses the data, performs hyperparameter tuning, executes k-fold cross-validation, and saves the trained model.
2. **Prediction Mode**: Reads a new dataset for prediction, preprocesses the data, loads the trained model, and performs predictions, saving the output with predictions appended.

#### To Train the Model

Run the script in training mode and provide the path to your training dataset:

```bash
python naive_bayes_classifier.py
Enter mode (train/predict): train
Enter path to training dataset: path/to/your/training_dataset.csv
```

#### To Predict Using the Model

Run the script in prediction mode and provide the path to your dataset:

```bash
python naive_bayes_classifier.py
Enter mode (train/predict): predict
Enter path to dataset for prediction: path/to/your/prediction_dataset.csv
```

### Adjustable Variables

- `ALPHA`: Smoothing parameter in the Naive Bayes classifier, used to handle zero frequency problems.
- `k` in `k_fold_cross_validation`: Number of folds used in the cross-validation process.

## Output

The scripts will output:
- The trained model saved as `naive_bayes_model.pkl`
- Updated dataset with predictions saved with a suffix `_with_predictions.csv`
- Accuracy metrics printed to the console during training and prediction phases.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Authors

- Justin Rudisal

