"""
This module is used for training and predicting using a Naive Bayes Classifier.
It includes functionality for preprocessing data, performing grid search for hyperparameter tuning,
k-fold cross-validation, and feature scaling.
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib

model_path = os.path.join(os.getcwd(), "naive_bayes_model.pkl")
ALPHA = 1


def main():
    """
    This handles the user input for the operation mode and performs the corresponding operations.
    If the mode is 'train', it reads the training dataset, preprocesses it, performs grid search
    for hyperparameter tuning, performs k-fold cross-validation, and saves the trained model to
    disk. If the mode is 'predict', it reads the dataset for prediction, preprocesses it, loads
    the trained model from disk, makes predictions, calculates accuracy if possible, and saves
    the updated dataset with predictions.
    """
    # User input for operation mode
    mode = input("Enter mode (train/predict): ").strip().lower()

    if mode == "train":
        csv_path = input("Enter path to training dataset: ")
        data_frame = pd.read_csv(csv_path)
        preprocess(data_frame)
        features, labels = (
            data_frame.drop("IsMalicious", axis=1).values,
            data_frame["IsMalicious"].values,
        )
        features = scale_features(features)  # Apply feature scaling

        best_alpha = grid_search(features, labels, {"alpha": [0.1, 0.5, 1.0, 2.0]})
        print(f"Best alpha found: {best_alpha}")

        model = NaiveBayesClassifier(alpha=best_alpha)
        accuracy = k_fold_cross_validation(model, features, labels, k=10)
        print(f"10-fold cross-validation accuracy: {accuracy:.2f}")

        with open(model_path, "wb") as file:
            joblib.dump(model, file)
        print(f"Model saved to {model_path}")

    elif mode == "predict":
        csv_path = input("Enter path to dataset for prediction: ")
        data_frame = pd.read_csv(csv_path)
        preprocess(data_frame)
        features = data_frame.drop(
            ["IsMalicious", "Prediction"], axis=1, errors="ignore"
        ).values
        features = scale_features(features)  # Apply feature scaling

        with open(model_path, "rb") as file:
            model = joblib.load(file)
        print(f"Model loaded from {model_path}")

        predictions = model.predict(features)
        data_frame["Prediction"] = predictions
        data_frame["Prediction"] = data_frame["Prediction"].map({1: "Yes", 0: "No"})

        if "IsMalicious" in data_frame.columns:
            true_labels = data_frame["IsMalicious"].values
            predicted_labels = data_frame["Prediction"].map({"Yes": 1, "No": 0}).values
            accuracy = np.mean(predicted_labels == true_labels)
            print(f"Total accuracy on the prediction dataset: {accuracy:.2f}")

        output_path = csv_path.replace(".csv", "_with_predictions.csv")
        data_frame.to_csv(output_path, index=False)
        print(f"Updated dataset saved to {output_path}")


def preprocess(df):
    """
    Preprocesses the input DataFrame for use in the Naive Bayes Classifier.

    Operations:
    1. Converts 'Timestamp' to a usable numeric format.
    2. Encodes categorical variables in 'UserID', 'ApplicationID', 'EventType',
       'Outcome', 'Reason', and 'IPAddress'.
    3. If present, binary encodes the 'IsMalicious' column.

    Parameters:
    df (pandas.DataFrame): The input DataFrame to be preprocessed.

    Returns:
    None. Modifies the input DataFrame in-place.
    """
    # Convert Timestamp to a usable numeric format
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).astype("int64") // 10**9
    # Encode categorical variables
    for col in [
        "UserID",
        "ApplicationID",
        "EventType",
        "Outcome",
        "Reason",
        "IPAddress",
    ]:
        df[col] = df[col].astype("category").cat.codes
    # Binary encode the IsMalicious column if present
    if "IsMalicious" in df.columns:
        df["IsMalicious"] = df["IsMalicious"].str.strip().str.lower()
        df["IsMalicious"] = df["IsMalicious"].map({"yes": 1, "no": 0})


class NaiveBayesClassifier:
    """
    This class represents a Naive Bayes Classifier for binary classification.

    The classifier uses Laplace smoothing and assumes that the features are normally distributed.

    Attributes:
    alpha (float): The Laplace smoothing parameter.
    class_freqs (numpy.ndarray): The frequencies of the classes in the training set.
    class_probs (numpy.ndarray): The probabilities of the classes in the training set.
    cond_probs (dict): The conditional probabilities of the features given a class.
    classes (numpy.ndarray): The unique classes in the training set.
    """

    def __init__(self, alpha=1):
        """
        Initializes a new instance of the NaiveBayesClassifier class.

        Parameters:
        alpha (float): The Laplace smoothing parameter.
        """
        self.alpha = alpha
        self.class_freqs = None
        self.class_probs = None
        self.cond_probs = None
        self.classes = None

    def fit(self, features, labels):
        """
        Fits the Naive Bayes Classifier on the training data.

        Parameters:
        features (numpy.ndarray): A 2D array where each row is a data point and
                           each column is a feature.
        labels (numpy.ndarray): A 1D array of class labels for the data points.
        """
        n_samples = features.shape[0]
        self.classes = np.unique(labels)
        n_classes = len(self.classes)

        # Initialize probability and counts
        self.class_freqs = np.zeros(n_classes)
        self.class_probs = np.zeros(n_classes)
        self.cond_probs = defaultdict(list)

        # Calculate frequencies and probabilities for each class
        for c in self.classes:
            features_c = features[labels == c]
            self.class_freqs[c] = features_c.shape[0]
            self.class_probs[c] = self.class_freqs[c] / n_samples
            # Calculate conditional probabilities using a Gaussian distribution
            self.cond_probs[c] = [
                (np.mean(feature), np.std(feature)) if len(feature) > 0 else (0, 1)
                for feature in zip(*features_c)
            ]

    def predict(self, features):
        """
        Predicts the class labels for the given data points.

        Parameters:
        features (numpy.ndarray): A 2D numpy array where each row represents a data
                                  point and each column represents a feature.

        Returns:
        numpy.ndarray: A 1D numpy array of predicted class labels for the given data
                       points.
        """
        y_pred = [self._predict(x) for x in features]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Predicts the class label for a single data point.

        Parameters:
        x (numpy.ndarray): A 1D numpy array representing a single data point.

        Returns:
        int: The predicted class label for the given data point.
        """
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.class_probs[idx])
            conditional = np.sum(
                np.log(self._calculate_cond_prob(c, x) + 1e-9)
            )  # Small value to prevent division by zero
            posterior = prior + conditional
            posteriors.append(posterior)

        # Return the class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def _calculate_cond_prob(self, class_idx, x):
        """
        Calculates the conditional probability of a data point given a class.

        Parameters:
        class_idx (int): The index of the class.
        x (numpy.ndarray): A 1D numpy array representing a single data point.

        Returns:
        float: The conditional probability of the data point given the class.
        """
        cond_probs = 1
        for i, (mean, std) in enumerate(self.cond_probs[class_idx]):
            # Use Gaussian distribution formula with epsilon added to std
            epsilon = 1e-9  # Small value to prevent division by zero
            std += epsilon
            prob = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(
                -((x[i] - mean) ** 2 / (2 * std**2))
            )
            cond_probs *= prob
        return cond_probs


def grid_search(features, labels, params):
    """
    Performs grid search to find the best hyperparameters for the Naive Bayes Classifier.

    The function iterates over the provided hyperparameters, trains the model using
    each set of hyperparameters, and calculates the accuracy of the model using
    k-fold cross-validation. The function returns the set of hyperparameters that
    resulted in the highest accuracy.

    Parameters:
    features (numpy.ndarray): A 2D array where each row is a data point and each
                              column is a feature.
    labels (numpy.ndarray): A 1D array of class labels for the data points.
    params (dict): A dictionary where the key is the name of the hyperparameter and
                   the value is a list of values to try.

    Returns:
    best_alpha (float): The alpha value that resulted in the highest accuracy during
                        cross-validation.
    """
    best_accuracy = 0
    best_alpha = None
    for alpha in params["alpha"]:
        model = NaiveBayesClassifier(alpha=alpha)
        accuracy = k_fold_cross_validation(model, features, labels, k=10)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha
    return best_alpha


# pylint: disable=too-many-locals
def k_fold_cross_validation(model, features, labels, k=10):
    """
    Performs k-fold cross validation on the given model and data.

    The function splits the data into k folds, and for each fold, it trains the model on the
    remaining data and tests it on the current fold. It then calculates the accuracy of the
    model for each fold and returns the average accuracy.

    Parameters:
    model (object): The machine learning model to be evaluated.
    features (numpy.ndarray): A 2D array where each row is a data point and each
                              column is a feature.
    labels (numpy.ndarray): A 1D array of class labels for the data points.
    k (int, optional): The number of folds for cross-validation. Defaults to 10.

    Returns:
    float: The average accuracy of the model across the k folds.
    """
    fold_size = len(features) // k
    accuracies = []
    all_predictions = []
    all_labels = []

    for fold in range(k):
        # Generate indices for train/test split
        test_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
        train_indices = [i for i in range(len(features)) if i not in test_indices]

        # Split the dataset
        features_train = features[train_indices]
        labels_train = labels[train_indices]
        features_test = features[test_indices]
        labels_test = labels[test_indices]

        # Scale features
        features_train = scale_features(features_train)
        features_test = scale_features(features_test)

        # Train the model and predict on the test set
        model.fit(features_train, labels_train)
        predictions = model.predict(features_test)
        # Calculate and store accuracy
        accuracies.append(np.mean(predictions == labels_test))

        # Store predictions and their correctness
        all_predictions.extend(predictions)
        all_labels.extend(labels_test)

    # Save accuracies to a CSV file
    accuracy_df = pd.DataFrame(accuracies, columns=["Accuracy"])
    accuracy_df.to_csv("k_fold_cross_validation_accuracies.csv", index=False)

    # Save predictions and their correctness to a CSV file
    predictions_df = pd.DataFrame(
        {
            "Prediction": all_predictions,
            "Actual": all_labels,
            "IsCorrect": np.array(all_predictions) == np.array(all_labels),
        }
    )
    predictions_df.to_csv("k_fold_cross_validation_predictions.csv", index=False)

    return np.mean(accuracies)


def scale_features(feature_matrix):
    """
    Scales the features in the input matrix using standardization.

    The function calculates the mean and standard deviation for each feature
    (column) in the input matrix, and then subtracts the mean and divides by
    the standard deviation for each value.

    Parameters:
    feature_matrix (numpy.ndarray): A 2D array where each row is a data point
    and each column is a feature.

    Returns:
    numpy.ndarray: A 2D array of the same shape as the input, but with the
    features scaled.
    """
    means = np.mean(feature_matrix, axis=0)
    stds = np.std(feature_matrix, axis=0)
    scaled_features = (feature_matrix - means) / stds
    return scaled_features


if __name__ == "__main__":
    main()
