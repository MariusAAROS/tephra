from flaml import AutoML
import pandas as pd
import numpy as np

from joblib import dump

basedir = r"data/"
savedir = r"params/"

train = pd.read_csv(basedir + 'train_preprocessed.csv')
test = pd.read_csv(basedir + 'test_preprocessed.csv')
groups = pd.read_csv(basedir + 'train.csv')["SampleID"].astype("category").cat.codes.to_numpy()

X_train = train.drop(["Event"], axis=1)
y_train = train["Event"]

X_test = test.drop(["Event"], axis=1)
y_test = test["Event"]

def balanced_accuracy(
    X_val, y_val, estimator, labels,
    X_train, y_train, weight_val=None, weight_train=None,
    config=None, groups_val=None, groups_train=None,
):
    """
    Compute balanced accuracy as a custom metric.

    Parameters:
    X_val : array-like of shape (n_samples, n_features)
        Validation data.

    y_val : array-like of shape (n_samples,)
        Ground truth (correct) target values for validation data.

    estimator : object
        Fitted estimator.

    labels : array-like of shape (n_classes,)
        Unique labels in true target values.

    X_train : array-like of shape (n_samples, n_features)
        Training data.

    y_train : array-like of shape (n_samples,)
        Ground truth (correct) target values for training data.

    weight_val : array-like of shape (n_samples,), default=None
        Sample weights for validation data.

    weight_train : array-like of shape (n_samples,), default=None
        Sample weights for training data.

    config : dict or None, default=None
        Configuration dictionary.

    groups_val : array-like of shape (n_samples,), default=None
        Group labels for validation data used for grouped cross-validation.

    groups_train : array-like of shape (n_samples,), default=None
        Group labels for training data used for grouped cross-validation.

    Returns:
    balanced_accuracy : float
        Balanced accuracy score.
    """
    unique_classes = np.unique(y_val)
    class_weights = {cls: 1 / len(unique_classes) for cls in unique_classes}
    
    # Initialize counts of true positives and true negatives for each class
    tp_counts = {cls: 0 for cls in unique_classes}
    tn_counts = {cls: 0 for cls in unique_classes}
    
    # Compute true positives and true negatives for each class
    y_pred = estimator.predict(X_val)
    for cls in unique_classes:
        for true, pred in zip(y_val, y_pred):
            if true == cls:
                if pred == cls:
                    tp_counts[cls] += 1
                else:
                    tn_counts[cls] += 1
    
    # Compute balanced accuracy
    balanced_accuracy = 0
    for cls in unique_classes:
        balanced_accuracy += class_weights[cls] * (tp_counts[cls] / (2 * tp_counts[cls] + tn_counts[cls]))
    
    return balanced_accuracy

automl = AutoML()
settings = {
    "time_budget": 10,  # total running time in seconds
    "metric": 'accuracy',  # primary metric
    "task": 'classification',
    "estimator_list": ['catboost'],  # list of ML learners
    "log_file_name": "logs/automl_cb.log",
}

automl.fit(X_train, y_train, **settings)

# Export best parameters for each model
dump(automl.model.estimator, savedir + "cat_best_estimator.joblib")