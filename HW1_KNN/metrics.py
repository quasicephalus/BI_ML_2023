import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    n = len(y_pred)
    fp = np.count_nonzero(np.logical_and(y_pred == 1, y_true == 0))
    fn = np.count_nonzero(np.logical_and(y_pred == 0, y_true == 1))
    tp = np.count_nonzero(np.logical_and(y_pred == 1, y_true == 1))
    tn = np.count_nonzero(np.logical_and(y_pred == 0, y_true == 0))

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fp > 0 else 0
    f1 = 2*(precision * recall) / (precision + recall)
    accuracy = (tp + tn)/n
    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    n = len(y_pred)
    true_values = np.count_nonzero(y_pred == y_true)
    
    return true_values / n


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    SSres = np.sum(np.power((y_true-y_pred),2))
    SStot = np.sum(np.power((y_true-y_true.mean()),2))
    return 1 - (SSres/SStot)

def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    return (1/len(y_true))*np.sum(np.power((y_true-y_pred),2))


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    return (1/len(y_true))*np.sum(np.abs((y_true-y_pred)))
    
