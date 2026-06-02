import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return precision_score(y_true, y_pred, zero_division=0)


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return recall_score(y_true, y_pred, zero_division=0)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, zero_division=0)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "precision": compute_precision(y_true, y_pred),
        "recall": compute_recall(y_true, y_pred),
        "f1": compute_f1(y_true, y_pred),
    }
