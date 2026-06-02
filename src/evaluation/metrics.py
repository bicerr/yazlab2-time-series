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


def compute_metrics_macro(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def compute_metrics_micro(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }


def compute_metrics_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
    }


def compare_models(results: dict) -> dict:
    """
    results: {model_name: {"y_true": ..., "y_pred": ...}}
    Her model için temel metrikleri hesaplar ve karşılaştırma tablosu döndürür.
    """
    report = {}
    for model_name, data in results.items():
        y_true = np.array(data["y_true"])
        y_pred = np.array(data["y_pred"])
        metrics = compute_all_metrics(y_true, y_pred)
        metrics.update(compute_metrics_macro(y_true, y_pred))
        report[model_name] = metrics
    return report


def print_report(report: dict):
    header = f"{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'F1-Macro':>10}"
    print(header)
    print("-" * len(header))
    for model, m in report.items():
        print(
            f"{model:<15} "
            f"{m['accuracy']:>10.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} "
            f"{m['f1_macro']:>10.4f}"
        )
