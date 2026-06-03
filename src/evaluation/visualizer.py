import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import os

OUTPUT_DIR = "notebooks"


def save_fig(fig, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Kaydedildi: {path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str, dataset: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomali"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name} ({dataset})")
    save_fig(fig, f"confusion_matrix_{model_name}_{dataset}.png")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   model_name: str, dataset: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Eğrisi — {model_name} ({dataset})")
    ax.legend()
    save_fig(fig, f"roc_{model_name}_{dataset}.png")


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray,
                  model_name: str, dataset: str):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Eğrisi — {model_name} ({dataset})")
    ax.legend()
    save_fig(fig, f"pr_{model_name}_{dataset}.png")


def plot_parameter_sensitivity(sweep_results: dict, param_name: str, dataset: str):
    """
    Parametre tarama sonuçlarını çizgi grafik olarak gösterir.
    sweep_results: {param_value: {model: f1_score}}
    """
    param_values = sorted(sweep_results.keys())
    models = list(next(iter(sweep_results.values())).keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        f1_scores = [sweep_results[v].get(model, 0) for v in param_values]
        ax.plot(param_values, f1_scores, marker="o", label=model)

    ax.set_xlabel(param_name)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Parametre Duyarlılık Analizi — {param_name} ({dataset})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, f"sensitivity_{param_name}_{dataset}.png")


