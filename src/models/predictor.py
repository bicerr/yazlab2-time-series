import torch
import numpy as np
from src.models.trainer import create_sequences


def predict(model, X_test: np.ndarray, y_test: np.ndarray, window_size: int, threshold: float = 0.5):
    X_seq, y_seq = create_sequences(X_test, y_test, window_size)

    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X_seq)).numpy()

    preds = (probs >= threshold).astype(int)
    return preds, y_seq.astype(int), probs
