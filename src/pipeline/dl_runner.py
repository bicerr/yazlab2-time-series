import numpy as np
import time
from src.models.lstm_model import build_lstm
from src.models.gru_model import build_gru
from src.models.cnn_model import build_cnn
from src.models.trainer import train_model
from src.models.predictor import predict
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.logger import log_experiment
from config.settings import cfg


def run_dl_model(model_name: str, data: tuple, dataset: str, seed: int, scenario: str) -> dict:
    """
    Tek bir DL modelini verilen veri üzerinde eğitip değerlendirir.
    data: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr, y_tr, X_val, y_val, X_te, y_te = data
    input_size = X_tr.shape[1]
    window_size = cfg["automata"]["window_size"]

    if model_name == "LSTM":
        model = build_lstm(input_size)
    elif model_name == "GRU":
        model = build_gru(input_size)
    elif model_name == "CNN":
        model = build_cnn(input_size)
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    # Eğitim süresi ölçümü
    t_start = time.time()
    model = train_model(model, X_tr, y_tr, X_val, y_val, window_size=window_size)
    train_time = time.time() - t_start

    # Inference süresi ölçümü
    t_start = time.time()
    preds, y_true, probs = predict(model, X_te, y_te, window_size=window_size)
    inference_time = time.time() - t_start

    metrics = compute_all_metrics(y_true, preds)
    metrics["train_time"] = round(train_time, 4)
    metrics["inference_time"] = round(inference_time, 4)

    log_experiment(model_name, dataset, seed, scenario, metrics)

    return {"model": model_name, "metrics": metrics, "preds": preds, "y_true": y_true, "probs": probs}


def run_all_dl_models(data: tuple, dataset: str, seed: int, scenario: str) -> dict:
    results = {}
    for model_name in ["LSTM", "GRU", "CNN"]:
        results[model_name] = run_dl_model(model_name, data, dataset, seed, scenario)
    return results
