import numpy as np
from src.evaluation.logger import load_logs


def summarize_runtime(logs: list = None) -> dict:
    """
    Log kayıtlarından model bazlı ortalama training ve inference sürelerini hesaplar.
    """
    if logs is None:
        logs = load_logs()

    runtime = {}
    for log in logs:
        model = log["model"]
        metrics = log.get("metrics", {})
        train_time = metrics.get("train_time", None)
        inference_time = metrics.get("inference_time", None)

        if model not in runtime:
            runtime[model] = {"train_times": [], "inference_times": []}

        if train_time is not None:
            runtime[model]["train_times"].append(train_time)
        if inference_time is not None:
            runtime[model]["inference_times"].append(inference_time)

    summary = {}
    for model, times in runtime.items():
        summary[model] = {
            "mean_train_time": round(float(np.mean(times["train_times"])), 4) if times["train_times"] else None,
            "mean_inference_time": round(float(np.mean(times["inference_times"])), 4) if times["inference_times"] else None,
        }

    return summary


def print_runtime_table(summary: dict):
    print(f"\n{'Model':<12} {'Train (sn)':>12} {'Inference (sn)':>16}")
    print("-" * 42)
    for model, times in summary.items():
        tr = f"{times['mean_train_time']:.4f}" if times["mean_train_time"] is not None else "N/A"
        inf = f"{times['mean_inference_time']:.4f}" if times["mean_inference_time"] is not None else "N/A"
        print(f"{model:<12} {tr:>12} {inf:>16}")
