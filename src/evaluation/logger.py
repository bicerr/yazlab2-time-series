import json
import os
from datetime import datetime
from config.settings import cfg


def create_log_entry(model_name: str, dataset: str, seed: int, scenario: str, metrics: dict) -> dict:
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "dataset": dataset,
        "seed": seed,
        "scenario": scenario,
        "metrics": metrics,
    }


def save_log(entry: dict, log_dir: str = None):
    if log_dir is None:
        log_dir = cfg["logging"]["log_dir"]

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "experiment_log.json")

    logs = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)

    logs.append(entry)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


def load_logs(log_dir: str = None) -> list:
    if log_dir is None:
        log_dir = cfg["logging"]["log_dir"]

    log_file = os.path.join(log_dir, "experiment_log.json")

    if not os.path.exists(log_file):
        return []

    with open(log_file, "r") as f:
        return json.load(f)


def log_experiment(model_name: str, dataset: str, seed: int, scenario: str, metrics: dict, log_dir: str = None):
    entry = create_log_entry(model_name, dataset, seed, scenario, metrics)
    save_log(entry, log_dir)
    return entry


def filter_logs(logs: list, model: str = None, dataset: str = None, scenario: str = None) -> list:
    filtered = logs
    if model:
        filtered = [l for l in filtered if l["model"] == model]
    if dataset:
        filtered = [l for l in filtered if l["dataset"] == dataset]
    if scenario:
        filtered = [l for l in filtered if l["scenario"] == scenario]
    return filtered


def summarize_logs(logs: list) -> dict:
    """Seed bazlı sonuçları gruplar ve ortalama/std hesaplar."""
    import numpy as np
    from collections import defaultdict

    grouped = defaultdict(list)
    for log in logs:
        key = (log["model"], log["dataset"], log["scenario"])
        grouped[key].append(log["metrics"].get("f1", 0.0))

    summary = {}
    for (model, dataset, scenario), f1_scores in grouped.items():
        summary[f"{model}_{dataset}_{scenario}"] = {
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
            "n_seeds": len(f1_scores),
        }
    return summary
