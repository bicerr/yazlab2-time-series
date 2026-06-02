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
