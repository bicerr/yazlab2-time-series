import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load_config(path: str = None) -> dict:
    config_path = path or ROOT / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


cfg = load_config()
