import pandas as pd
from pathlib import Path
from config.settings import cfg

ROOT = Path(__file__).parent.parent.parent


def load_skab() -> pd.DataFrame:
    skab_path = ROOT / cfg["data"]["skab_path"]
    groups = ["valve1", "valve2"]
    frames = []

    for group in groups:
        group_path = skab_path / group
        for csv_file in sorted(group_path.glob("*.csv")):
            df = pd.read_csv(csv_file, sep=";", parse_dates=["datetime"])
            df["source_group"] = group
            df["source_file"] = csv_file.name
            frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["source_group", "source_file", "datetime"]).reset_index(drop=True)
    return data


def load_batadal() -> pd.DataFrame:
    batadal_path = ROOT / cfg["data"]["batadal_path"]
    csv_files = list(batadal_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"BATADAL CSV bulunamadı: {batadal_path}")

    df = pd.read_csv(csv_files[0], sep=",", skipinitialspace=True)
    df.columns = df.columns.str.strip()

    df["ATT_FLAG"] = df["ATT_FLAG"].apply(lambda x: 0 if x == -999 else 1)
    return df


def get_skab_features_target(df: pd.DataFrame):
    drop_cols = ["datetime", "changepoint", "source_group", "source_file", "anomaly"]
    X = df.drop(columns=drop_cols)
    y = df["anomaly"].astype(int)
    groups = df["source_file"]
    return X, y, groups


def get_batadal_features_target(df: pd.DataFrame):
    drop_cols = ["DATETIME", "ATT_FLAG"]
    X = df.drop(columns=drop_cols)
    y = df["ATT_FLAG"].astype(int)
    return X, y
