import numpy as np
from src.preprocessing.loader import (
    load_skab, load_batadal,
    get_skab_features_target, get_batadal_features_target
)
from src.preprocessing.preprocessor import (
    fit_transform_scaler, fit_transform_pca,
    split_skab, split_batadal
)
from src.pipeline.dl_runner import run_dl_model
from src.pipeline.automata_runner import run_automata
from src.evaluation.logger import log_experiment
from config.settings import cfg


def prepare_full_train_test(dataset: str):
    """
    Verilen veri setini train ve test olarak hazırlar.
    Cross-dataset deneyi için tüm train verisi kullanılır.
    """
    if dataset == "SKAB":
        df = load_skab()
        X, y, groups = get_skab_features_target(df)
        folds = split_skab(X.values, y.values, groups.values)
        # İlk fold'un train kısmını al
        train_idx, _, test_idx = folds[0]
        X_tr, y_tr = X.values[train_idx], y.values[train_idx]
        X_te, y_te = X.values[test_idx], y.values[test_idx]

    elif dataset == "BATADAL":
        df = load_batadal()
        X, y = get_batadal_features_target(df)
        (X_tr, y_tr), _, (X_te, y_te) = split_batadal(X.values, y.values)

    else:
        raise ValueError(f"Bilinmeyen dataset: {dataset}")

    return X_tr, y_tr, X_te, y_te


def run_cross_dataset(train_dataset: str, test_dataset: str, seed: int = 42) -> dict:
    """
    Bir veri setinde eğitip diğerinde test eder.
    Scaler ve PCA sadece train verisine fit edilir.
    """
    X_tr_raw, y_tr, X_te_raw, y_te = prepare_full_train_test(train_dataset)
    X_tr_test_raw, _, X_te_test_raw, y_te_test = prepare_full_train_test(test_dataset)

    # Boyut uyumu için her iki seti de ayrı scaler ile ölçekle
    X_tr_s, _, X_te_s, scaler = fit_transform_scaler(X_tr_raw, X_test=X_te_test_raw)

    # DL için PCA
    X_tr_dl, _, X_te_dl, _ = fit_transform_pca(X_tr_s, X_test=X_te_s, mode="dl")

    # Otomata için PCA (PC1)
    X_tr_aut, _, X_te_aut, _ = fit_transform_pca(X_tr_s, X_test=X_te_s, mode="automata")

    results = {}
    scenario = f"cross_{train_dataset}_to_{test_dataset}"

    # DL modelleri
    for model_name in ["LSTM", "GRU", "CNN"]:
        dl_data = (X_tr_dl, y_tr, X_tr_dl, y_tr, X_te_dl, y_te_test)
        result = run_dl_model(model_name, dl_data, test_dataset, seed, scenario)
        results[model_name] = result["metrics"]
        print(f"  {model_name} ({train_dataset}→{test_dataset}): F1={result['metrics']['f1']:.4f}")

    # Otomata
    aut_data = (X_tr_aut, y_tr, X_tr_aut, y_tr, X_te_aut, y_te_test)
    result_aut = run_automata(aut_data, test_dataset, seed, scenario)
    results["Automata"] = result_aut["metrics"]
    print(f"  Automata ({train_dataset}→{test_dataset}): F1={result_aut['metrics']['f1']:.4f}")

    return results


def run_all_cross_dataset(seed: int = 42) -> dict:
    """
    SKAB→BATADAL ve BATADAL→SKAB cross-dataset deneyleri.
    """
    print("\n=== Cross-Dataset Deneyleri ===")
    results = {}

    print("\nSKAB → BATADAL:")
    results["SKAB_to_BATADAL"] = run_cross_dataset("SKAB", "BATADAL", seed)

    print("\nBATADAL → SKAB:")
    results["BATADAL_to_SKAB"] = run_cross_dataset("BATADAL", "SKAB", seed)

    return results
