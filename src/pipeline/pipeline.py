import numpy as np
from src.preprocessing.loader import (
    load_skab, load_batadal,
    get_skab_features_target, get_batadal_features_target
)
from src.preprocessing.preprocessor import (
    fit_transform_scaler, fit_transform_pca,
    split_skab, split_batadal
)
from config.settings import cfg


def prepare_skab():
    """SKAB verisini yükler, böler, ölçekler ve PCA uygular."""
    df = load_skab()
    X, y, groups = get_skab_features_target(df)
    folds = split_skab(X.values, y.values, groups.values)

    prepared = []
    for train_idx, val_idx, test_idx in folds:
        X_tr, X_val, X_te = X.values[train_idx], X.values[val_idx], X.values[test_idx]
        y_tr, y_val, y_te = y.values[train_idx], y.values[val_idx], y.values[test_idx]

        X_tr_s, X_val_s, X_te_s, scaler = fit_transform_scaler(X_tr, X_val, X_te)

        # DL için PCA
        X_tr_dl, X_val_dl, X_te_dl, pca_dl = fit_transform_pca(
            X_tr_s, X_val_s, X_te_s, mode="dl"
        )
        # Otomata için PCA (PC1)
        X_tr_aut, X_val_aut, X_te_aut, pca_aut = fit_transform_pca(
            X_tr_s, X_val_s, X_te_s, mode="automata"
        )

        prepared.append({
            "dl": (X_tr_dl, y_tr, X_val_dl, y_val, X_te_dl, y_te),
            "automata": (X_tr_aut, y_tr, X_val_aut, y_val, X_te_aut, y_te),
            "scaler": scaler,
            "pca_dl": pca_dl,
            "pca_aut": pca_aut,
        })

    return prepared


def prepare_batadal():
    """BATADAL verisini yükler, böler, ölçekler ve PCA uygular."""
    df = load_batadal()
    X, y = get_batadal_features_target(df)

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_batadal(X.values, y.values)

    X_tr_s, X_val_s, X_te_s, scaler = fit_transform_scaler(X_tr, X_val, X_te)

    # DL için PCA
    X_tr_dl, X_val_dl, X_te_dl, pca_dl = fit_transform_pca(
        X_tr_s, X_val_s, X_te_s, mode="dl"
    )
    # Otomata için PCA (PC1)
    X_tr_aut, X_val_aut, X_te_aut, pca_aut = fit_transform_pca(
        X_tr_s, X_val_s, X_te_s, mode="automata"
    )

    return {
        "dl": (X_tr_dl, y_tr, X_val_dl, y_val, X_te_dl, y_te),
        "automata": (X_tr_aut, y_tr, X_val_aut, y_val, X_te_aut, y_te),
        "scaler": scaler,
        "pca_dl": pca_dl,
        "pca_aut": pca_aut,
    }


if __name__ == "__main__":
    from src.pipeline.experiment import run_all_experiments
    run_all_experiments()
