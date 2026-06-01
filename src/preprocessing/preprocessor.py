import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from config.settings import cfg


def get_scaler():
    scaler_type = cfg["preprocessing"]["scaler"]
    if scaler_type == "minmax":
        return MinMaxScaler()
    return StandardScaler()


def fit_transform_scaler(X_train, X_val=None, X_test=None):
    scaler = get_scaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def fit_transform_pca(X_train, X_val=None, X_test=None, mode: str = "dl"):
    """
    mode='dl'       -> DL modelleri: varyansın %95'ini korur
    mode='automata' -> PDF gereği tek boyut (PC1)
    """
    if mode == "automata":
        n_components = cfg["preprocessing"]["pca_components_automata"]
    else:
        n_components = cfg["preprocessing"]["pca_components_dl"] or 0.95

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)

    X_val_pca = pca.transform(X_val) if X_val is not None else None
    X_test_pca = pca.transform(X_test) if X_test is not None else None

    return X_train_pca, X_val_pca, X_test_pca, pca


def split_batadal(X, y):
    ratios = cfg["experiment"]["batadal_split"]
    n = len(X)
    t1 = int(n * ratios[0])
    t2 = int(n * (ratios[0] + ratios[1]))

    return (X[:t1], y[:t1]), (X[t1:t2], y[t1:t2]), (X[t2:], y[t2:])


def split_skab(X, y, groups):
    from sklearn.model_selection import GroupKFold
    import numpy as np

    # 1. split: train+val vs test (farklı gruplar)
    kf_outer = GroupKFold(n_splits=5)
    folds = []
    for trainval_idx, test_idx in kf_outer.split(X, y, groups):
        X_trainval = X[trainval_idx] if hasattr(X, '__getitem__') else X.iloc[trainval_idx]
        y_trainval = y[trainval_idx] if hasattr(y, '__getitem__') else y.iloc[trainval_idx]
        groups_trainval = groups[trainval_idx] if hasattr(groups, '__getitem__') else groups.iloc[trainval_idx]

        # 2. split: train vs val (farklı gruplar)
        kf_inner = GroupKFold(n_splits=4)
        inner_splits = list(kf_inner.split(X_trainval, y_trainval, groups_trainval))
        train_local, val_local = inner_splits[0]

        train_idx = trainval_idx[train_local]
        val_idx = trainval_idx[val_local]

        folds.append((train_idx, val_idx, test_idx))
    return folds
