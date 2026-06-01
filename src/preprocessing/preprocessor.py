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


def fit_transform_pca(X_train, X_val=None, X_test=None):
    n_components = cfg["preprocessing"]["pca_components"] or 0.95
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

    kf = GroupKFold(n_splits=5)
    folds = []
    for train_idx, test_idx in kf.split(X, y, groups):
        mid = len(test_idx) // 2
        val_idx = test_idx[:mid]
        test_idx_final = test_idx[mid:]
        folds.append((train_idx, val_idx, test_idx_final))
    return folds
