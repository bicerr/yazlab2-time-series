import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
