import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from config.settings import cfg


def create_sequences(X: np.ndarray, y: np.ndarray, window_size: int):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i + window_size])
        ys.append(y[i + window_size])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_model(model, X_train, y_train, X_val, y_val, window_size: int):
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    patience = cfg["training"]["patience"]
    lr = cfg["training"]["learning_rate"]

    X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, window_size)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, window_size)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq)),
        batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(torch.tensor(X_val_seq))
            val_loss = criterion(val_preds, torch.tensor(y_val_seq)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    return model


def train_gru(model, X_train, y_train, X_val, y_val, window_size: int):
    return train_model(model, X_train, y_train, X_val, y_val, window_size)


def train_lstm(model, X_train, y_train, X_val, y_val, window_size: int):
    return train_model(model, X_train, y_train, X_val, y_val, window_size)
