import torch
import torch.nn as nn
from config.settings import cfg


class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])  # son zaman adımı
        out = self.fc(out)
        return self.sigmoid(out).squeeze(1)


def build_gru(input_size: int) -> GRUModel:
    return GRUModel(input_size=input_size)
