import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, input_size: int, num_filters: int = 64, kernel_size: int = 3, dropout: float = 0.2):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, input_size) → (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(2)
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x).squeeze(1)


def build_cnn(input_size: int) -> CNNModel:
    return CNNModel(input_size=input_size)
