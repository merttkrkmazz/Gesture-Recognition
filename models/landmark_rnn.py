import torch
import torch.nn as nn


class LandmarkBiLSTM(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=256, num_layers=2,
                 num_classes=8, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T, 63)
        out, _ = self.lstm(x)
        # Son timestep'in forward + backward hidden state'ini al
        out = out[:, -1, :]    # (B, hidden_dim*2)
        return self.classifier(out)
