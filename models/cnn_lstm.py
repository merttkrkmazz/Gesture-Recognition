import torch
import torch.nn as nn
from torchvision import models


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=8, hidden_dim=256,
                 num_layers=2, dropout=0.3, freeze_cnn=False):
        super().__init__()

        # Feature extractor: MobileNetV2 (hafif ve hızlı)
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.feature_dim = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        self.cnn = backbone

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # Her frame'i CNN'den geçir
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)          # (B*T, feature_dim)
        features = features.view(B, T, -1)  # (B, T, feature_dim)

        out, _ = self.lstm(features)
        out = out[:, -1, :]             # son timestep
        return self.classifier(out)
