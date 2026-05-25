import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class R3DGesture(nn.Module):
    def __init__(self, num_classes=8, dropout=0.3):
        super().__init__()
        backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        in_features = backbone.fc.in_features   # 512
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        # x: (B, C, T, H, W)
        return self.model(x)
