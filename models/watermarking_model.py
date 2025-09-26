import torch
import torch.nn as nn
from .cnn_model import CNNWatermark
from .vit_model import ViTWatermark

class WatermarkingModel(nn.Module):
    def __init__(self):
        super(WatermarkingModel, self).__init__()
        self.cnn = CNNWatermark()
        self.vit = ViTWatermark()

        self.fc = nn.Linear(512, 150528)
        self.upsample = nn.Unflatten(1, (3, 224, 224))

        # Add a second decoder for multi-level tampering detection
        self.decoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, original, tampered):
        cnn_features = self.cnn(original)[1]
        vit_features = self.vit(original)
        combined = torch.cat((cnn_features, vit_features), dim=1)

        # Watermark extracted at stage 1 (after social media compression)
        extracted1 = self.decoder1(original)

        # Watermark extracted at stage 2 (after severe tampering)
        extracted2 = self.decoder2(tampered)

        return extracted1, extracted2
