import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNWatermark(nn.Module):
    def __init__(self):
        super(CNNWatermark, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, image, watermark):
        watermarked = self.encoder(image) + watermark
        extracted = self.decoder(watermarked)
        return watermarked, extracted
