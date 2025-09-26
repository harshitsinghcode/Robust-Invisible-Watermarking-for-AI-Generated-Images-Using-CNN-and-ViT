import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16

class ViTWatermark(nn.Module):
    def __init__(self):
        super(ViTWatermark, self).__init__()
        self.vit = vit_b_16(weights=None)  # Corrected 'pretrained' to 'weights'
        self.vit.heads = nn.Identity()  # Remove classification layer, keep raw features
        self.linear = nn.Linear(768, 256)  # Map ViT features to 256

    def forward(self, x):
        features = self.vit(x)  # Output: [batch_size, 768]
        return self.linear(features)  # Output: [batch_size, 256]

if __name__ == "__main__":
    model = ViTWatermark()
    print(model)
