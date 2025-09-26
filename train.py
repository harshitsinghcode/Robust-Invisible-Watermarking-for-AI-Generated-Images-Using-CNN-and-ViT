import torch
import torch.optim as optim
import torch.nn as nn
from models.cnn_model import CNNWatermark
from models.vit_model import ViTWatermark
from utils.data_loader import get_dataloader

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
image_dir = "data/flickr8k/images/"
dataloader = get_dataloader(image_dir)

# Initialize models
cnn = CNNWatermark().to(device)
vit = ViTWatermark().to(device)

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer_cnn = optim.Adam(cnn.parameters(), lr=0.001)
optimizer_vit = optim.Adam(vit.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss_cnn = 0.0
    total_loss_vit = 0.0

    for original, tampered, watermarks in dataloader:
        original, watermarks = original.to(device, non_blocking=True), watermarks.to(device, non_blocking=True)

        # Forward pass
        watermarked, extracted = cnn(original, watermarks)
        vit_output = vit(original)

        # **Fix CNN output shape mismatch**
        extracted_gray = extracted.mean(dim=1, keepdim=True)  # Convert [batch, 3, 224, 224] → [batch, 1, 224, 224]

        # Compute losses
        loss_cnn = criterion(extracted_gray, watermarks)

        # Ensure ViT target size matches output
        watermarks_resized = torch.nn.functional.adaptive_avg_pool2d(watermarks, (16, 16))
        watermarks_flattened = watermarks_resized.view(watermarks.shape[0], -1)

        if vit_output.shape != watermarks_flattened.shape:
            print(f"Warning: ViT output shape {vit_output.shape} does not match target {watermarks_flattened.shape}")
        
        loss_vit = criterion(vit_output, watermarks_flattened)

        # Backpropagation
        optimizer_cnn.zero_grad()
        optimizer_vit.zero_grad()
        loss_cnn.backward()
        loss_vit.backward()
        optimizer_cnn.step()
        optimizer_vit.step()

        total_loss_cnn += loss_cnn.item()
        total_loss_vit += loss_vit.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss CNN: {total_loss_cnn/len(dataloader):.6f}, Loss ViT: {total_loss_vit/len(dataloader):.6f}")
