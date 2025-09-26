import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def apply_tampering(self, image):
        """Simulates multiple levels of tampering"""
        tampering_levels = random.choice(["social_media", "edited", "severe"])
        
        if tampering_levels == "social_media":
            image = transforms.ColorJitter(brightness=0.2, contrast=0.2)(image)
        elif tampering_levels == "edited":
            image = transforms.GaussianBlur(3)(image)
        elif tampering_levels == "severe":
            image = transforms.RandomAffine(10, translate=(0.1, 0.1), shear=5)(image)
        
        return image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
            raise IndexError("Index out of range")

        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        return image
