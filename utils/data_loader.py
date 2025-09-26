import torch
from torch.utils.data import DataLoader
from utils.flickr8k_dataset import Flickr8kDataset  # Ensure correct import

def get_dataloader(image_dir, batch_size=32, shuffle=True, pin_memory=False):
    dataset = Flickr8kDataset(image_dir)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
