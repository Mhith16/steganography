"""
Dataset classes for handling X-ray images and patient data.
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
from glob import glob


class XrayDataset(Dataset):
    """
    Dataset class for X-ray images.
    """
    def __init__(self, image_dir, transform=None, size=(256, 256)):
        self.image_dir = image_dir
        self.transform = transform
        self.size = size
        
        # Find all image files
        self.image_files = sorted(glob(os.path.join(image_dir, "*.jpg")))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.size)
        
        if self.transform:
            image = self.transform(image)
        
        return image


class PatientDataset(Dataset):
    """
    Dataset class for paired X-ray images and patient data.
    """
    def __init__(self, image_dir, data_dir, transform=None, size=(256, 256)):
        self.image_dir = image_dir
        self.data_dir = data_dir
        self.transform = transform
        self.size = size
        
        # Find all image files
        self.image_files = sorted(glob(os.path.join(image_dir, "*.jpg")))
        
        # Create a mapping of patient IDs to both image and data files
        self.pairs = []
        for img_path in self.image_files:
            # Extract base name without extension
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            data_path = os.path.join(data_dir, f"{base_name}.txt")
            
            # Only include if both image and data exist
            if os.path.exists(data_path):
                self.pairs.append((img_path, data_path))
            else:
                print(f"Warning: No matching text file for {img_path}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, data_path = self.pairs[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.size)
        
        if self.transform:
            image = self.transform(image)
        
        # Load text data
        with open(data_path, 'r', encoding='utf-8') as f:
            text_data = f.read()
        
        return image, text_data


def get_default_transforms():
    """
    Returns default image transforms for training and validation.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    return train_transform, val_transform


def create_train_val_split(dataset, val_ratio=0.2, seed=42):
    """
    Split a dataset into training and validation sets.
    
    Args:
        dataset: The dataset to split
        val_ratio: Ratio of validation samples (default: 0.2)
        seed: Random seed for reproducibility
        
    Returns:
        train_dataset, val_dataset
    """
    # Calculate sizes
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    
    # Create random split
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def text_to_binary_tensor(text, data_depth, height, width, device='cpu'):
    """
    Convert text to a binary tensor compatible with the steganography model.
    
    Args:
        text: The text to convert
        data_depth: The number of binary channels
        height, width: Dimensions of the tensor
        device: The device to create the tensor on
    
    Returns:
        A binary tensor of shape [1, data_depth, height, width]
    """
    # Convert text to binary
    binary = ''.join(format(ord(char), '08b') for char in text)
    
    # Create tensor
    tensor = torch.zeros(1, data_depth, height, width, device=device)
    
    # Fill tensor with binary data
    idx = 0
    for d in range(data_depth):
        for h in range(height):
            for w in range(width):
                if idx < len(binary):
                    tensor[0, d, h, w] = float(binary[idx])
                    idx = (idx + 1) % len(binary)  # Loop if we run out of data
    
    return tensor


def binary_tensor_to_text(tensor, max_length=1000):
    """
    Convert a binary tensor back to text.
    
    Args:
        tensor: Binary tensor from the decoder [1, data_depth, height, width]
        max_length: Maximum number of characters to decode
    
    Returns:
        The decoded text message
    """
    # Convert tensor to binary string
    if isinstance(tensor, torch.Tensor):
        # Convert logits to binary
        binary = (tensor >= 0).cpu().numpy().flatten()
    else:
        binary = tensor.flatten()
    
    # Convert binary to text
    text = ""
    for i in range(0, min(len(binary), max_length * 8), 8):
        if i + 8 <= len(binary):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | int(binary[i + j])
                
            # Only include printable ASCII characters
            if 32 <= byte <= 126:
                text += chr(byte)
    
    # Look for repeating patterns
    if len(text) > 0:
        # Try to find a repeating pattern
        for pattern_length in range(1, len(text) // 2):
            pattern = text[:pattern_length]
            if text[:pattern_length * 2] == pattern * 2:
                return pattern
    
    return text