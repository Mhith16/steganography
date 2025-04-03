"""
Consistent metrics calculation for image quality assessment.
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch


def calculate_psnr(original, stego, rgb=True):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        original: Original image (numpy array or torch tensor)
        stego: Stego image (numpy array or torch tensor)
        rgb: Whether images are RGB (True) or grayscale (False)
    
    Returns:
        PSNR value in dB
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(original, torch.Tensor):
        original = original.cpu().detach().numpy()
    if isinstance(stego, torch.Tensor):
        stego = stego.cpu().detach().numpy()
    
    # Ensure images are in the right shape for skimage functions
    if len(original.shape) == 4:  # batch of images
        # Take the first image if it's a batch
        original = original[0]
        stego = stego[0]
    
    # Handle channel dimension for torch tensors (C,H,W) -> (H,W,C)
    if original.shape[0] == 3 and len(original.shape) == 3:
        original = np.transpose(original, (1, 2, 0))
        stego = np.transpose(stego, (1, 2, 0))
    
    # Ensure pixel values are in [0, 1] range for proper PSNR calculation
    if original.max() > 1.0:
        original = original / 255.0
    if stego.max() > 1.0:
        stego = stego / 255.0
    
    # Calculate PSNR
    return peak_signal_noise_ratio(original, stego, data_range=1.0)


def calculate_ssim(original, stego, rgb=True):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        original: Original image (numpy array or torch tensor)
        stego: Stego image (numpy array or torch tensor)
        rgb: Whether images are RGB (True) or grayscale (False)
    
    Returns:
        SSIM value [0, 1]
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(original, torch.Tensor):
        original = original.cpu().detach().numpy()
    if isinstance(stego, torch.Tensor):
        stego = stego.cpu().detach().numpy()
    
    # Ensure images are in the right shape for skimage functions
    if len(original.shape) == 4:  # batch of images
        # Take the first image if it's a batch
        original = original[0]
        stego = stego[0]
    
    # Handle channel dimension for torch tensors (C,H,W) -> (H,W,C)
    if original.shape[0] == 3 and len(original.shape) == 3:
        original = np.transpose(original, (1, 2, 0))
        stego = np.transpose(stego, (1, 2, 0))
    
    # Ensure pixel values are in [0, 1] range for proper SSIM calculation
    if original.max() > 1.0:
        original = original / 255.0
    if stego.max() > 1.0:
        stego = stego / 255.0
    
    # Calculate SSIM
    if rgb:
        return structural_similarity(
            original, stego, 
            channel_axis=2 if len(original.shape) > 2 else None,
            data_range=1.0
        )
    else:
        # For grayscale, convert to grayscale if needed
        if len(original.shape) > 2 and original.shape[2] == 3:
            original_gray = np.mean(original, axis=2)
            stego_gray = np.mean(stego, axis=2)
            return structural_similarity(
                original_gray, stego_gray,
                data_range=1.0
            )
        else:
            return structural_similarity(
                original, stego,
                data_range=1.0
            )


def calculate_bit_accuracy(original_bits, decoded_bits):
    """
    Calculate bit accuracy between original and decoded binary data.
    
    Args:
        original_bits: Original binary data (numpy array or torch tensor)
        decoded_bits: Decoded binary data (numpy array or torch tensor)
    
    Returns:
        Accuracy as a fraction [0, 1]
    """
    if isinstance(original_bits, torch.Tensor):
        original_bits = original_bits.cpu().detach().numpy()
    if isinstance(decoded_bits, torch.Tensor):
        decoded_bits = decoded_bits.cpu().detach().numpy()
    
    # Handle logits in decoded bits
    if np.issubdtype(decoded_bits.dtype, np.floating):
        decoded_bits = decoded_bits >= 0
    
    # Flatten arrays if needed
    original_bits = original_bits.flatten()
    decoded_bits = decoded_bits.flatten()
    
    # Ensure lengths match
    min_len = min(len(original_bits), len(decoded_bits))
    original_bits = original_bits[:min_len]
    decoded_bits = decoded_bits[:min_len]
    
    # Calculate accuracy
    return np.mean(original_bits == decoded_bits)