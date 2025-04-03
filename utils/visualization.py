"""
Utilities for visualizing steganography results.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import io
import base64
from .metrics import calculate_psnr, calculate_ssim


def denormalize_image(image):
    """
    Denormalize image from [-1, 1] to [0, 1] range.
    
    Args:
        image: Normalized image tensor or numpy array
        
    Returns:
        Denormalized image in [0, 1] range
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        
    # Convert from [-1, 1] to [0, 1]
    return (image + 1) / 2


def visualize_result(original, stego, title=None, show_metrics=True, save_path=None):
    """
    Visualize original and stego images side by side.
    
    Args:
        original: Original image tensor or numpy array
        stego: Stego image tensor or numpy array
        title: Optional title for the plot
        show_metrics: Whether to calculate and show metrics
        save_path: Optional path to save the visualization
        
    Returns:
        Figure object
    """
    # Denormalize images
    original_img = denormalize_image(original)
    stego_img = denormalize_image(stego)
    
    # Handle batch dimension
    if len(original_img.shape) == 4:
        original_img = original_img[0]
    if len(stego_img.shape) == 4:
        stego_img = stego_img[0]
    
    # Convert from CHW to HWC format if needed
    if original_img.shape[0] == 3 and len(original_img.shape) == 3:
        original_img = np.transpose(original_img, (1, 2, 0))
    if stego_img.shape[0] == 3 and len(stego_img.shape) == 3:
        stego_img = np.transpose(stego_img, (1, 2, 0))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot images
    ax1.imshow(np.clip(original_img, 0, 1))
    ax1.set_title("Original X-ray")
    ax1.axis('off')
    
    # Calculate and display metrics
    metrics_text = ""
    if show_metrics:
        psnr = calculate_psnr(original_img, stego_img)
        ssim = calculate_ssim(original_img, stego_img)
        metrics_text = f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}"
    
    ax2.imshow(np.clip(stego_img, 0, 1))
    ax2.set_title(f"Steganographic X-ray\n{metrics_text}")
    ax2.axis('off')
    
    # Set main title
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_difference(original, stego, amplification=10, title=None, save_path=None):
    """
    Visualize the difference between original and stego images.
    
    Args:
        original: Original image tensor or numpy array
        stego: Stego image tensor or numpy array
        amplification: Factor to amplify the difference
        title: Optional title for the plot
        save_path: Optional path to save the visualization
        
    Returns:
        Figure object
    """
    # Denormalize images
    original_img = denormalize_image(original)
    stego_img = denormalize_image(stego)
    
    # Handle batch dimension
    if len(original_img.shape) == 4:
        original_img = original_img[0]
    if len(stego_img.shape) == 4:
        stego_img = stego_img[0]
    
    # Convert from CHW to HWC format if needed
    if original_img.shape[0] == 3 and len(original_img.shape) == 3:
        original_img = np.transpose(original_img, (1, 2, 0))
    if stego_img.shape[0] == 3 and len(stego_img.shape) == 3:
        stego_img = np.transpose(stego_img, (1, 2, 0))
    
    # Calculate difference
    diff = stego_img - original_img
    
    # Amplify difference for visibility
    diff_amplified = diff * amplification
    
    # Map difference to color range
    diff_colored = np.zeros_like(diff)
    diff_colored[:, :, 0] = np.clip(diff_amplified[:, :, 0], -1, 0) * -1  # Red for negative
    diff_colored[:, :, 1] = np.clip(diff_amplified[:, :, 1], 0, 1)        # Green for positive
    diff_colored[:, :, 2] = np.clip(diff_amplified[:, :, 2], -1, 0) * -1  # Blue for negative
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot images
    ax1.imshow(np.clip(original_img, 0, 1))
    ax1.set_title("Original X-ray")
    ax1.axis('off')
    
    ax2.imshow(np.clip(stego_img, 0, 1))
    ax2.set_title("Steganographic X-ray")
    ax2.axis('off')
    
    ax3.imshow(np.clip(diff_colored, 0, 1))
    ax3.set_title(f"Difference (×{amplification})")
    ax3.axis('off')
    
    # Set main title
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_binary_data(binary_data, title=None, save_path=None):
    """
    Visualize binary data as a heatmap.
    
    Args:
        binary_data: Binary data tensor or numpy array
        title: Optional title for the plot
        save_path: Optional path to save the visualization
        
    Returns:
        Figure object
    """
    # Convert to numpy if tensor
    if isinstance(binary_data, torch.Tensor):
        binary_data = binary_data.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(binary_data.shape) == 4 and binary_data.shape[0] == 1:
        binary_data = binary_data[0]
    
    # Get dimensions
    data_depth, height, width = binary_data.shape
    
    # Create figure
    fig, axes = plt.subplots(1, data_depth, figsize=(data_depth * 4, 4))
    
    # Handle single depth case
    if data_depth == 1:
        axes = [axes]
    
    # Plot each depth as a heatmap
    for d in range(data_depth):
        im = axes[d].imshow(binary_data[d], cmap='Blues', vmin=0, vmax=1)
        axes[d].set_title(f"Channel {d+1}")
        axes[d].axis('off')
        fig.colorbar(im, ax=axes[d], fraction=0.046, pad=0.04)
    
    # Set main title
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def convert_figure_to_base64(fig):
    """
    Convert a matplotlib figure to base64 string.
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        Base64 encoded string of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str