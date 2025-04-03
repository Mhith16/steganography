"""
Enhanced decoder model for steganography.
"""
import torch
import torch.nn as nn
from .layers import ResidualBlock, ConvBlock, DownsampleBlock


class SteganoDecoder(nn.Module):
    """
    Enhanced decoder network for extracting hidden messages from images.
    """
    def __init__(self, data_depth=1, hidden_blocks=4, hidden_channels=32):
        super(SteganoDecoder, self).__init__()
        self.data_depth = data_depth
        
        # Initial feature extraction from stego image
        self.initial = ConvBlock(3, hidden_channels, kernel_size=3, padding=1)
        
        # Downsampling path for feature extraction
        self.down1 = DownsampleBlock(hidden_channels, hidden_channels*2)
        self.down2 = DownsampleBlock(hidden_channels*2, hidden_channels*4)
        
        # Residual blocks for processing
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels*4) for _ in range(hidden_blocks)]
        )
        
        # Data recovery path
        self.recovery = nn.Sequential(
            ConvBlock(hidden_channels*4, hidden_channels*2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(hidden_channels*2, hidden_channels, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(hidden_channels, hidden_channels//2, kernel_size=3, padding=1),
        )
        
        # Final data extraction
        self.final = nn.Conv2d(hidden_channels//2, data_depth, kernel_size=3, padding=1)
    
    def forward(self, stego_image):
        """
        Forward pass of the decoder.
        
        Args:
            stego_image: The stego image with hidden data [B, 3, H, W]
            
        Returns:
            The extracted data [B, data_depth, H, W]
        """
        # Extract features from stego image
        x = self.initial(stego_image)  # [B, hidden_channels, H, W]
        
        # Downsample
        x = self.down1(x)  # [B, hidden_channels*2, H/2, W/2]
        x = self.down2(x)  # [B, hidden_channels*4, H/4, W/4]
        
        # Process through residual blocks
        x = self.res_blocks(x)  # [B, hidden_channels*4, H/4, W/4]
        
        # Data recovery
        x = self.recovery(x)  # [B, hidden_channels//2, H, W]
        
        # Final data extraction
        data = self.final(x)  # [B, data_depth, H, W]
        
        return data


class ProgressiveSteganoDecoder(nn.Module):
    """
    Progressive decoder with multiple data depths.
    """
    def __init__(self, data_depths=[1, 2, 4], hidden_blocks=4, hidden_channels=32):
        super(ProgressiveSteganoDecoder, self).__init__()
        
        # Create a decoder for each data depth
        self.decoders = nn.ModuleList([
            SteganoDecoder(depth, hidden_blocks, hidden_channels) 
            for depth in data_depths
        ])
        
        self.data_depths = data_depths
        self.current_level = 0  # Start with the smallest data depth
    
    def forward(self, stego_image):
        """
        Forward pass using the current level decoder.
        
        Args:
            stego_image: The stego image with hidden data [B, 3, H, W]
            
        Returns:
            The extracted data
        """
        return self.decoders[self.current_level](stego_image)
    
    def set_level(self, level):
        """Set the current decoding level."""
        assert 0 <= level < len(self.decoders), f"Level must be between 0 and {len(self.decoders)-1}"
        self.current_level = level
        return self.data_depths[level]