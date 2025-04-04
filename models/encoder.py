"""
Enhanced encoder model for steganography.
"""
import torch
import torch.nn as nn
from .layers import ResidualBlock, ConvBlock, DownsampleBlock, UpsampleBlock


class SteganoEncoder(nn.Module):
    """
    Enhanced encoder network for hiding messages in images.
    """
    def __init__(self, data_depth=1, hidden_blocks=4, hidden_channels=32, image_size=256):
        super(SteganoEncoder, self).__init__()
        self.data_depth = data_depth
        self.image_size = image_size
        
        # Initial feature extraction
        self.initial = ConvBlock(3, hidden_channels, kernel_size=3, padding=1)
        
        # Downsample path
        self.down1 = DownsampleBlock(hidden_channels, hidden_channels*2)
        self.down2 = DownsampleBlock(hidden_channels*2, hidden_channels*4)
        
        # Data processing
        self.data_conv = ConvBlock(data_depth, hidden_channels, kernel_size=3, padding=1)
        
        # Process data and features
        self.combine = ConvBlock(hidden_channels*4 + hidden_channels, hidden_channels*4, kernel_size=3, padding=1)
        
        # Residual blocks for processing
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels*4) for _ in range(hidden_blocks)]
        )
        
        # Upsample path
        self.up1 = UpsampleBlock(hidden_channels*4, hidden_channels*2)
        self.up2 = UpsampleBlock(hidden_channels*2, hidden_channels)
        
        # Final image generation with less impact on the image
        self.final = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.LayerNorm([3, image_size, image_size])  # Add layer normalization
        )
    
    def forward(self, image, data):
        """
        Forward pass of the encoder.
        
        Args:
            image: The cover image [B, 3, H, W]
            data: The data to hide [B, D, H, W]
            
        Returns:
            The stego image with hidden data
        """
        # Extract features from image
        x = self.initial(image)  # [B, hidden_channels, H, W]
        
        # Downsample
        x1 = self.down1(x)  # [B, hidden_channels*2, H/2, W/2]
        x2 = self.down2(x1)  # [B, hidden_channels*4, H/4, W/4]
        
        # Process data
        d = self.data_conv(data)  # [B, hidden_channels, H, W]
        
        # Downsample data to match feature map size
        d_down = nn.functional.interpolate(d, size=x2.shape[2:], mode='bilinear', align_corners=False)
        
        # Combine features and data
        combined = torch.cat([x2, d_down], dim=1)  # [B, hidden_channels*4 + hidden_channels, H/4, W/4]
        combined = self.combine(combined)  # [B, hidden_channels*4, H/4, W/4]
        
        # Process through residual blocks
        res_out = self.res_blocks(combined)  # [B, hidden_channels*4, H/4, W/4]
        
        # Upsample
        u1 = self.up1(res_out)  # [B, hidden_channels*2, H/2, W/2]
        u2 = self.up2(u1)  # [B, hidden_channels, H, W]
        
        # Generate residual
        delta = self.final(u2)  # [B, 3, H, W]
        
        # Apply residual connection
        stego = image + delta
        
        # Ensure output is in valid range [-1, 1]
        return torch.clamp(stego, -1, 1)


class ProgressiveSteganoEncoder(nn.Module):
    """
    Progressive encoder with multiple data depths.
    """
    def __init__(self, data_depths=[1, 2, 4], hidden_blocks=4, hidden_channels=32, image_size=256):
        super(ProgressiveSteganoEncoder, self).__init__()
        
        # Create an encoder for each data depth
        self.encoders = nn.ModuleList([
            SteganoEncoder(depth, hidden_blocks, hidden_channels, image_size) 
            for depth in data_depths
        ])
        
        self.data_depths = data_depths
        self.current_level = 0  # Start with the smallest data depth
    
    def forward(self, image, data):
        """
        Forward pass using the current level encoder.
        
        Args:
            image: The cover image [B, 3, H, W]
            data: The data to hide [B, D, H, W] (D should match current level's data_depth)
            
        Returns:
            The stego image with hidden data
        """
        return self.encoders[self.current_level](image, data)
    
    def set_level(self, level):
        """Set the current encoding level."""
        assert 0 <= level < len(self.encoders), f"Level must be between 0 and {len(self.encoders)-1}"
        self.current_level = level
        return self.data_depths[level]