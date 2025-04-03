"""
Critic model for adversarial training.
"""
import torch
import torch.nn as nn
from .layers import ConvBlock, DownsampleBlock


class Critic(nn.Module):
    """
    Critic network that tries to distinguish between cover and stego images.
    """
    def __init__(self, hidden_channels=64):
        super(Critic, self).__init__()
        
        self.features = nn.Sequential(
            # Initial feature extraction
            ConvBlock(3, hidden_channels, kernel_size=3, padding=1, use_bn=False),
            
            # Downsampling layers
            DownsampleBlock(hidden_channels, hidden_channels*2),
            ConvBlock(hidden_channels*2, hidden_channels*2, kernel_size=3, padding=1),
            
            DownsampleBlock(hidden_channels*2, hidden_channels*4),
            ConvBlock(hidden_channels*4, hidden_channels*4, kernel_size=3, padding=1),
            
            DownsampleBlock(hidden_channels*4, hidden_channels*8),
            ConvBlock(hidden_channels*8, hidden_channels*8, kernel_size=3, padding=1),
        )
        
        # Global average pooling + final classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels*8, 1, kernel_size=1, padding=0)
        )
    
    def forward(self, image):
        """
        Forward pass of the critic.
        
        Args:
            image: The input image (cover or stego) [B, 3, H, W]
            
        Returns:
            Criticism score (higher means more likely to be stego)
        """
        features = self.features(image)
        score = self.classifier(features)
        return score.view(image.size(0), -1)


class ProgressiveCritic(nn.Module):
    """
    Progressive critic for different data capacity levels.
    """
    def __init__(self, num_levels=3, hidden_channels=64):
        super(ProgressiveCritic, self).__init__()
        
        # Create a critic for each level
        self.critics = nn.ModuleList([
            Critic(hidden_channels) for _ in range(num_levels)
        ])
        
        self.current_level = 0
    
    def forward(self, image):
        """
        Forward pass using the current level critic.
        
        Args:
            image: The input image (cover or stego) [B, 3, H, W]
            
        Returns:
            Criticism score
        """
        return self.critics[self.current_level](image)
    
    def set_level(self, level):
        """Set the current criticism level."""
        assert 0 <= level < len(self.critics), f"Level must be between 0 and {len(self.critics)-1}"
        self.current_level = level