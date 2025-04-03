"""
Custom layers and building blocks for steganography networks.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = self.relu(out)
        return out


class ConvBlock(nn.Module):
    """
    Convolutional block with batch normalization and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 use_bn=True, activation=True, use_leaky=True):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding)
        self.use_bn = use_bn
        self.activation = activation
        
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
            
        if activation:
            if use_leaky:
                self.act = nn.LeakyReLU(0.2, inplace=True)
            else:
                self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        
        if self.use_bn:
            x = self.bn(x)
            
        if self.activation:
            x = self.act(x)
            
        return x


class UpsampleBlock(nn.Module):
    """
    Upsampling block using nearest neighbor and convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.upsample(x)


class DownsampleBlock(nn.Module):
    """
    Downsampling block using convolution with stride.
    """
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.downsample(x)