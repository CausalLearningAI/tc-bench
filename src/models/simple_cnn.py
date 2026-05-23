"""
Simple CNN from scratch for cyclone intensity regression.

This is a baseline CNN trained from random initialization (no pretraining)
to validate that intensity is learnable directly from satellite imagery.
"""

import torch
import torch.nn as nn
from typing import Dict
from .base_regressor import BaseIntensityRegressor


class SimpleCNN(BaseIntensityRegressor):
    """
    Simple CNN trained from scratch for intensity prediction.
    
    Architecture:
    - 5 convolutional blocks with increasing channels
    - BatchNorm and ReLU activations
    - MaxPooling for downsampling
    - Global average pooling
    - Separate regression heads for pressure
    
    This validates that the raw frames contain learnable signal
    for intensity prediction without relying on pretrained features.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        **kwargs
    ):
        """
        Args:
            in_channels: Number of input channels (1 for infrared)
            base_channels: Base number of channels (will be multiplied)
            hidden_dim: Hidden dimension for MLP heads
            dropout: Dropout rate
            **kwargs: Arguments for BaseIntensityRegressor
        """
        super().__init__(**kwargs)
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.hidden_dim = hidden_dim
        
        # Convolutional backbone
        # Input: (B, 1, 224, 224)
        self.conv1 = self._make_conv_block(in_channels, base_channels)  # -> (B, 64, 112, 112)
        self.conv2 = self._make_conv_block(base_channels, base_channels * 2)  # -> (B, 128, 56, 56)
        self.conv3 = self._make_conv_block(base_channels * 2, base_channels * 4)  # -> (B, 256, 28, 28)
        self.conv4 = self._make_conv_block(base_channels * 4, base_channels * 8)  # -> (B, 512, 14, 14)
        self.conv5 = self._make_conv_block(base_channels * 8, base_channels * 8)  # -> (B, 512, 7, 7)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> (B, 512, 1, 1)
        
        feature_dim = base_channels * 8
        
        # Regression heads
        self.pressure_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block with Conv-BN-ReLU-Pool."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 1, H, W) or (B, H, W)
            
        Returns:
            dict with 'pressure' predictions
        """
        # Ensure 4D input (B, C, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional backbone
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Global pooling
        x = self.global_pool(x)  # (B, 512, 1, 1)
        x = x.flatten(1)  # (B, 512)
        
        # Regression heads
        pressure = self.pressure_head(x).squeeze(-1)  # (B,)
                
        return {
            'pressure': pressure,
        }


if __name__ == '__main__':
    # Test the model
    model = SimpleCNN()
    
    # Test with single frame
    x = torch.randn(4, 1, 224, 224)
    output = model(x)
    
    print("Input shape:", x.shape)
    print("Pressure output shape:", output['pressure'].shape)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
