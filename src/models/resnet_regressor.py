"""
ResNet-based intensity regressor trained from scratch.

A ResNet architecture trained from random initialization (no pretraining)
to validate that intensity is learnable directly from satellite imagery.
"""

import torch
import torch.nn as nn
from typing import Dict
from .base_regressor import BaseIntensityRegressor


class BasicBlock(nn.Module):
    """Basic ResNet block with two 3x3 convolutions."""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block with 1x1, 3x3, 1x1 convolutions."""
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class ResNetRegressor(BaseIntensityRegressor):
    """
    ResNet trained from scratch for cyclone intensity prediction.

    Architecture:
    - Initial 7x7 conv with stride 2
    - 4 residual stages with increasing channels
    - Global average pooling
    - MLP head for pressure prediction

    Uses BatchNorm for regularization (no dropout).
    Weight decay should be applied via optimizer config.

    Input: Single frame (B, 1, H, W) or (B, H, W)
    Output: Pressure prediction
    """

    # ResNet configurations: (block_type, [num_blocks per stage])
    CONFIGS = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
    }

    def __init__(
        self,
        in_channels: int = 1,
        arch: str = 'resnet50',
        base_channels: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        **kwargs
    ):
        """
        Args:
            in_channels: Number of input channels (1 for infrared)
            arch: Architecture variant ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            base_channels: Base number of channels for first stage
            hidden_dim: Hidden dimension for MLP head
            dropout: Dropout rate for regularization (0.0 = no dropout)
            **kwargs: Arguments for BaseIntensityRegressor
        """
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.arch = arch
        self.base_channels = base_channels
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        if arch not in self.CONFIGS:
            raise ValueError(f"Unknown architecture: {arch}. Choose from {list(self.CONFIGS.keys())}")

        block, num_blocks = self.CONFIGS[arch]

        # Initial convolution: 7x7, stride 2
        # Input: (B, 1, 224, 224) -> (B, 64, 112, 112)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # After maxpool: (B, 64, 56, 56)

        # Residual stages
        self.current_channels = base_channels
        self.layer1 = self._make_layer(block, base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, num_blocks[3], stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Feature dimension after backbone
        feature_dim = base_channels * 8 * block.expansion

        # Regression head with optional dropout for regularization
        if self.dropout_rate > 0:
            self.pressure_head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            self.pressure_head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1)
            )

        # Initialize weights
        self._initialize_weights()

        # Compile model if requested (PyTorch 2.0+)
        if self.compile_model:
            self = torch.compile(self)

    def _make_layer(self, block, out_channels: int, num_blocks: int, stride: int):
        """Create a residual stage."""
        layers = []
        # First block may downsample
        layers.append(block(self.current_channels, out_channels, stride))
        self.current_channels = out_channels * block.expansion
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.current_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

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
            x: Input frames, shape (B, H, W) or (B, 1, H, W)

        Returns:
            Dict with 'pressure' predictions
        """
        # Ensure input has channel dimension
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        # Initial conv + pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.avgpool(x)
        x = x.flatten(1)

        # Pressure prediction
        pressure = self.pressure_head(x).squeeze(-1)

        return {
            "pressure": pressure,
        }


if __name__ == "__main__":
    # Test the model
    print("Testing ResNetRegressor (from scratch)...")

    for arch in ['resnet18', 'resnet34', 'resnet50']:
        print(f"\n{'='*60}")
        print(f"Architecture: {arch}")
        print('='*60)

        model = ResNetRegressor(
            arch=arch,
            in_channels=1,
            base_channels=64,
            hidden_dim=512,
            learning_rate=1e-4,
            weight_decay=1e-4,
            target_variable="pressure"
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 224, 224)
        print(f"Input shape: {x.shape}")

        outputs = model(x)
        print(f"Pressure output shape: {outputs['pressure'].shape}")

    print("\nResNetRegressor test complete!")
