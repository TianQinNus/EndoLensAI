"""Standalone copy of the project's frozen ResNet34 U-Net inference architecture.

State-dict names and operations match e4_windows/core/models.py. No pretrained
download, training code, PVT weights, or research workspace import is required.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet34


class DecoderBlock(nn.Module):
    def __init__(self, cin, skip_channels, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + skip_channels, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None, size=None):
        target = skip.shape[-2:] if skip is not None else size
        x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        return self.conv(x)


class ResNet34UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet34(weights=None)
        self.encoder.fc = nn.Identity()
        self.decoder = nn.ModuleList([
            DecoderBlock(512, 256, 256), DecoderBlock(256, 128, 128),
            DecoderBlock(128, 64, 64), DecoderBlock(64, 64, 32), DecoderBlock(32, 0, 16),
        ])
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        original_size, e = x.shape[-2:], self.encoder
        f0 = e.relu(e.bn1(e.conv1(x)))
        f1 = e.layer1(e.maxpool(f0))
        f2 = e.layer2(f1)
        f3 = e.layer3(f2)
        f4 = e.layer4(f3)
        z = f4
        for block, skip in zip(self.decoder[:4], (f3, f2, f1, f0)):
            z = block(z, skip)
        return self.head(self.decoder[4](z, size=original_size))
