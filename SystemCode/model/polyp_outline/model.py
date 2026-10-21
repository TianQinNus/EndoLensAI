"""Frozen inference architecture extracted from the selected training source."""
import torch
from torch import nn

def double_conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """UNet++ nested skip topology, project GN blocks, final-head only.

    Independent implementation of Zhou et al. 2018 equation (1).
    No deep supervision; name UNet retained for evaluation interface.
    """
    def __init__(self, base=32):
        super().__init__()
        channels = [base * 2**i for i in range(5)]
        self.encoder = nn.ModuleList([double_conv(3, base)] +
                                    [double_conv(a, b) for a, b in zip(channels, channels[1:])])
        self.upsample = nn.ModuleDict()
        self.nested = nn.ModuleDict()
        for j in range(1, 5):
            for i in range(5-j):
                key = f'{i}_{j}'
                self.upsample[key] = nn.ConvTranspose2d(channels[i+1], channels[i], 2, 2)
                self.nested[key] = double_conv((j+1)*channels[i], channels[i])
        # Parallel dilated convolutions at the bottleneck; residual fusion.
        c = channels[-1]
        self.context = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, c//4, 3, padding=d, dilation=d),
                          nn.GroupNorm(8, c//4), nn.ReLU(inplace=True))
            for d in (1, 2, 4)])
        self.context_fuse = nn.Conv2d(3*(c//4), c, 1)
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        nodes = {}
        for i, block in enumerate(self.encoder):
            x = block(nn.functional.max_pool2d(x, 2) if i else x)
            if i == 4:
                x = x + self.context_fuse(torch.cat([b(x) for b in self.context], dim=1))
            nodes[i, 0] = x
        for j in range(1, 5):
            for i in range(5-j):
                key = f'{i}_{j}'
                up = self.upsample[key](nodes[i+1, j-1])
                nodes[i, j] = self.nested[key](torch.cat(
                    [nodes[i, k] for k in range(j)] + [up], dim=1))
        return self.output(nodes[0, 4])

