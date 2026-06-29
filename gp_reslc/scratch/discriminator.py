"""Lightweight discriminators for scratch perceptual compression experiments."""

from __future__ import annotations

from torch import nn


def sn(module: nn.Module) -> nn.Module:
    return nn.utils.spectral_norm(module)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 64, max_ch: int = 512):
        super().__init__()
        layers: list[nn.Module] = [
            sn(nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_ch
        for _ in range(3):
            out_ch = min(ch * 2, max_ch)
            layers.extend([
                sn(nn.Conv2d(ch, out_ch, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            ch = out_ch
        layers.extend([
            sn(nn.Conv2d(ch, ch, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(ch, 1, 3, padding=1)),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
