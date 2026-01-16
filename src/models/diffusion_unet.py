import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for timestep embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: [batch_size]"""
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DoubleConv(nn.Module):
    """Double convolution block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNetModel(nn.Module):
    """U-Net model for diffusion."""
    def __init__(self, in_channels=3, base_channels=64, num_classes=3):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_classes = num_classes

        # Time embedding
        self.time_embed = nn.Sequential(
            PositionalEncoding(base_channels),
            nn.Linear(base_channels, base_channels * 4),
            nn.ReLU(),
            nn.Linear(base_channels * 4, base_channels)
        )

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Output
        self.outc = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x, t):
        """
        Args:
            x: [batch_size, in_channels, height, width]
            t: [batch_size] timestep indices
        Returns:
            out: [batch_size, num_classes, height, width]
        """
        # Time embedding
        t_emb = self.time_embed(t.float())
        t_emb = t_emb[:, :, None, None]  # [batch_size, base_channels, 1, 1]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        # Output
        out = self.outc(d1)
        return out
