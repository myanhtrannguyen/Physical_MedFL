# Federated-Learning/src/models/unet.py

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Một khối tích chập kép (Conv -> BN -> ReLU)*2."""

    def __init__(self, in_ch, out_ch, dropout=0.1):
        """Private init function."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        """Forward function."""
        return self.block(x)


class UNet(nn.Module):
    """
    Kiến trúc mạng U-Net chuẩn cho bài toán phân vùng.
    Đây là kiến trúc bạn đã cung cấp từ notebook Kaggle.
    """

    def __init__(self, in_ch=1, num_classes=4, base_c=32):
        """Private init function."""
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_c)
        self.enc2 = ConvBlock(base_c, base_c * 2)
        self.enc3 = ConvBlock(base_c * 2, base_c * 4)
        self.enc4 = ConvBlock(base_c * 4, base_c * 8)

        self.pool = nn.MaxPool2d(2)

        self.center = ConvBlock(base_c * 8, base_c * 16)

        self.up4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_c * 16, base_c * 8)

        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_c * 8, base_c * 4)

        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_c * 4, base_c * 2)

        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_c * 2, base_c)

        self.final = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        c = self.center(self.pool(e4))

        d4 = self.up4(c)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out
