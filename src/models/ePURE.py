# --- ePURE ---
class ePURE(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 1, 3, padding=1) # Ensure output is 1 channel for noise profile
        )

    def forward(self, x):
        x_float = x.float()

        # Estimate a base noise map from the input features
        noise_map_raw = self.conv(x_float) # Output is [B, 1, H, W]

        # Simple approach: just output the learned map directly.
        # The adaptive smoothing uses sigmoid, so the network learns to output values
        # that sigmoid can map to appropriate blending weights.
        noise_map = noise_map_raw # [B, 1, H, W]

        return noise_map # Noise profile estimate (1 channel)