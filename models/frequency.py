import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


class FrequencyAttention(nn.Module):
    """
    Attention mechanism for wavelet-domain low-frequency features.
    Applies adaptive channel attention to refine low-frequency features.
    """
    def __init__(self, in_channels, reduction=16):
        super(FrequencyAttention, self).__init__()
        hidden_channels = max(in_channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FrequencyBranchProcessor(nn.Module):
    """
    Processes features in the wavelet domain.

    Replaces FFT with single-level 2D DWT:
    - DWT decomposition -> yl (low-frequency), yh (high-frequency)
    - process yl
    - reconstruct with IDWT

    Notes:
    - `yl` shape: [B, C, H/2, W/2]
    - `yh[0]` shape: [B, C, 3, H/2, W/2]
      where 3 corresponds to LH, HL, HH
    """
    def __init__(self, in_channels, out_channels, wave='haar', mode='zero'):
        super(FrequencyBranchProcessor, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 2D DWT / IDWT layers
        self.dwt = DWTForward(J=1, mode=mode, wave=wave)
        self.idwt = DWTInverse(mode=mode, wave=wave)

        # Process low-frequency branch (yl)
        self.freq_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.freq_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.freq_attention = FrequencyAttention(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU(out_channels)

        # To make IDWT channel-compatible with original high-frequency bands
        if out_channels != in_channels:
            self.low_restore = nn.Conv2d(out_channels, in_channels, kernel_size=1)
            self.output_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.low_restore = nn.Identity()
            self.output_proj = nn.Identity()

    def forward(self, x):
        # Wavelet decomposition
        yl, yh = self.dwt(x)   # yl: low-frequency, yh: list of high-frequency bands

        # Process low-frequency component
        processed_yl = self.freq_conv1(yl)
        processed_yl = self.relu(self.bn(processed_yl))
        processed_yl = self.freq_conv2(processed_yl)
        processed_yl = self.freq_attention(processed_yl)

        # Restore channels for reconstruction if needed
        restored_yl = self.low_restore(processed_yl)

        # Reconstruct using original high-frequency bands
        spatial_output = self.idwt((restored_yl, yh))

        # Crop if needed to exactly match input spatial size
        spatial_output = spatial_output[:, :, :x.size(2), :x.size(3)]

        # If out_channels != in_channels, project final output
        spatial_output = self.output_proj(spatial_output)

        return spatial_output


class FrequencyGuidedFusion(nn.Module):
    """
    Fuses spatial and wavelet-domain features.
    Uses attention mechanisms to adaptively combine features from both domains.
    """
    def __init__(self, channels):
        super(FrequencyGuidedFusion, self).__init__()
        from .attention import CBAM

        self.spatial_attention = CBAM(channels)

        # spatial_features: channels
        # frequency_features: channels
        # concat -> 2 * channels
        self.fusion_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.activate = nn.PReLU(channels)

    def forward(self, spatial_features, frequency_features):
        # Apply attention to spatial features
        attended_spatial = self.spatial_attention(spatial_features)

        # Adjust channel mismatch if needed
        if attended_spatial.size(1) != frequency_features.size(1):
            adapter = nn.Conv2d(
                frequency_features.size(1),
                attended_spatial.size(1),
                kernel_size=1
            ).to(spatial_features.device)
            frequency_features = adapter(frequency_features)

        # Concatenate and fuse
        fused = torch.cat([attended_spatial, frequency_features], dim=1)
        output = self.fusion_conv(fused)
        output = self.activate(self.norm(output))

        return output