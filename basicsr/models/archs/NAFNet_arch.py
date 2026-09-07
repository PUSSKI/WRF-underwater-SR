import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class HaarDWT2D(nn.Module):
    """One-level fixed Haar DWT."""
    def forward(self, x):
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh


class HaarIDWT2D(nn.Module):
    """Inverse transform for the fixed Haar DWT used above."""
    def forward(self, ll, lh, hl, hh):
        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5

        b, c, h, w = ll.shape
        out = ll.new_zeros(b, c, h * 2, w * 2)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        return out


class WaveletAttentionRefinement(nn.Module):
    """Lightweight Wavelet-Attention refinement.

    This is inspired by SwinWave-SR, but deliberately much lighter than a full
    Swin transformer. It uses DWT as an invertible downsampling operation,
    learns local context in the wavelet subbands, reconstructs it with IDWT,
    and applies a weak spatial-channel gate. This improves texture/structure
    without the strong global color shift that ordinary Swin blocks caused.
    """

    def __init__(self, channels, reduction=4, residual_scale=0.08):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()
        self.norm = LayerNorm2d(channels)

        # Contextualize four wavelet subbands. Depthwise convolution keeps this
        # inexpensive and avoids over-mixing RGB/color-related features.
        self.subband_context = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 4, 3, 1, 1, groups=channels * 4, bias=True),
            nn.Conv2d(channels * 4, channels * 2, 1, 1, 0, bias=True),
            SimpleGate(),
            nn.Conv2d(channels, channels * 4, 1, 1, 0, bias=True),
        )

        # Weak attention gate between the original fused feature and reconstructed
        # wavelet context. Zero initialization makes the block start as identity.
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x):
        identity = x
        _, _, h, w = x.shape
        x_norm = self.norm(x)

        ll, lh, hl, hh = self.dwt(x_norm)
        subbands = torch.cat([ll, lh, hl, hh], dim=1)
        subbands = self.subband_context(subbands)
        ll2, lh2, hl2, hh2 = subbands.chunk(4, dim=1)

        context = self.idwt(ll2, lh2, hl2, hh2)
        context = context[:, :, :h, :w]

        gate = self.gate(torch.cat([x_norm, context], dim=1))
        refined = self.out_proj(context * gate)
        return identity + self.scale * refined


class MultiLevelWaveletRefinement(nn.Module):
    """Stacked lightweight Wavelet-Attention blocks.

    Two blocks behave like a stable multi-level wavelet refinement: the first
    focuses on fine local details, and the second re-contextualizes the fused
    features again without explicitly shrinking the whole SR pipeline too much.
    """

    def __init__(self, channels, num_blocks=2, residual_scale=0.08):
        super().__init__()
        self.blocks = nn.Sequential(*[
            WaveletAttentionRefinement(channels, residual_scale=residual_scale)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)



class NAFStack(nn.Module):
    def __init__(self, channels, num_blocks):
        super().__init__()
        self.body = nn.Sequential(*[NAFBlock(channels) for _ in range(num_blocks)])

    def forward(self, x):
        return self.body(x)


class PyramidContextRefine(nn.Module):
    """Efficient-USR-style LR pyramid context for SR structure recovery.

    The branch reads the padded LR image at 1x/0.5x/0.25x scales, lifts each
    scale into feature space, and returns a small residual correction. The final
    projection starts from zero so enabling the block does not disturb an
    existing stable SR path at initialization.
    """

    def __init__(self, channels, img_channel=3, scale=0.08):
        super().__init__()
        self.feat_1x = nn.Sequential(
            nn.Conv2d(img_channel, channels, 3, 1, 1, bias=True),
            NAFBlock(channels),
        )
        self.feat_2x = nn.Sequential(
            nn.Conv2d(img_channel, channels, 3, 1, 1, bias=True),
            NAFBlock(channels),
        )
        self.feat_4x = nn.Sequential(
            nn.Conv2d(img_channel, channels, 3, 1, 1, bias=True),
            NAFBlock(channels),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, 1, 1, 0, bias=True),
            SimpleGate(),
            NAFBlock(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, feat, inp):
        h, w = feat.shape[-2:]
        inp_2x = F.interpolate(inp, scale_factor=0.5, mode='bicubic', align_corners=False)
        inp_4x = F.interpolate(inp_2x, scale_factor=0.5, mode='bicubic', align_corners=False)

        ctx_1x = self.feat_1x(inp)
        ctx_2x = self.feat_2x(inp_2x)
        ctx_4x = self.feat_4x(inp_4x)
        ctx_2x = F.interpolate(ctx_2x, size=(h, w), mode='bilinear', align_corners=False)
        ctx_4x = F.interpolate(ctx_4x, size=(h, w), mode='bilinear', align_corners=False)

        ctx = self.fuse(torch.cat([feat, ctx_1x, ctx_2x, ctx_4x], dim=1))
        return feat + self.scale.clamp(0.0, 0.3) * ctx


class MSCASpatialRefine(nn.Module):
    """Large-kernel spatial context inspired by Efficient-USR's MSCA block."""

    def __init__(self, channels, scale=0.06):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.proj_1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.spatial_5 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=True)
        self.spatial_7 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 7), 1, (0, 3), groups=channels, bias=True),
            nn.Conv2d(channels, channels, (7, 1), 1, (3, 0), groups=channels, bias=True),
        )
        self.spatial_11 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 11), 1, (0, 5), groups=channels, bias=True),
            nn.Conv2d(channels, channels, (11, 1), 1, (5, 0), groups=channels, bias=True),
        )
        self.spatial_21 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 21), 1, (0, 10), groups=channels, bias=True),
            nn.Conv2d(channels, channels, (21, 1), 1, (10, 0), groups=channels, bias=True),
        )
        self.proj_2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        nn.init.zeros_(self.proj_2.weight)
        nn.init.zeros_(self.proj_2.bias)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        identity = x
        x = self.proj_1(self.norm(x))
        attn = self.spatial_5(x)
        attn = attn + self.spatial_7(attn) + self.spatial_11(attn) + self.spatial_21(attn)
        x = self.proj_2(attn * x)
        return identity + self.scale.clamp(0.0, 0.3) * x


class FFTFeatureRefineBlock(nn.Module):
    """Efficient-USR-style lightweight global frequency refinement.

    Haar/DWT captures local frequency changes well, but it is weak at global
    texture and blur patterns. This block refines fused LR features in the
    Fourier domain, then returns a small residual. The final projection starts
    from zero, so enabling it is conservative and easy to ablate.
    """

    def __init__(self, channels, hidden_scale=1.0, scale=0.08):
        super().__init__()
        hidden = max(8, int(channels * hidden_scale))
        self.norm = LayerNorm2d(channels)
        self.freq_in = nn.Conv2d(channels, hidden, 1, 1, 0, bias=True)
        self.freq_dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=True)
        self.freq_out = nn.Conv2d(hidden, channels, 1, 1, 0, bias=True)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=True),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        identity = x
        feat = self.norm(x)
        freq = torch.fft.fftn(feat, dim=(-2, -1)).real
        freq = self.freq_out(self.freq_dw(F.gelu(self.freq_in(freq))))
        freq = torch.fft.ifftn(freq, dim=(-2, -1)).real
        freq = self.proj(freq * self.spatial_gate(feat))
        return identity + self.scale.clamp(0.0, 0.3) * freq


class ResidualDenseBlock(nn.Module):
    """Residual dense block without BN, better suited for SR texture recovery."""

    def __init__(self, channels, growth_channels=32, num_layers=3, residual_scale=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = channels
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, growth_channels, 3, 1, 1, bias=True),
                nn.ReLU(inplace=True),
            ))
            in_channels += growth_channels
        self.fuse = nn.Conv2d(in_channels, channels, 1, 1, 0, bias=True)
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            feats.append(layer(torch.cat(feats, dim=1)))
        return x + self.scale * self.fuse(torch.cat(feats, dim=1))


class ResidualDenseStack(nn.Module):
    """Stack residual dense blocks with a global residual shortcut."""

    def __init__(self, channels, num_blocks=3, growth_channels=32, residual_scale=0.2):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResidualDenseBlock(
                channels,
                growth_channels=growth_channels,
                residual_scale=residual_scale,
            )
            for _ in range(num_blocks)
        ])
        self.out = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return x + self.out(self.blocks(x))


class FeatureUpsample(nn.Module):
    """Stable learnable feature upsampling for low/high frequency branches.

    The previous implementation used plain bilinear interpolation for both
    low- and high-frequency features. That is safe, but it can over-smooth
    high-frequency features before they enter the encoder-decoder. This module
    keeps a simple interpolation shortcut and learns a residual correction after
    PixelShuffle. The last convolution is zero-initialized, so it starts close
    to the shortcut and learns sharper feature upsampling during training.
    """

    def __init__(self, channels, base_mode='bilinear', residual_scale=0.2):
        super().__init__()
        self.base_mode = base_mode
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, 1, 1, bias=True),
            nn.PixelShuffle(2),
            NAFBlock(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x):
        if self.base_mode in ['bilinear', 'bicubic']:
            base = F.interpolate(x, scale_factor=2, mode=self.base_mode, align_corners=False)
        else:
            base = F.interpolate(x, scale_factor=2, mode=self.base_mode)
        return base + self.scale * self.body(x)


class HighFrequencyEnhancer(nn.Module):
    """Strengthen wavelet high-frequency features without changing tensor size."""
    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=True),
            nn.Conv2d(channels, channels * 2, 1, 1, 0, bias=True),
            SimpleGate(),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        detail = self.refine(x)
        detail = detail * self.attn(detail)
        return x + self.scale * detail


class SoftForegroundBackgroundMask(nn.Module):
    """Learnable soft foreground/background mask predictor.

    The mask is not a hard 0/1 segmentation. It predicts a continuous weight
    M in [0, 1] for each spatial location:
      - M close to 1: foreground / structure / texture region, more detail-aware;
      - M close to 0: background / smooth water region, more enhancement-stable.

    Inputs:
      x:           RGB image or padded LR image, B x 3 x H x W
      high_energy: wavelet high-frequency energy, B x 1 x H x W
    Output:
      mask:        B x 1 x H x W
    """
    def __init__(self, img_channel=3, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channel + 1, hidden, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        # Start from an uncertain soft mask (approximately 0.5 everywhere).
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, x, high_energy):
        if high_energy.shape[-2:] != x.shape[-2:]:
            high_energy = F.interpolate(high_energy, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.net(torch.cat([x, high_energy], dim=1))


class DegradationPromptEncoder(nn.Module):
    """Encode image-level underwater degradation cues into a compact prompt.

    The prompt is intentionally lightweight. It summarizes color cast, contrast,
    brightness, and high-frequency energy so later fusion layers can adapt per
    input image instead of using one fixed low/high-frequency balance.
    """

    def __init__(self, img_channel=3, prompt_dim=16, hidden=32):
        super().__init__()
        stat_dim = img_channel * 3 + 5
        self.mlp = nn.Sequential(
            nn.Conv2d(stat_dim, hidden, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, prompt_dim, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x, high_energy):
        if high_energy.shape[-2:] != x.shape[-2:]:
            high_energy = F.interpolate(high_energy, size=x.shape[-2:], mode='bilinear', align_corners=False)

        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True, unbiased=False)
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        contrast = (x_max - x_min).mean(dim=1, keepdim=True)
        brightness = mean.mean(dim=1, keepdim=True)
        rg_ratio = mean[:, 0:1] / (mean[:, 1:2] + 1e-6)
        bg_ratio = mean[:, 2:3] / (mean[:, 1:2] + 1e-6)
        hf_mean = high_energy.mean(dim=(2, 3), keepdim=True)
        stats = torch.cat([mean, std, x_max - x_min, brightness, contrast, rg_ratio, bg_ratio, hf_mean], dim=1)
        return self.mlp(stats)


class ResidualMultiplierHighGate(nn.Module):
    """SRDRM-style multiplicative gate for wavelet high-frequency features.

    The module predicts a bounded multiplier from low-frequency structure,
    high-frequency texture, and the soft foreground/background mask. It starts
    close to identity, then learns to amplify high-frequency features mainly in
    structure/detail regions while suppressing noisy background texture.
    """

    def __init__(self, channels, max_gain=0.20, hidden_ratio=2, init_bias=0.0):
        super().__init__()
        hidden = max(channels // hidden_ratio, 8)
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, hidden, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.constant_(self.net[-2].bias, float(init_bias))
        self.max_gain = nn.Parameter(torch.tensor(float(max_gain)))

    def forward(self, f_low, f_high, mask=None):
        if mask is None:
            mask = f_high.new_ones(f_high.shape[0], 1, f_high.shape[2], f_high.shape[3]) * 0.5
        if mask.shape[-2:] != f_high.shape[-2:]:
            mask = F.interpolate(mask, size=f_high.shape[-2:], mode='bilinear', align_corners=False)
        if f_low.shape[-2:] != f_high.shape[-2:]:
            f_low = F.interpolate(f_low, size=f_high.shape[-2:], mode='bilinear', align_corners=False)

        mask = mask.clamp(0.0, 1.0)
        gate = self.net(torch.cat([f_low, f_high, mask], dim=1))
        gain = self.max_gain.clamp(0.0, 0.5)
        multiplier = 1.0 + gain * (2.0 * gate - 1.0) * mask
        return f_high * multiplier


class RegionAwareFrequencyFusion(nn.Module):
    """Soft foreground/background guided low-/high-frequency fusion.

    Compared with plain frequency fusion, this module explicitly constructs four
    region-frequency components:
      Low-Foreground, Low-Background, High-Foreground, High-Background.

    The soft mask M controls foreground/background weighting:
      LF = M * F_low,       LB = (1 - M) * F_low
      HF = M * F_high,      HB = (1 - M) * F_high

    Foreground regions are encouraged to keep more high-frequency details, while
    background regions are encouraged to preserve low-frequency stability and
    suppress water/noise artifacts. The final fusion is still learnable, so this
    is not a hard rule.
    """
    def __init__(self, channels, high_boost=0.35, region_boost=0.25):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        # Input channels: F_low, F_high, low_fg, low_bg, high_fg, high_bg, M
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 6 + 1, channels * 2, 1, 1, 0, bias=True),
            SimpleGate(),
            NAFBlock(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
        )
        self.high_boost = nn.Parameter(torch.tensor(float(high_boost)))
        self.region_boost = nn.Parameter(torch.tensor(float(region_boost)))

    def forward(self, f_low, f_high, mask):
        if mask.shape[-2:] != f_low.shape[-2:]:
            mask = F.interpolate(mask, size=f_low.shape[-2:], mode='bilinear', align_corners=False)
        mask = mask.clamp(0.0, 1.0)

        low_fg = mask * f_low
        low_bg = (1.0 - mask) * f_low
        high_fg = mask * f_high
        high_bg = (1.0 - mask) * f_high

        region_cat = torch.cat([f_low, f_high, low_fg, low_bg, high_fg, high_bg, mask], dim=1)
        fused = self.fuse(region_cat)

        gate = self.gate(torch.cat([f_low, f_high, mask], dim=1))
        base = gate * f_low + (1.0 - gate) * f_high
        region_prior = mask * f_high + (1.0 - mask) * f_low

        return fused + base + self.region_boost * region_prior + self.high_boost * high_fg


class PromptGuidedRegionAwareFrequencyFusion(nn.Module):
    """Region-aware frequency fusion conditioned by degradation prompt."""

    def __init__(self, channels, prompt_dim=16, high_boost=0.35, region_boost=0.25):
        super().__init__()
        self.base_fusion = RegionAwareFrequencyFusion(
            channels,
            high_boost=high_boost,
            region_boost=region_boost,
        )
        self.prompt_gate = nn.Sequential(
            nn.Conv2d(prompt_dim, channels * 2, 1, 1, 0, bias=True),
            SimpleGate(),
            nn.Conv2d(channels, channels * 2, 1, 1, 0, bias=True),
        )
        self.prompt_mask = nn.Sequential(
            nn.Conv2d(prompt_dim, channels, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.prompt_gate[-1].weight)
        nn.init.zeros_(self.prompt_gate[-1].bias)
        nn.init.zeros_(self.prompt_mask[-1].weight)
        nn.init.zeros_(self.prompt_mask[-1].bias)

    def forward(self, f_low, f_high, mask, prompt):
        if prompt.shape[-2:] != f_low.shape[-2:]:
            prompt = prompt.expand(-1, -1, f_low.shape[-2], f_low.shape[-1])

        prompt_bias = self.prompt_gate(prompt)
        low_bias, high_bias = prompt_bias.chunk(2, dim=1)
        f_low = f_low * (1.0 + 0.10 * torch.tanh(low_bias))
        f_high = f_high * (1.0 + 0.10 * torch.tanh(high_bias))
        mask = (mask + 0.10 * torch.tanh(self.prompt_mask(prompt))).clamp(0.0, 1.0)
        return self.base_fusion(f_low, f_high, mask)


class FrequencyFusion(nn.Module):
    """
    Frequency fusion biased toward high-frequency detail.

    The old model often let low-frequency features dominate, producing a soft image.
    This module keeps an explicit high-frequency residual and uses a learnable scale.
    """
    def __init__(self, channels, high_boost=0.35):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=True),
            NAFBlock(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
        )
        self.high_boost = nn.Parameter(torch.tensor(float(high_boost)))

    def forward(self, f_low, f_high):
        cat = torch.cat([f_low, f_high], dim=1)
        gate = self.gate(cat)
        fused = self.fuse(cat)
        return fused + gate * f_low + (1.0 - gate) * f_high + self.high_boost * f_high


class RGBColorStabilizer(nn.Module):
    """
    Bounded RGB-space color stabilization.

    Important: color correction is done after RGB reconstruction, not in feature space.
    The last layer is zero-initialized, so the module starts as identity and only learns
    a small bounded residual during training. This reduces red over-compensation.
    """
    def __init__(self, img_channel=3, hidden=16, max_residual=0.04):
        super().__init__()
        self.max_residual = float(max_residual)
        self.net = nn.Sequential(
            nn.Conv2d(img_channel, hidden, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, img_channel, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.max_residual * torch.tanh(self.net(x))


class LowRankFeatureAdapter(nn.Module):
    """Zero-initialized bottleneck adapter for conservative domain adaptation."""

    def __init__(self, channels, rank=8, scale=0.1):
        super().__init__()
        rank = max(1, int(rank))
        self.down = nn.Conv2d(channels, rank, 1, 1, 0, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(rank, channels, 1, 1, 0, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.scale * self.up(self.act(self.down(x)))


class HRSRHead(nn.Module):
    """SR reconstruction head with explicit HR-space refinement.

    The previous SR head converted LR features directly to an RGB residual with
    Conv + PixelShuffle. That is efficient, but it gives the network no chance
    to refine edges/textures after upsampling. This head first upsamples to HR
    feature space, then applies several NAFBlocks at HR resolution, and finally
    predicts the RGB residual. This is designed to strengthen real SR detail
    reconstruction instead of relying mainly on the interpolated base image.
    """

    def __init__(self, channels, img_channel=3, up_scale=2, num_hr_blocks=6,
                 detail_scale=0.12, use_dense_refine=False, dense_growth=32,
                 use_edge_guided_detail=True, detail_gate_bias=-2.0,
                 use_structure_adapter=False, structure_adapter_hidden=32,
                 structure_adapter_scale=0.10):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * up_scale * up_scale, 3, 1, 1, bias=True),
            nn.PixelShuffle(up_scale),
        )
        self.hr_refine = nn.Sequential(*[NAFBlock(channels) for _ in range(num_hr_blocks)])
        self.dense_refine = (
            ResidualDenseStack(
                channels,
                num_blocks=max(1, num_hr_blocks // 2),
                growth_channels=dense_growth,
                residual_scale=0.15,
            )
            if use_dense_refine
            else nn.Identity()
        )
        self.out = nn.Conv2d(channels, img_channel, 3, 1, 1, bias=True)

        # A very small HR detail residual path. The last layer starts from zero,
        # so the branch will not create strong fake texture at the beginning.
        self.detail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=True),
            nn.Conv2d(channels, channels * 2, 1, 1, 0, bias=True),
            SimpleGate(),
            nn.Conv2d(channels, img_channel, 3, 1, 1, bias=True),
        )
        nn.init.zeros_(self.detail[-1].weight)
        nn.init.zeros_(self.detail[-1].bias)
        self.use_edge_guided_detail = bool(use_edge_guided_detail)
        gate_hidden = max(channels // 2, 8)
        self.detail_gate = nn.Sequential(
            nn.Conv2d(channels, gate_hidden, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.detail_gate[2].weight)
        nn.init.constant_(self.detail_gate[2].bias, float(detail_gate_bias))
        self.detail_scale = nn.Parameter(torch.tensor(float(detail_scale)))
        self.structure_adapter = (
            StructureGuidedDetailAdapter(
                channels=channels,
                img_channel=img_channel,
                hidden=structure_adapter_hidden,
                scale=structure_adapter_scale,
            )
            if use_structure_adapter
            else None
        )

    def forward(self, x, guide=None):
        hr_feat = self.up(x)
        hr_feat = self.hr_refine(hr_feat)
        hr_feat = self.dense_refine(hr_feat)
        adapter_res = None
        if self.structure_adapter is not None and guide is not None:
            hr_feat, adapter_res = self.structure_adapter(hr_feat, guide)
        base_res = self.out(hr_feat)
        detail = self.detail(hr_feat)
        if self.use_edge_guided_detail:
            detail = self.detail_gate(hr_feat) * detail
        detail_res = self.detail_scale * detail
        if adapter_res is not None:
            detail_res = detail_res + adapter_res
        return base_res + detail_res


class StructureGuidedDetailAdapter(nn.Module):
    """Zero-init adapter guided by bicubic/Sobel/Laplacian structure.

    It borrows the useful part of ControlNet-LLLite for SR: a lightweight
    conditioning branch that starts as identity and only learns a small,
    structure-aligned HR detail residual. This keeps the main SR path stable.
    """

    def __init__(self, channels, img_channel=3, hidden=32, scale=0.10):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.register_buffer(
            'sobel_x',
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            'sobel_y',
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            'lap',
            torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
        )
        in_ch = img_channel + 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.aspp1 = nn.Conv2d(hidden, hidden, 3, 1, 1, dilation=1, bias=True)
        self.aspp2 = nn.Conv2d(hidden, hidden, 3, 1, 2, dilation=2, bias=True)
        self.aspp4 = nn.Conv2d(hidden, hidden, 3, 1, 4, dilation=4, bias=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
        )
        self.film = nn.Conv2d(hidden, channels * 2, 1, 1, 0, bias=True)
        self.detail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=True),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, img_channel, 3, 1, 1, bias=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(hidden, 1, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.detail[-1].weight)
        nn.init.zeros_(self.detail[-1].bias)
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, -1.5)

    def _gray(self, x):
        if x.shape[1] != 3:
            return x.mean(dim=1, keepdim=True)
        weight = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x * weight).sum(dim=1, keepdim=True)

    def _structure(self, guide):
        guide = guide.clamp(0.0, 1.0)
        gray = self._gray(guide)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
        lap = torch.abs(F.conv2d(gray, self.lap, padding=1))
        edge = edge / (edge.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        lap = lap / (lap.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return torch.cat([guide, edge, lap], dim=1)

    def forward(self, feat, guide):
        cond = self._structure(guide)
        cond = self.stem(cond)
        cond = self.fuse(torch.cat([
            F.relu(self.aspp1(cond), inplace=True),
            F.relu(self.aspp2(cond), inplace=True),
            F.relu(self.aspp4(cond), inplace=True),
        ], dim=1))
        gamma, beta = self.film(cond).chunk(2, dim=1)
        scale = self.scale.clamp(0.0, 0.3)
        feat = feat * (1.0 + scale * torch.tanh(gamma)) + scale * beta
        detail = scale * self.gate(cond) * self.detail(feat)
        return feat, detail


class DetailRefineHead(nn.Module):
    """Extra RGB residual head for sharper edges and textures."""
    def __init__(self, channels, img_channel, up_scale):
        super().__init__()
        self.body = nn.Sequential(
            NAFBlock(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.Conv2d(channels, img_channel * up_scale * up_scale, 3, 1, 1, bias=True),
            nn.PixelShuffle(up_scale),
        )
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        return self.scale * self.body(x)


class EnhancementHead(nn.Module):
    """Very weak LR-space enhancement head for joint enhancement + SR.

    This branch is deliberately lightweight. It provides a joint enhancement
    output for supervision, but the training config gives it a very small loss
    weight so it will not dominate the SR branch or introduce whole-image
    color style shifts.
    """
    def __init__(self, channels, img_channel=3, residual_scale=0.01):
        super().__init__()
        self.body = nn.Sequential(
            NAFBlock(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.Conv2d(channels, img_channel, 3, 1, 1, bias=True),
        )
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, feat, inp):
        return inp + self.scale * self.body(feat)


class EnhanceFeedbackFusion(nn.Module):
    """Fuse LR enhancement cues back into the SR feature stream."""

    def __init__(self, channels, img_channel=3, hidden=None, scale=0.08):
        super().__init__()
        hidden = hidden or channels
        self.body = nn.Sequential(
            nn.Conv2d(img_channel * 2, hidden, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Conv2d(hidden, channels, 3, 1, 1, bias=True),
            NAFBlock(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, feat, enhance, inp):
        residual = enhance - inp
        cue = torch.cat([enhance, residual], dim=1)
        feedback = self.body(cue)
        return feat + self.scale.clamp(0.0, 0.30) * self.gate(feat) * feedback


class RFDBlock(nn.Module):
    """Residual feature distillation block for SR-oriented local structure."""

    def __init__(self, channels, distill_ratio=0.25, residual_scale=0.2):
        super().__init__()
        distilled = max(8, int(channels * distill_ratio))
        remaining = channels - distilled
        self.c1_d = nn.Conv2d(channels, distilled, 1, 1, 0, bias=True)
        self.c1_r = nn.Conv2d(channels, remaining, 3, 1, 1, bias=True)
        self.c2_d = nn.Conv2d(remaining, distilled, 1, 1, 0, bias=True)
        self.c2_r = nn.Conv2d(remaining, remaining, 3, 1, 1, bias=True)
        self.c3_d = nn.Conv2d(remaining, distilled, 1, 1, 0, bias=True)
        self.c3_r = nn.Conv2d(remaining, remaining, 3, 1, 1, bias=True)
        self.c4 = nn.Conv2d(remaining, distilled, 3, 1, 1, bias=True)
        self.fuse = nn.Conv2d(distilled * 4, channels, 1, 1, 0, bias=True)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 8), 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.LeakyReLU(0.05, inplace=True)
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x):
        d1 = self.act(self.c1_d(x))
        r1 = self.act(self.c1_r(x))
        d2 = self.act(self.c2_d(r1))
        r2 = self.act(self.c2_r(r1))
        d3 = self.act(self.c3_d(r2))
        r3 = self.act(self.c3_r(r2))
        d4 = self.act(self.c4(r3))
        out = self.fuse(torch.cat([d1, d2, d3, d4], dim=1))
        out = out * self.attn(out)
        return x + self.scale * out


class UWNAFRFDNSR(nn.Module):
    """UFO-120 oriented SR backbone with wavelet high-frequency injection.

    This avoids the encoder-decoder downsampling path and follows a more
    standard SR design: shallow features -> RFDB trunk -> PixelShuffle residual.
    The wavelet branch injects LR-resolution high-frequency cues as auxiliary
    guidance instead of becoming the main reconstruction path.
    """

    def __init__(
        self,
        img_channel=3,
        width=64,
        up_scale=2,
        num_blocks=8,
        distill_ratio=0.25,
        high_width=32,
        high_scale=0.20,
        rgb_color_residual=0.0,
        use_rgb_color_stabilizer=False,
        train_size=(1, 3, 128, 128),
        fast_imp=False,
    ):
        super().__init__()
        self.up_scale = up_scale
        self.dwt = HaarDWT2D()
        self.shallow = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.high_intro = nn.Sequential(
            nn.Conv2d(img_channel * 3, high_width, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Conv2d(high_width, width * 4, 3, 1, 1, bias=True),
            nn.PixelShuffle(2),
            nn.Conv2d(width, width, 3, 1, 1, bias=True),
        )
        self.high_scale = nn.Parameter(torch.tensor(float(high_scale)))
        self.blocks = nn.ModuleList([
            RFDBlock(width, distill_ratio=distill_ratio, residual_scale=0.2)
            for _ in range(num_blocks)
        ])
        self.trunk_fuse = nn.Conv2d(width * num_blocks, width, 1, 1, 0, bias=True)
        self.trunk_out = nn.Conv2d(width, width, 3, 1, 1, bias=True)
        self.upsampler = nn.Sequential(
            nn.Conv2d(width, width * up_scale * up_scale, 3, 1, 1, bias=True),
            nn.PixelShuffle(up_scale),
            nn.Conv2d(width, img_channel, 3, 1, 1, bias=True),
        )
        self.color_stabilizer = RGBColorStabilizer(
            img_channel=img_channel,
            max_residual=rgb_color_residual,
        ) if use_rgb_color_stabilizer else nn.Identity()

    def forward(self, inp):
        b, c, h, w = inp.shape
        base = F.interpolate(inp, scale_factor=self.up_scale, mode='bicubic', align_corners=False)
        feat = self.shallow(inp)

        ll, lh, hl, hh = self.dwt(inp)
        high = self.high_intro(torch.cat([lh, hl, hh], dim=1))
        high = high[:, :, :h, :w]
        feat = feat + self.high_scale.clamp(0.0, 0.5) * high

        outs = []
        x = feat
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        trunk = self.trunk_out(self.trunk_fuse(torch.cat(outs, dim=1)))
        sr = self.upsampler(feat + trunk)
        out = self.color_stabilizer(base + sr)
        return out[:, :, :h * self.up_scale, :w * self.up_scale]


class UWNAFRFDNSRLocal(Local_Base, UWNAFRFDNSR):
    def __init__(self, *args, train_size=(1, 3, 128, 128), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        UWNAFRFDNSR.__init__(self, *args, train_size=train_size, fast_imp=fast_imp, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class UWNAFWaveCCSR(nn.Module):
    """
    Underwater SR network, tuned for the issues seen in the current results:
      1) remove feature-space color compensation to avoid red cast;
      2) strengthen wavelet high-frequency branch for sharper edges;
      3) apply only bounded RGB-space color stabilization;
      4) use HR-space refinement in the SR head for sharper reconstruction.
    """
    def __init__(
        self,
        img_channel=3,
        width=32,
        up_scale=2,
        low_blk_num=2,
        high_blk_num=5,
        middle_blk_num=2,
        enc_blk_nums=(1, 1, 2, 4),
        dec_blk_nums=(1, 1, 1, 1),
        color_comp_strength=0.0,   # kept for config compatibility; no feature color comp is used
        high_boost=0.35,
        sr_res_scale=0.8,
        use_bicubic_residual=True,
        rgb_color_residual=0.005,
        use_rgb_color_stabilizer=True,
        use_enhance_head=True,
        return_enhance=True,
        enhance_residual_scale=0.01,
        use_enhance_feedback=False,
        enhance_feedback_scale=0.08,
        enhance_feedback_hidden=0,
        use_original_naf_input=False,
        use_wavelet_frontend=True,
        use_high_frequency_enhancer=True,
        use_wavelet_attention=True,
        wavelet_attention_blocks=2,
        wavelet_attention_scale=0.08,
        use_region_fusion=True,
        mask_hidden=16,
        return_mask=False,
        use_prompt_fusion=False,
        prompt_dim=16,
        prompt_hidden=32,
        use_low_rank_adapters=False,
        adapter_rank=8,
        adapter_scale=0.1,
        use_local_dense_trunk=False,
        local_dense_blocks=6,
        local_dense_growth=32,
        use_hr_sr_head=True,
        hr_refine_blocks=6,
        hr_detail_scale=0.12,
        use_dense_high_branch=True,
        high_dense_blocks=3,
        dense_growth=32,
        use_dense_hr_refine=True,
        use_edge_guided_detail=True,
        detail_gate_bias=-2.0,
        use_structure_adapter=False,
        structure_adapter_hidden=32,
        structure_adapter_scale=0.10,
        use_residual_multiplier=True,
        residual_multiplier_gain=0.20,
        residual_multiplier_bias=0.0,
        use_pyramid_context=False,
        pyramid_context_scale=0.08,
        use_msca_refine=False,
        msca_refine_scale=0.06,
        use_fft_refine=False,
        fft_refine_scale=0.08,
        fft_hidden_scale=1.0,
    ):
        super().__init__()
        self.up_scale = up_scale
        self.padder_size = 2 ** len(enc_blk_nums)
        self.use_bicubic_residual = bool(use_bicubic_residual)
        self.use_original_naf_input = bool(use_original_naf_input)
        self.use_wavelet_frontend = bool(use_wavelet_frontend)
        self.dwt = HaarDWT2D() if self.use_wavelet_frontend else None
        self.use_enhance_head = bool(use_enhance_head)
        self.use_high_frequency_enhancer = bool(use_high_frequency_enhancer)
        self.return_enhance = bool(return_enhance)
        self.use_enhance_feedback = bool(use_enhance_feedback)
        self.return_mask = bool(return_mask)
        self.use_region_fusion = bool(use_region_fusion)
        self.use_residual_multiplier = bool(use_residual_multiplier)
        self.use_prompt_fusion = bool(use_prompt_fusion)
        self.use_low_rank_adapters = bool(use_low_rank_adapters)
        self.use_local_dense_trunk = bool(use_local_dense_trunk)
        self.use_hr_sr_head = bool(use_hr_sr_head)
        self.use_pyramid_context = bool(use_pyramid_context)
        self.use_msca_refine = bool(use_msca_refine)
        self.use_fft_refine = bool(use_fft_refine)

        # A0 ablation path: keep the same NAF encoder-decoder and reconstruction
        # head, but replace the complete wavelet frontend with one RGB projection.
        self.baseline_intro = (
            nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
            if not self.use_wavelet_frontend
            else None
        )

        # Low-frequency branch: structure/context only. No feature-space color correction.
        self.low_intro = (
            nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
            if self.use_wavelet_frontend
            else None
        )
        self.low_body = NAFStack(width, low_blk_num) if self.use_wavelet_frontend else None

        # High-frequency branch: texture/edge recovery.
        self.high_intro = (
            nn.Conv2d(img_channel * 3, width, 3, 1, 1, bias=True)
            if self.use_wavelet_frontend
            else None
        )
        self.high_body = NAFStack(width, high_blk_num) if self.use_wavelet_frontend else None
        self.high_dense_refine = (
            ResidualDenseStack(
                width,
                num_blocks=high_dense_blocks,
                growth_channels=dense_growth,
                residual_scale=0.15,
            )
            if self.use_wavelet_frontend and use_dense_high_branch
            else (nn.Identity() if self.use_wavelet_frontend else None)
        )
        self.high_enhance = (
            HighFrequencyEnhancer(width)
            if self.use_wavelet_frontend and self.use_high_frequency_enhancer
            else (nn.Identity() if self.use_wavelet_frontend else None)
        )

        # Learnable upsampling before low/high fusion.
        # Low branch keeps bilinear shortcut for stable structure; high branch uses
        # nearest shortcut to avoid smoothing wavelet detail features too early.
        self.low_up = (
            FeatureUpsample(width, base_mode='bilinear', residual_scale=0.2)
            if self.use_wavelet_frontend
            else None
        )
        self.high_up = (
            FeatureUpsample(width, base_mode='nearest', residual_scale=0.2)
            if self.use_wavelet_frontend
            else None
        )

        self.high_multiplier = ResidualMultiplierHighGate(
            width,
            max_gain=residual_multiplier_gain,
            init_bias=residual_multiplier_bias,
        ) if self.use_wavelet_frontend and self.use_residual_multiplier else None

        self.mask_predictor = SoftForegroundBackgroundMask(
            img_channel=img_channel,
            hidden=mask_hidden,
        ) if self.use_wavelet_frontend and self.use_region_fusion else None
        self.prompt_encoder = DegradationPromptEncoder(
            img_channel=img_channel,
            prompt_dim=prompt_dim,
            hidden=prompt_hidden,
        ) if self.use_wavelet_frontend and self.use_prompt_fusion else None
        if not self.use_wavelet_frontend:
            self.freq_fusion = None
        elif self.use_region_fusion and self.use_prompt_fusion:
            self.freq_fusion = PromptGuidedRegionAwareFrequencyFusion(
                width,
                prompt_dim=prompt_dim,
                high_boost=high_boost,
            )
        elif self.use_region_fusion:
            self.freq_fusion = RegionAwareFrequencyFusion(width, high_boost=high_boost)
        else:
            self.freq_fusion = FrequencyFusion(width, high_boost=high_boost)
        self.wavelet_refine = MultiLevelWaveletRefinement(
            width,
            num_blocks=wavelet_attention_blocks,
            residual_scale=wavelet_attention_scale,
        ) if self.use_wavelet_frontend and use_wavelet_attention else nn.Identity()
        self.fft_refine = FFTFeatureRefineBlock(
            width,
            hidden_scale=fft_hidden_scale,
            scale=fft_refine_scale,
        ) if self.use_fft_refine else nn.Identity()
        self.pyramid_context = PyramidContextRefine(
            width,
            img_channel=img_channel,
            scale=pyramid_context_scale,
        ) if self.use_pyramid_context else nn.Identity()
        self.msca_refine = MSCASpatialRefine(
            width,
            scale=msca_refine_scale,
        ) if self.use_msca_refine else nn.Identity()
        self.fusion_adapter = LowRankFeatureAdapter(
            width,
            rank=adapter_rank,
            scale=adapter_scale,
        ) if self.use_low_rank_adapters else nn.Identity()
        self.intro = (
            nn.Identity()
            if self.use_original_naf_input and not self.use_wavelet_frontend
            else nn.Conv2d(width, width, 3, 1, 1, bias=True)
        )
        self.enhance_head = EnhancementHead(width, img_channel, residual_scale=enhance_residual_scale) if self.use_enhance_head else None
        self.enhance_feedback = (
            EnhanceFeedbackFusion(
                width,
                img_channel=img_channel,
                hidden=enhance_feedback_hidden or width,
                scale=enhance_feedback_scale,
            )
            if self.use_enhance_head and self.use_enhance_feedback
            else nn.Identity()
        )
        self.local_dense_trunk = (
            ResidualDenseStack(
                width,
                num_blocks=local_dense_blocks,
                growth_channels=local_dense_growth,
                residual_scale=0.12,
            )
            if self.use_local_dense_trunk
            else nn.Identity()
        )
        self.local_trunk_scale = nn.Parameter(torch.tensor(0.5)) if self.use_local_dense_trunk else None

        # NAFNet-style encoder-decoder backbone.
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.ending = (
            nn.Conv2d(width, width, 3, 1, 1, bias=True)
            if self.use_hr_sr_head
            else None
        )
        self.recon_adapter = LowRankFeatureAdapter(
            width,
            rank=adapter_rank,
            scale=adapter_scale,
        ) if self.use_hr_sr_head and self.use_low_rank_adapters else nn.Identity()
        # HR-space SR reconstruction head. It upsamples features first, refines
        # them at HR resolution, then predicts the RGB SR residual.
        self.sr_head = (
            HRSRHead(
                channels=width,
                img_channel=img_channel,
                up_scale=up_scale,
                num_hr_blocks=hr_refine_blocks,
                detail_scale=hr_detail_scale,
                use_dense_refine=use_dense_hr_refine,
                dense_growth=dense_growth,
                use_edge_guided_detail=use_edge_guided_detail,
                detail_gate_bias=detail_gate_bias,
                use_structure_adapter=use_structure_adapter,
                structure_adapter_hidden=structure_adapter_hidden,
                structure_adapter_scale=structure_adapter_scale,
            )
            if self.use_hr_sr_head
            else None
        )
        self.simple_sr_head = (
            nn.Sequential(
                nn.Conv2d(width, img_channel * up_scale * up_scale, 3, 1, 1, bias=True),
                nn.PixelShuffle(up_scale),
            )
            if not self.use_hr_sr_head
            else None
        )
        # Keep SR residual strength fixed. If this is learnable, training can
        # shrink it and make the output too close to the interpolated base image.
        self.register_buffer('sr_res_scale', torch.tensor(float(sr_res_scale)))

        self.color_stabilizer = RGBColorStabilizer(
            img_channel=img_channel,
            max_residual=rgb_color_residual,
        ) if use_rgb_color_stabilizer else nn.Identity()

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='replicate')
        return x

    def forward(self, inp):
        b, c, h, w = inp.shape
        inp_pad = self.check_image_size(inp)
        # Bicubic is a stronger SR base than bilinear. The network predicts a
        # residual on top of this base instead of relying on a blurry bilinear image.
        inp_up = F.interpolate(inp_pad, scale_factor=self.up_scale, mode='bicubic', align_corners=False)

        soft_mask = None
        if self.use_wavelet_frontend:
            # Wavelet dual branch.
            ll, lh, hl, hh = self.dwt(inp_pad)
            high = torch.cat([lh, hl, hh], dim=1)
            # A one-channel high-frequency energy map for soft foreground/background mask prediction.
            high_energy = torch.mean(torch.abs(high), dim=1, keepdim=True)

            f_low = self.low_intro(ll)
            f_low = self.low_body(f_low)
            f_low = self.low_up(f_low)

            f_high = self.high_intro(high)
            f_high = self.high_body(f_high)
            f_high = self.high_dense_refine(f_high)
            f_high = self.high_enhance(f_high)
            f_high = self.high_up(f_high)

            if self.mask_predictor is not None:
                soft_mask = self.mask_predictor(inp_pad, high_energy)
                prompt = self.prompt_encoder(inp_pad, high_energy) if self.prompt_encoder is not None else None
                if self.high_multiplier is not None:
                    f_high = self.high_multiplier(f_low, f_high, soft_mask)
                if prompt is not None:
                    feat = self.freq_fusion(f_low, f_high, soft_mask, prompt)
                else:
                    feat = self.freq_fusion(f_low, f_high, soft_mask)
            else:
                if self.high_multiplier is not None:
                    f_high = self.high_multiplier(f_low, f_high)
                feat = self.freq_fusion(f_low, f_high)
        else:
            feat = self.baseline_intro(inp_pad)

        feat = self.wavelet_refine(feat)
        feat = self.fft_refine(feat)
        if self.use_pyramid_context:
            feat = self.pyramid_context(feat, inp_pad)
        feat = self.msca_refine(feat)
        feat = self.fusion_adapter(feat)
        x = self.intro(feat)
        local_feat = self.local_dense_trunk(x)

        enhance_out = None
        if self.enhance_head is not None:
            enhance_full = self.enhance_head(x, inp_pad)
            if self.use_enhance_feedback:
                x = self.enhance_feedback(x, enhance_full, inp_pad)
            enhance_out = enhance_full[:, :, :h, :w]

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        if self.use_hr_sr_head:
            feat_out = self.ending(x)
            if self.local_trunk_scale is not None:
                feat_out = feat_out + self.local_trunk_scale.clamp(0.0, 1.0) * local_feat
            feat_out = self.recon_adapter(feat_out)
            sr_residual = self.sr_head(feat_out, guide=inp_up)
        else:
            sr_residual = self.simple_sr_head(x)
        out = self.sr_res_scale * sr_residual
        if self.use_bicubic_residual:
            out = inp_up + out
        out = self.color_stabilizer(out)
        out = out[:, :, :h * self.up_scale, :w * self.up_scale]

        if self.training and self.return_enhance and enhance_out is not None:
            outputs = [enhance_out, out]
            if self.return_mask and soft_mask is not None:
                outputs.append(soft_mask[:, :, :h, :w])
            return outputs
        if self.return_mask and soft_mask is not None:
            return out, soft_mask[:, :, :h, :w]
        return out


# Optional losses. Put these into your loss config or copy to your loss file.
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('kernel', kernel)

    def laplacian(self, x):
        c = x.shape[1]
        weight = self.kernel.repeat(c, 1, 1, 1)
        return F.conv2d(x, weight, padding=1, groups=c)

    def forward(self, pred, target):
        return F.l1_loss(self.laplacian(pred), self.laplacian(target))


class FFTLoss(nn.Module):
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class DWTHighFrequencyLoss(nn.Module):
    """High-frequency wavelet loss for sharper SR details.

    It compares the LH/HL/HH subbands of prediction and target. This is more
    directly aligned with the wavelet branch than FFT-only supervision.
    """
    def __init__(self):
        super().__init__()
        self.dwt = HaarDWT2D()

    def forward(self, pred, target):
        _, p_lh, p_hl, p_hh = self.dwt(pred)
        _, t_lh, t_hl, t_hh = self.dwt(target)
        return (F.l1_loss(p_lh, t_lh) +
                F.l1_loss(p_hl, t_hl) +
                F.l1_loss(p_hh, t_hh))


class MaskSmoothnessLoss(nn.Module):
    """Optional smoothness regularization for the predicted soft mask."""
    def forward(self, mask):
        dx = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]).mean()
        dy = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]).mean()
        return dx + dy


class UWSRLoss(nn.Module):
    """Recommended start: L = Charbonnier + Edge + FFT + DWT-high.

    If the output is still soft, try increasing edge_weight or dwt_weight first.
    """
    def __init__(self, edge_weight=0.1, fft_weight=0.01, dwt_weight=0.05):
        super().__init__()
        self.pixel = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.fft = FFTLoss()
        self.dwt_high = DWTHighFrequencyLoss()
        self.edge_weight = edge_weight
        self.fft_weight = fft_weight
        self.dwt_weight = dwt_weight

    def forward(self, pred, target):
        return (self.pixel(pred, target) +
                self.edge_weight * self.edge(pred, target) +
                self.fft_weight * self.fft(pred, target) +
                self.dwt_weight * self.dwt_high(pred, target))


class UWNAFWaveCCSRLocal(Local_Base, UWNAFWaveCCSR):
    def __init__(self, *args, train_size=(1, 3, 64, 64), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        UWNAFWaveCCSR.__init__(self, *args, **kwargs)

        n, c, h, w = train_size
        base_size = (int(h * 1.5), int(w * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    net = UWNAFWaveCCSR(img_channel=3, width=32, up_scale=2)
    x = torch.randn(1, 3, 64, 64)
    y = net(x)
    print('input:', x.shape)
    if isinstance(y, list):
        print('enhance:', y[0].shape)
        print('output:', y[-1].shape)
    else:
        print('output:', y.shape)
