import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.arch_util import LayerNorm2d


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
        x = inp
        x = self.norm1(x)
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
    def forward(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh


class FeatureUpsample(nn.Module):
    def __init__(self, channels, mode='bilinear'):
        super().__init__()
        self.mode = mode
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, 1, 1, bias=True),
            nn.PixelShuffle(2),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=2, mode=self.mode, align_corners=False if self.mode == 'bilinear' else None)
        return base + self.scale.clamp(0.0, 0.3) * self.refine(x)


class FrequencyFusion(nn.Module):
    def __init__(self, channels, high_boost=0.25):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=True),
            NAFBlock(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
        )
        self.high_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.high_boost = nn.Parameter(torch.tensor(float(high_boost)))

    def forward(self, low, high):
        cat = torch.cat([low, high], dim=1)
        fused = self.fuse(cat)
        return fused + self.high_boost.clamp(0.0, 0.6) * self.high_gate(cat) * high


class EnhanceHead(nn.Module):
    def __init__(self, channels, img_channel=3, residual_scale=0.08):
        super().__init__()
        self.body = nn.Sequential(
            NAFBlock(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.Conv2d(channels, img_channel, 3, 1, 1, bias=True),
        )
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, feat, inp):
        return inp + self.scale.clamp(0.0, 0.3) * self.body(feat)


class UWEnhance(nn.Module):
    """Isolated scale-1 underwater enhancement network.

    This network is intentionally separate from UWNAFWaveCCSR. It has no SR
    upsampling head and is meant only for same-size enhancement training.
    """

    def __init__(
        self,
        img_channel=3,
        width=40,
        low_blk_num=3,
        high_blk_num=4,
        middle_blk_num=3,
        enc_blk_nums=(1, 2, 2, 3),
        dec_blk_nums=(1, 1, 1, 2),
        high_boost=0.25,
        enhance_residual_scale=0.08,
    ):
        super().__init__()
        self.padder_size = 2 ** len(enc_blk_nums)
        self.dwt = HaarDWT2D()

        self.low_intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.low_body = nn.Sequential(*[NAFBlock(width) for _ in range(low_blk_num)])
        self.low_up = FeatureUpsample(width, mode='bilinear')

        self.high_intro = nn.Conv2d(img_channel * 3, width, 3, 1, 1, bias=True)
        self.high_body = nn.Sequential(*[NAFBlock(width) for _ in range(high_blk_num)])
        self.high_up = FeatureUpsample(width, mode='bilinear')

        self.freq_fusion = FrequencyFusion(width, high_boost=high_boost)
        self.intro = nn.Conv2d(width, width, 3, 1, 1, bias=True)

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
                nn.PixelShuffle(2),
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.ending = nn.Conv2d(width, width, 3, 1, 1, bias=True)
        self.enhance_head = EnhanceHead(width, img_channel=img_channel, residual_scale=enhance_residual_scale)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='replicate')

    def forward(self, inp):
        _, _, h, w = inp.shape
        inp_pad = self.check_image_size(inp)

        ll, lh, hl, hh = self.dwt(inp_pad)
        high = torch.cat([lh, hl, hh], dim=1)

        low_feat = self.low_up(self.low_body(self.low_intro(ll)))
        high_feat = self.high_up(self.high_body(self.high_intro(high)))
        feat = self.intro(self.freq_fusion(low_feat, high_feat))

        encs = []
        x = feat
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        feat = self.ending(x)
        out = self.enhance_head(feat, inp_pad)
        return out[:, :, :h, :w]
