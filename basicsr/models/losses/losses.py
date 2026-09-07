import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from torchvision import models

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss."""

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss."""

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False
            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            pred, target = pred / 255., target / 255.
        assert len(pred.size()) == 4
        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class CharbonnierLoss(nn.Module):
    """Robust L1 loss. Better than plain L1 for SR/restoration stability."""

    def __init__(self, loss_weight=1.0, eps=1e-3, reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        loss = torch.sqrt((pred - target) ** 2 + self.eps ** 2)
        if weight is not None:
            loss = loss * weight
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return self.loss_weight * loss


class GradientLoss(nn.Module):
    """First-order gradient loss for sharper SR edges and local texture.

    EdgeLoss below uses a Laplacian operator. This loss complements it by
    matching horizontal and vertical first-order gradients directly, which is
    often more stable for SR detail recovery.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = float(loss_weight)

    def forward(self, pred, target, **kwargs):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
        return self.loss_weight * loss


class SobelSharpnessLoss(nn.Module):
    """Match Sobel gradient magnitude, following Deep SESR's sharpness idea."""

    def __init__(self, loss_weight=1.0, eps=1e-6):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)
        kernel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def gradient_magnitude(self, x):
        c = x.shape[1]
        weight_x = self.kernel_x.repeat(c, 1, 1, 1)
        weight_y = self.kernel_y.repeat(c, 1, 1, 1)
        grad_x = F.conv2d(x, weight_x, padding=1, groups=c)
        grad_y = F.conv2d(x, weight_y, padding=1, groups=c)
        return torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)

    def forward(self, pred, target, **kwargs):
        pred_grad = self.gradient_magnitude(pred)
        target_grad = self.gradient_magnitude(target)
        return self.loss_weight * F.l1_loss(pred_grad, target_grad)


class EdgeAwareHighFrequencyLoss(nn.Module):
    """Match details on GT edges while suppressing false texture in smooth areas."""

    def __init__(self, loss_weight=1.0, false_edge_weight=0.25, temperature=0.7, eps=1e-6):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.false_edge_weight = float(false_edge_weight)
        self.temperature = float(temperature)
        self.eps = float(eps)
        kernel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def gradient_magnitude(self, x):
        gray = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.kernel_x, padding=1)
        grad_y = F.conv2d(gray, self.kernel_y, padding=1)
        return torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)

    def target_edge_mask(self, target_grad):
        b = target_grad.shape[0]
        flat = target_grad.flatten(1)
        mean = flat.mean(dim=1).view(b, 1, 1, 1)
        std = flat.std(dim=1, unbiased=False).view(b, 1, 1, 1).clamp_min(self.eps)
        return torch.sigmoid((target_grad - mean) / (std * self.temperature)).detach()

    def forward(self, pred, target, **kwargs):
        pred_grad = self.gradient_magnitude(pred)
        target_grad = self.gradient_magnitude(target)
        edge_mask = self.target_edge_mask(target_grad)

        edge_norm = edge_mask.sum().clamp_min(self.eps)
        bg_mask = 1.0 - edge_mask
        bg_norm = bg_mask.sum().clamp_min(self.eps)

        edge_match = (torch.abs(pred_grad - target_grad) * edge_mask).sum() / edge_norm
        false_edge = (F.relu(pred_grad - target_grad.detach()) * bg_mask).sum() / bg_norm
        return self.loss_weight * (edge_match + self.false_edge_weight * false_edge)


class EdgeLoss(nn.Module):
    """Laplacian edge loss to sharpen boundaries and texture."""

    def __init__(self, loss_weight=1.0):
        super().__init__()
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel', kernel)
        self.loss_weight = float(loss_weight)

    def laplacian(self, x):
        c = x.shape[1]
        weight = self.kernel.repeat(c, 1, 1, 1)
        return F.conv2d(x, weight, padding=1, groups=c)

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * F.l1_loss(self.laplacian(pred), self.laplacian(target))


class FFTLoss(nn.Module):
    """Frequency-domain magnitude loss for high-frequency SR detail."""

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = float(loss_weight)

    def forward(self, pred, target, **kwargs):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        return self.loss_weight * F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class HaarDWT2D(nn.Module):
    """One-level fixed Haar DWT used only for loss supervision."""

    def forward(self, x):
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
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


class DWTHighFrequencyLoss(nn.Module):
    """L1 loss on Haar high-frequency subbands LH/HL/HH.

    This directly encourages sharper wavelet-domain detail recovery and is
    useful when pixel/Charbonnier losses make SR outputs visually smooth.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.dwt = HaarDWT2D()

    def forward(self, pred, target, **kwargs):
        _, p_lh, p_hl, p_hh = self.dwt(pred)
        _, t_lh, t_hl, t_hh = self.dwt(target)
        loss = (F.l1_loss(p_lh, t_lh) +
                F.l1_loss(p_hl, t_hl) +
                F.l1_loss(p_hh, t_hh))
        return self.loss_weight * loss


class LaplacianPyramidLoss(nn.Module):
    """Multi-scale Laplacian residual loss for complex underwater textures."""

    def __init__(self, loss_weight=1.0, levels=3, weights=None):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.levels = int(levels)
        if weights is None:
            weights = [1.0, 0.5, 0.25]
        weights = torch.tensor(weights[:self.levels], dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-12)
        self.register_buffer('weights', weights)

        kernel_1d = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
        kernel_2d = (kernel_1d[:, None] @ kernel_1d[None, :])
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.register_buffer('kernel', kernel_2d.view(1, 1, 5, 5))

    def blur(self, x):
        c = x.shape[1]
        weight = self.kernel.repeat(c, 1, 1, 1)
        return F.conv2d(x, weight, padding=2, groups=c)

    def pyramid(self, x):
        pyr = []
        cur = x
        for _ in range(self.levels):
            if min(cur.shape[-2:]) < 8:
                break
            low = self.blur(cur)
            down = F.avg_pool2d(low, kernel_size=2, stride=2)
            up = F.interpolate(down, size=cur.shape[-2:], mode='bilinear', align_corners=False)
            pyr.append(cur - self.blur(up))
            cur = down
        return pyr

    def forward(self, pred, target, **kwargs):
        pred_pyr = self.pyramid(pred)
        target_pyr = self.pyramid(target)
        loss = pred.new_tensor(0.0)
        for idx, (p, t) in enumerate(zip(pred_pyr, target_pyr)):
            weight = self.weights[min(idx, self.weights.numel() - 1)]
            loss = loss + weight * F.l1_loss(p, t)
        return self.loss_weight * loss


class SoftCMIContrastLoss(nn.Module):
    """Saliency-style foreground/background contrast loss without mask labels.

    Deep SESR uses annotated saliency maps with CMI contrast supervision. This
    project does not have mask labels, so we derive a soft foreground prior from
    the target image's Sobel gradient energy and match foreground/background
    contrast on that stable prior.
    """

    def __init__(self, loss_weight=1.0, temperature=0.7, eps=1e-6):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.temperature = float(temperature)
        self.eps = float(eps)
        kernel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def soft_mask(self, target):
        gray = target.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.kernel_x, padding=1)
        grad_y = F.conv2d(gray, self.kernel_y, padding=1)
        energy = torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)
        mean = energy.mean(dim=(2, 3), keepdim=True)
        std = energy.std(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return torch.sigmoid((energy - mean) / (std * self.temperature))

    def cmi(self, image, mask):
        intensity = image.mean(dim=1, keepdim=True)
        fg = (intensity * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(self.eps)
        bg_mask = 1.0 - mask
        bg = (intensity * bg_mask).sum(dim=(2, 3)) / bg_mask.sum(dim=(2, 3)).clamp_min(self.eps)
        return (fg - bg) / (fg + bg + self.eps)

    def forward(self, pred, target, **kwargs):
        mask = self.soft_mask(target).detach()
        return self.loss_weight * F.l1_loss(self.cmi(pred, mask), self.cmi(target, mask))


class SSIMLoss(nn.Module):
    """Differentiable single-scale SSIM loss for structural consistency.

    Kept as a small auxiliary term. Too much SSIM loss can smooth textures,
    so the default weight in UWSRLoss is conservative.
    """

    def __init__(self, loss_weight=1.0, window_size=11, sigma=1.5, data_range=1.0):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.data_range = float(data_range)
        coords = torch.arange(self.window_size, dtype=torch.float32) - self.window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g = g / g.sum()
        kernel_2d = (g[:, None] @ g[None, :]).view(1, 1, self.window_size, self.window_size)
        self.register_buffer('kernel', kernel_2d)

    def forward(self, pred, target, **kwargs):
        c = pred.shape[1]
        weight = self.kernel.repeat(c, 1, 1, 1)
        pad = self.window_size // 2
        mu_x = F.conv2d(pred, weight, padding=pad, groups=c)
        mu_y = F.conv2d(target, weight, padding=pad, groups=c)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x = F.conv2d(pred * pred, weight, padding=pad, groups=c) - mu_x2
        sigma_y = F.conv2d(target * target, weight, padding=pad, groups=c) - mu_y2
        sigma_xy = F.conv2d(pred * target, weight, padding=pad, groups=c) - mu_xy
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2) + 1e-12)
        return self.loss_weight * (1.0 - ssim_map.mean())


class MSSSIMLoss(nn.Module):
    """Simple multi-scale SSIM loss.

    This follows the SwinWave-SR finding that structural similarity supervision
    helps PSNR/SSIM and texture recovery when combined with Charbonnier and
    gradient/edge losses. The implementation is intentionally lightweight.
    """

    def __init__(self, loss_weight=1.0, scales=3, weights=None):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.scales = int(scales)
        if weights is None:
            weights = [0.5, 0.3, 0.2]
        weights = torch.tensor(weights[:self.scales], dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-12)
        self.register_buffer('weights', weights)
        self.ssim = SSIMLoss(loss_weight=1.0)

    def forward(self, pred, target, **kwargs):
        loss = 0.0
        x, y = pred, target
        for i in range(self.scales):
            loss = loss + self.weights[i] * self.ssim(x, y)
            if i != self.scales - 1:
                if min(x.shape[-2:]) < 16 or min(y.shape[-2:]) < 16:
                    break
                x = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=False)
                y = F.avg_pool2d(y, kernel_size=2, stride=2, ceil_mode=False)
        return self.loss_weight * loss


class LumaMSSSIMLoss(MSSSIMLoss):
    """MS-SSIM on luminance, which is usually more stable for SR structure."""

    @staticmethod
    def rgb_to_luma(x):
        if x.shape[1] != 3:
            return x.mean(dim=1, keepdim=True)
        weight = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x * weight).sum(dim=1, keepdim=True)

    def forward(self, pred, target, **kwargs):
        return super().forward(self.rgb_to_luma(pred), self.rgb_to_luma(target), **kwargs)


class ExternalRGBSSIMLoss(nn.Module):
    """Differentiable RGB SSIM loss matching the external 2D per-channel metric."""

    def __init__(self, loss_weight=1.0, window_size=11, sigma=1.5, data_range=1.0):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.data_range = float(data_range)
        coords = torch.arange(self.window_size, dtype=torch.float32) - self.window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g = g / g.sum()
        kernel_2d = (g[:, None] @ g[None, :]).view(1, 1, self.window_size, self.window_size)
        self.register_buffer('kernel', kernel_2d)

    def forward(self, pred, target, **kwargs):
        c = pred.shape[1]
        weight = self.kernel.repeat(c, 1, 1, 1)
        pad = self.window_size // 2
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        mu_x = F.conv2d(pred, weight, padding=pad, groups=c)
        mu_y = F.conv2d(target, weight, padding=pad, groups=c)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x = F.conv2d(pred * pred, weight, padding=pad, groups=c) - mu_x2
        sigma_y = F.conv2d(target * target, weight, padding=pad, groups=c) - mu_y2
        sigma_xy = F.conv2d(pred * target, weight, padding=pad, groups=c) - mu_xy
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2) + 1e-12)
        return self.loss_weight * (1.0 - ssim_map.mean())


class DownsampleConsistencyLoss(nn.Module):
    """Keep the SR output consistent with its LR input after area downsampling."""

    def __init__(self, loss_weight=1.0, eps=1e-3):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)

    def forward(self, pred, lq=None, **kwargs):
        if lq is None or self.loss_weight == 0:
            return pred.new_tensor(0.0)
        pred_down = F.interpolate(pred, size=lq.shape[-2:], mode='area')
        return self.loss_weight * torch.mean(torch.sqrt((pred_down - lq.detach()) ** 2 + self.eps ** 2))


class AuxEnhanceLoss(nn.Module):
    """Very weak LR enhancement loss for joint SESR auxiliary output.

    It avoids edge/FFT/red terms. The enhancement branch should help feature
    learning but must not dominate color style or final SR reconstruction.
    """

    def __init__(self, loss_weight=1.0, color_weight=0.002, ratio_weight=0.002, eps=1e-3, reduction='mean'):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.pixel = CharbonnierLoss(loss_weight=1.0, eps=eps, reduction=reduction)
        self.color = ColorMeanLoss(loss_weight=color_weight)
        self.ratio = ChannelRatioLoss(loss_weight=ratio_weight)

    def forward(self, pred, target, weight=None, **kwargs):
        loss = self.pixel(pred, target, weight=weight)
        loss = loss + self.color(pred, target)
        loss = loss + self.ratio(pred, target)
        return self.loss_weight * loss


class VGGPerceptualLoss(nn.Module):
    """Small-weight VGG perceptual loss for final SR output.

    Notes:
      - Use this only on the final HR SR output, not on the LR enhancement branch.
      - Input tensors are expected to be RGB in [0, 1].
      - If pretrained weight loading fails in your environment, set pretrained: false
        in the YAML to make the code run, but pretrained weights are recommended.
    """

    layer_name_mapping = {
        'conv1_1': 0, 'conv1_2': 2,
        'conv2_1': 5, 'conv2_2': 7,
        'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_4': 16,
        'conv4_1': 19, 'conv4_2': 21, 'conv4_3': 23, 'conv4_4': 25,
        'conv5_1': 28, 'conv5_2': 30, 'conv5_3': 32, 'conv5_4': 34,
    }

    def __init__(self, layer_weights=None, perceptual_weight=0.003, style_weight=0.0,
                 criterion='l1', use_input_norm=True, range_norm=False,
                 pretrained=True, requires_grad=False):
        super().__init__()
        if layer_weights is None:
            layer_weights = {'conv3_4': 1.0}
        self.layer_weights = layer_weights
        self.perceptual_weight = float(perceptual_weight)
        self.style_weight = float(style_weight)
        self.use_input_norm = bool(use_input_norm)
        self.range_norm = bool(range_norm)

        max_idx = max(self.layer_name_mapping[k] for k in self.layer_weights.keys())
        self.vgg = self._build_vgg19(pretrained=pretrained)[:max_idx + 1].eval()

        for p in self.vgg.parameters():
            p.requires_grad = requires_grad

        if criterion == 'l1':
            self.criterion = F.l1_loss
        elif criterion == 'l2':
            self.criterion = F.mse_loss
        else:
            raise ValueError(f'Unsupported perceptual criterion: {criterion}')

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def _build_vgg19(self, pretrained=True):
        if pretrained:
            # Compatible with both new and old torchvision APIs.
            try:
                weights = models.VGG19_Weights.IMAGENET1K_V1
                return models.vgg19(weights=weights).features
            except Exception:
                try:
                    return models.vgg19(pretrained=True).features
                except Exception:
                    return models.vgg19(pretrained=False).features
        return models.vgg19(pretrained=False).features

    def _preprocess(self, x):
        if self.range_norm:
            # Convert [-1, 1] to [0, 1] when needed.
            x = (x + 1.0) * 0.5
        x = x.clamp(0, 1)
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return x

    def _extract_features(self, x):
        features = {}
        x = self._preprocess(x)
        requested = {self.layer_name_mapping[k]: k for k in self.layer_weights.keys()}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in requested:
                features[requested[i]] = x
        return features

    @staticmethod
    def _gram_mat(x):
        b, c, h, w = x.size()
        feat = x.view(b, c, h * w)
        gram = torch.bmm(feat, feat.transpose(1, 2)) / (c * h * w)
        return gram

    def forward(self, pred, target):
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target.detach())

        perceptual_loss = pred.new_tensor(0.0)
        style_loss = pred.new_tensor(0.0)

        if self.perceptual_weight > 0:
            for k, weight in self.layer_weights.items():
                perceptual_loss = perceptual_loss + float(weight) * self.criterion(pred_features[k], target_features[k])
            perceptual_loss = perceptual_loss * self.perceptual_weight
        else:
            perceptual_loss = None

        if self.style_weight > 0:
            for k, weight in self.layer_weights.items():
                style_loss = style_loss + float(weight) * self.criterion(
                    self._gram_mat(pred_features[k]), self._gram_mat(target_features[k])
                )
            style_loss = style_loss * self.style_weight
        else:
            style_loss = None

        return perceptual_loss, style_loss


class ColorMeanLoss(nn.Module):
    """Match per-image RGB channel means to reduce global color cast."""

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = float(loss_weight)

    def forward(self, pred, target, **kwargs):
        pred_mean = pred.mean(dim=(2, 3))
        target_mean = target.mean(dim=(2, 3))
        return self.loss_weight * F.l1_loss(pred_mean, target_mean)


class ChannelRatioLoss(nn.Module):
    """Match RGB proportions, useful for suppressing red/yellow over-compensation."""

    def __init__(self, loss_weight=1.0, eps=1e-6):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)

    def forward(self, pred, target, **kwargs):
        pred_mean = pred.mean(dim=(2, 3)).clamp_min(self.eps)
        target_mean = target.mean(dim=(2, 3)).clamp_min(self.eps)
        pred_ratio = pred_mean / pred_mean.sum(dim=1, keepdim=True).clamp_min(self.eps)
        target_ratio = target_mean / target_mean.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return self.loss_weight * F.l1_loss(pred_ratio, target_ratio)


class LowFrequencyColorLoss(nn.Module):
    """Match low-frequency color/illumination without forcing texture details."""

    def __init__(self, loss_weight=1.0, pool_size=16):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.pool_size = int(pool_size)

    def lowpass(self, x):
        if self.pool_size <= 1 or min(x.shape[-2:]) < self.pool_size:
            return x
        return F.avg_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * F.l1_loss(self.lowpass(pred), self.lowpass(target))


class RedSuppressLoss(nn.Module):
    """
    Penalize only global red-ratio over-compensation.

    This is conservative: it does not force images to be blue/green; it only
    reduces cases where the predicted global R proportion is higher than GT.
    """

    def __init__(self, loss_weight=1.0, eps=1e-6):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)

    def forward(self, pred, target, **kwargs):
        pred_mean = pred.mean(dim=(2, 3)).clamp_min(self.eps)
        target_mean = target.mean(dim=(2, 3)).clamp_min(self.eps)
        pred_ratio = pred_mean / pred_mean.sum(dim=1, keepdim=True).clamp_min(self.eps)
        target_ratio = target_mean / target_mean.sum(dim=1, keepdim=True).clamp_min(self.eps)
        extra_red = F.relu(pred_ratio[:, 0] - target_ratio[:, 0])
        return self.loss_weight * extra_red.mean()


class RedExcessLoss(nn.Module):
    """
    Pixel-wise red excess constraint.

    It compares how much the red channel exceeds the average of green/blue in
    pred and GT. This targets the remaining pink/red cast in sand and highlights
    without suppressing naturally red objects.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = float(loss_weight)

    def forward(self, pred, target, **kwargs):
        pred_r = pred[:, 0:1]
        pred_gb = 0.5 * (pred[:, 1:2] + pred[:, 2:3])
        target_r = target[:, 0:1]
        target_gb = 0.5 * (target[:, 1:2] + target[:, 2:3])
        pred_excess = F.relu(pred_r - pred_gb)
        target_excess = F.relu(target_r - target_gb)
        return self.loss_weight * F.l1_loss(pred_excess, target_excess)


class UWSRLoss(nn.Module):
    """
    Underwater SR loss inspired by SwinWave-SR:
    Charbonnier + Edge/Gradient + MS-SSIM, with only lightweight color constraints.

    The goal is to improve PSNR/SSIM and clarity first, while still suppressing
    global red over-compensation.
    """

    def __init__(self, loss_weight=1.0, edge_weight=0.15, grad_weight=0.10, sobel_weight=0.05,
                 fft_weight=0.01, dwt_weight=0.05, lap_pyr_weight=0.03,
                 edge_aware_weight=0.0, false_edge_weight=0.25, cmi_weight=0.01,
                 ssim_weight=0.0, ms_ssim_weight=0.06, luma_ms_ssim_weight=0.0,
                 external_rgb_ssim_weight=0.0,
                 lr_consistency_weight=0.0,
                 hf_decay_start_iter=0, hf_decay_end_iter=0, hf_min_scale=1.0,
                 color_weight=0.04, ratio_weight=0.04, lowfreq_color_weight=0.0,
                 red_weight=0.025,
                 red_excess_weight=0.008, eps=1e-3, reduction='mean'):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.hf_decay_start_iter = int(hf_decay_start_iter)
        self.hf_decay_end_iter = int(hf_decay_end_iter)
        self.hf_min_scale = float(hf_min_scale)
        self.pixel = CharbonnierLoss(loss_weight=1.0, eps=eps, reduction=reduction)
        self.edge = EdgeLoss(loss_weight=edge_weight)
        self.grad = GradientLoss(loss_weight=grad_weight)
        self.sobel = SobelSharpnessLoss(loss_weight=sobel_weight)
        self.fft = FFTLoss(loss_weight=fft_weight)
        self.dwt_high = DWTHighFrequencyLoss(loss_weight=dwt_weight)
        self.lap_pyr = LaplacianPyramidLoss(loss_weight=lap_pyr_weight)
        self.edge_aware_hf = EdgeAwareHighFrequencyLoss(
            loss_weight=edge_aware_weight,
            false_edge_weight=false_edge_weight,
        )
        self.cmi = SoftCMIContrastLoss(loss_weight=cmi_weight)
        # Prefer MS-SSIM for structure; keep ssim_weight for backward compatibility.
        if ms_ssim_weight is None:
            ms_ssim_weight = ssim_weight
        self.ssim = MSSSIMLoss(loss_weight=ms_ssim_weight)
        self.luma_ssim = LumaMSSSIMLoss(loss_weight=luma_ms_ssim_weight)
        self.external_rgb_ssim = ExternalRGBSSIMLoss(loss_weight=external_rgb_ssim_weight)
        self.lr_consistency = DownsampleConsistencyLoss(loss_weight=lr_consistency_weight, eps=eps)
        self.color = ColorMeanLoss(loss_weight=color_weight)
        self.ratio = ChannelRatioLoss(loss_weight=ratio_weight)
        self.lowfreq_color = LowFrequencyColorLoss(loss_weight=lowfreq_color_weight)
        self.red = RedSuppressLoss(loss_weight=red_weight)
        self.red_excess = RedExcessLoss(loss_weight=red_excess_weight)

    def _hf_scale(self, current_iter, pred):
        if current_iter is None or self.hf_decay_end_iter <= self.hf_decay_start_iter:
            return pred.new_tensor(1.0)
        cur = float(current_iter)
        if cur <= self.hf_decay_start_iter:
            return pred.new_tensor(1.0)
        if cur >= self.hf_decay_end_iter:
            return pred.new_tensor(self.hf_min_scale)
        progress = (cur - self.hf_decay_start_iter) / max(1.0, self.hf_decay_end_iter - self.hf_decay_start_iter)
        scale = 1.0 + progress * (self.hf_min_scale - 1.0)
        return pred.new_tensor(scale)

    def forward(self, pred, target, weight=None, **kwargs):
        current_iter = kwargs.get('current_iter', None)
        lq = kwargs.get('lq', None)
        hf_scale = self._hf_scale(current_iter, pred)
        loss = self.pixel(pred, target, weight=weight)
        loss = loss + hf_scale * self.edge(pred, target)
        loss = loss + hf_scale * self.grad(pred, target)
        loss = loss + hf_scale * self.sobel(pred, target)
        loss = loss + hf_scale * self.fft(pred, target)
        loss = loss + hf_scale * self.dwt_high(pred, target)
        loss = loss + hf_scale * self.lap_pyr(pred, target)
        loss = loss + hf_scale * self.edge_aware_hf(pred, target)
        loss = loss + hf_scale * self.cmi(pred, target)
        loss = loss + self.ssim(pred, target)
        loss = loss + self.luma_ssim(pred, target)
        loss = loss + self.external_rgb_ssim(pred, target)
        loss = loss + self.lr_consistency(pred, lq=lq)
        loss = loss + self.color(pred, target)
        loss = loss + self.ratio(pred, target)
        loss = loss + self.lowfreq_color(pred, target)
        loss = loss + self.red(pred, target)
        loss = loss + self.red_excess(pred, target)
        return self.loss_weight * loss
