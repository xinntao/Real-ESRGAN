"""Custom loss functions for Real-ESRGAN with enhanced perceptual metrics.

This module provides LPIPS and DISTS losses for improved perceptual quality.
"""

import torch
import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY

try:
    import lpips
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False

try:
    import pyiqa
    _HAS_PYIQA = True
except ImportError:
    _HAS_PYIQA = False


@LOSS_REGISTRY.register()
class LPIPSLoss(nn.Module):
    """LPIPS (Learned Perceptual Image Patch Similarity) loss.

    LPIPS measures perceptual similarity using deep features from pre-trained networks.
    Lower values indicate more similar images from a perceptual standpoint.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        net (str): Network to use for computing LPIPS. Options: 'alex', 'vgg', 'squeeze'.
            Default: 'alex' (AlexNet - fastest and recommended).
        use_input_norm (bool): If True, normalize from [0, 1] to ImageNet stats. Default: False.
    """

    def __init__(self, loss_weight=1.0, net='alex', use_input_norm=False):
        super(LPIPSLoss, self).__init__()
        if not _HAS_LPIPS:
            raise ImportError('lpips package is required for LPIPSLoss. Install with: pip install lpips')

        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm

        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(net=net, verbose=False)

        # Freeze LPIPS model parameters
        for param in self.lpips_model.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        """Forward function.

        Args:
            pred (Tensor): Predicted images with shape (n, 3, h, w). Range [0, 1].
            target (Tensor): Target images with shape (n, 3, h, w). Range [0, 1].

        Returns:
            Tensor: LPIPS loss value.
        """
        # LPIPS expects input in range [-1, 1]
        if not self.use_input_norm:
            pred = pred * 2.0 - 1.0
            target = target * 2.0 - 1.0

        # Compute LPIPS distance
        lpips_val = self.lpips_model(pred, target)

        return lpips_val.mean() * self.loss_weight


@LOSS_REGISTRY.register()
class DISTSLoss(nn.Module):
    """DISTS (Deep Image Structure and Texture Similarity) loss.

    DISTS is a full-reference image quality assessment metric that correlates well
    with human perception. It captures both structure and texture information.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        as_loss (bool): If True, return 1-DISTS (for use as a loss). Default: True.
    """

    def __init__(self, loss_weight=1.0, as_loss=True):
        super(DISTSLoss, self).__init__()
        if not _HAS_PYIQA:
            raise ImportError('pyiqa package is required for DISTSLoss. Install with: pip install pyiqa')

        self.loss_weight = loss_weight
        self.as_loss = as_loss

        # Initialize DISTS metric
        self.dists_metric = pyiqa.create_metric('dists', device=torch.device('cuda'))

        # Freeze DISTS model parameters
        for param in self.dists_metric.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        """Forward function.

        Args:
            pred (Tensor): Predicted images with shape (n, 3, h, w). Range [0, 1].
            target (Tensor): Target images with shape (n, 3, h, w). Range [0, 1].

        Returns:
            Tensor: DISTS loss value.
        """
        # Clamp inputs to [0, 1] range as required by DISTS
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        # Compute DISTS score (lower is better, range: 0-1)
        dists_val = self.dists_metric(pred, target)

        # If as_loss is True, we can use it directly (lower is better)
        # Otherwise return 1-DISTS to make higher scores better
        if self.as_loss:
            return dists_val.mean() * self.loss_weight
        else:
            return (1.0 - dists_val).mean() * self.loss_weight


@LOSS_REGISTRY.register()
class CombinedPerceptualLoss(nn.Module):
    """Combined perceptual loss using multiple metrics.

    This combines LPIPS, DISTS, and traditional perceptual losses for comprehensive
    perceptual quality optimization.

    Args:
        lpips_weight (float): Weight for LPIPS loss. Default: 1.0.
        dists_weight (float): Weight for DISTS loss. Default: 1.0.
        lpips_net (str): Network for LPIPS. Default: 'alex'.
    """

    def __init__(self, lpips_weight=1.0, dists_weight=1.0, lpips_net='alex'):
        super(CombinedPerceptualLoss, self).__init__()

        self.lpips_loss = None
        self.dists_loss = None

        if _HAS_LPIPS and lpips_weight > 0:
            self.lpips_loss = LPIPSLoss(loss_weight=lpips_weight, net=lpips_net)

        if _HAS_PYIQA and dists_weight > 0:
            self.dists_loss = DISTSLoss(loss_weight=dists_weight)

    def forward(self, pred, target):
        """Forward function.

        Args:
            pred (Tensor): Predicted images with shape (n, 3, h, w). Range [0, 1].
            target (Tensor): Target images with shape (n, 3, h, w). Range [0, 1].

        Returns:
            Tensor: Combined perceptual loss value.
            tuple: Individual loss components (lpips, dists).
        """
        total_loss = 0
        lpips_val = None
        dists_val = None

        if self.lpips_loss is not None:
            lpips_val = self.lpips_loss(pred, target)
            total_loss += lpips_val

        if self.dists_loss is not None:
            dists_val = self.dists_loss(pred, target)
            total_loss += dists_val

        return total_loss, lpips_val, dists_val
