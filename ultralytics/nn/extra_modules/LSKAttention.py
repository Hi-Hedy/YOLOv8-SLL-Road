"""Large Separable Kernel Attention (LSKA v3) with channel attention and residual connection.

Architecture:
  input -> depthwise conv (1xk + kx1) -> activation
        -> spatial dilated conv (1xK + Kx1) -> activation
        -> SE channel attention -> residual -> output

Reference: LSKA: Large Separable Kernel Attention for Object Detection
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Union, Tuple

__all__ = ['LSKA']


class LSKA(nn.Module):
    """Large Separable Kernel Attention with SE-style channel attention.

    Decomposes a large KxK convolution into sequential 1xK and Kx1
    depth-wise dilated convolutions for efficient long-range context modelling.

    Args:
        dim:        Input channels.
        k_size:     Large kernel size (7, 11, 23, 35, 41, 53). Must be odd.
        reduction:  SE channel-attention reduction ratio (default 4).
        use_res:    Whether to use residual connection (default True).
        act_layer:  Activation function (default nn.ReLU).

    Input/Output shape: (B, C, H, W)
    """

    def __init__(self,
                 dim: int,
                 k_size: Union[int, Tuple[int, int]] = 23,
                 reduction: int = 3,
                 use_res: bool = True,
                 use_proj: bool = False,
                 act_layer: nn.Module = nn.ReLU):
        super().__init__()
        self.dim = dim

        # ---------- Validate k_size ----------
        if isinstance(k_size, int):
            assert k_size % 2 == 1, "k_size must be odd"
            k_size = (k_size, k_size)
        self.k_h, self.k_w = k_size
        self.use_res = use_res
        self.reduction = reduction

        # ---------- Stage 1: local depth-wise conv (1xk + kx1) ----------
        self.dw_conv_h = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1),
                                   groups=dim, bias=False)
        self.dw_conv_v = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0),
                                   groups=dim, bias=False)
        self.act1 = nn.ReLU(inplace=True)

        # ---------- Stage 2: dilated large-kernel spatial conv ----------
        self.dilation_h = self._compute_dilation(self.k_h)
        self.dilation_w = self._compute_dilation(self.k_w)

        self.spatial_conv_h = nn.Conv2d(
            dim, dim, (1, self.k_w),
            padding=(0, (self.k_w // 2) * self.dilation_w),
            dilation=(1, self.dilation_w), groups=dim, bias=False)
        self.spatial_conv_v = nn.Conv2d(
            dim, dim, (self.k_h, 1),
            padding=((self.k_h // 2) * self.dilation_h, 0),
            dilation=(self.dilation_h, 1), groups=dim, bias=False)
        self.act2 = nn.ReLU(inplace=True)

        # ---------- Stage 3: SE channel attention ----------
        reduced = max(1, dim // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, reduced, 1, bias=False),
            act_layer(inplace=True),
            nn.Conv2d(reduced, dim, 1, bias=False),
            nn.Sigmoid(),
        )

        # ---------- Optional projection ----------
        self.use_proj = use_proj
        if use_proj:
            self.proj = nn.Conv2d(dim, dim, 1)

        # ---------- Weight init ----------
        self._init_weights()

    def _compute_dilation(self, kernel_size: int) -> int:
        """Auto-compute dilation rate from kernel size."""
        base_dilation = {7: 2, 11: 2, 23: 3}
        return base_dilation.get(kernel_size, max(1, kernel_size // 7))

    def _init_weights(self):
        """Kaiming + Xavier weight initialisation."""
        for m in [self.dw_conv_h, self.dw_conv_v,
                  self.spatial_conv_h, self.spatial_conv_v]:
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        init.normal_(self.spatial_conv_h.weight, mean=0, std=0.01)
        init.normal_(self.spatial_conv_v.weight, mean=0, std=0.01)
        for m in self.se:
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        if self.use_proj:
            init.normal_(self.proj.weight, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Stage 1: local context
        x = self.dw_conv_h(x)
        x = self.dw_conv_v(x)
        x = self.act1(x)

        # Stage 2: long-range spatial context
        x = self.spatial_conv_h(x)
        x = self.spatial_conv_v(x)
        x = self.act2(x)

        # Stage 3: channel attention
        x = x * self.se(x)

        # Optional projection
        if self.use_proj:
            x = self.proj(x)

        # Residual
        return x + identity if self.use_res else x

    def __repr__(self):
        return f"LSKA(dim={self.dim}, k_size=({self.k_h},{self.k_w}), reduction={self.reduction})"
