"""
Classification and regression decoder heads.

Ported from ActionFormer's PtTransformerClsHead and PtTransformerRegHead
(originally in meta_archs.py) with these adaptations:
  - Binary classification (num_classes=1, single sigmoid) instead of C-class.
  - Configurable num_classes for future multi-class (racial/religious/sexist/…).
  - Regression output (d_start, d_end) with ReLU to ensure non-negative distances.
  - Per-FPN-level Scale multipliers on regression output (from ActionFormer).
  - No registry decorator.

Differences from vanilla ActionFormer heads:
  1. cls head supports num_classes=1 (binary) via configurable parameter.
  2. Kept prior_prob initialization for numerical stability.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import MaskedConv1D, LayerNorm, Scale


class ClsHead(nn.Module):
    """
    Shared 1D Conv classification head applied at every FPN level.

    Architecture (per level):
        [MaskedConv1D → LayerNorm → ReLU] × (n_layers - 1)  →  MaskedConv1D classifier

    Args:
        input_dim  : Input channel dimension (= FPN output dim).
        feat_dim   : Internal channel dimension.
        num_classes: Number of output classes (1 for binary hate/no-hate).
        prior_prob : Prior probability used to initialise classifier bias (for stability).
        n_layers   : Total number of conv layers including the classifier (default 3).
        kernel_size: Conv kernel size.
        with_ln    : If True, LayerNorm after intermediate conv layers.
    """

    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        n_layers=3,
        kernel_size=3,
        with_ln=True,
    ):
        super().__init__()
        self.act = nn.ReLU()

        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(n_layers - 1):
            in_dim  = input_dim if idx == 0 else feat_dim
            out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1, padding=kernel_size // 2,
                    bias=(not with_ln),
                )
            )
            self.norm.append(LayerNorm(out_dim) if with_ln else nn.Identity())

        # Final classifier conv
        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size,
            stride=1, padding=kernel_size // 2,
        )

        # Prior probability initialisation for better early-training stability
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            nn.init.constant_(self.cls_head.conv.bias, bias_value)

    def forward(self, fpn_feats, fpn_masks):
        """
        Args:
            fpn_feats: list of (B, C, T_i)
            fpn_masks: list of (B, 1, T_i)  bool

        Returns:
            out_logits: tuple of (B, num_classes, T_i)  (raw logits, not sigmoided)
        """
        assert len(fpn_feats) == len(fpn_masks)
        out_logits = tuple()
        for cur_feat, cur_mask in zip(fpn_feats, fpn_masks):
            x = cur_feat
            for conv, norm in zip(self.head, self.norm):
                x, _ = conv(x, cur_mask)
                x = self.act(norm(x))
            logits, _ = self.cls_head(x, cur_mask)
            out_logits += (logits,)
        return out_logits


class RegHead(nn.Module):
    """
    Shared 1D Conv regression head applied at every FPN level.
    Predicts (d_start, d_end) — distances to segment boundaries, non-negative.

    Architecture (per level):
        [MaskedConv1D → LayerNorm → ReLU] × (n_layers - 1)  →  MaskedConv1D offset head
        → ReLU (ensures non-negative distances) → Scale (per-level learnable multiplier)

    Args:
        input_dim  : Input channel dimension.
        feat_dim   : Internal channel dimension.
        fpn_levels : Number of FPN levels (one Scale module per level).
        n_layers   : Total number of conv layers (default 3).
        kernel_size: Conv kernel size.
        with_ln    : If True, LayerNorm after intermediate conv layers.
    """

    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        n_layers=3,
        kernel_size=3,
        with_ln=True,
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = nn.ReLU()

        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(n_layers - 1):
            in_dim  = input_dim if idx == 0 else feat_dim
            out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1, padding=kernel_size // 2,
                    bias=(not with_ln),
                )
            )
            self.norm.append(LayerNorm(out_dim) if with_ln else nn.Identity())

        # Per-level learnable scale (from ActionFormer)
        self.scale = nn.ModuleList([Scale() for _ in range(fpn_levels)])

        # Offset regression: output 2 channels (d_start, d_end)
        self.offset_head = MaskedConv1D(
            feat_dim, 2, kernel_size,
            stride=1, padding=kernel_size // 2,
        )

    def forward(self, fpn_feats, fpn_masks):
        """
        Args:
            fpn_feats: list of (B, C, T_i)
            fpn_masks: list of (B, 1, T_i)  bool

        Returns:
            out_offsets: tuple of (B, 2, T_i)  — non-negative offset predictions
        """
        assert len(fpn_feats) == len(fpn_masks) == self.fpn_levels
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            x = cur_feat
            for conv, norm in zip(self.head, self.norm):
                x, _ = conv(x, cur_mask)
                x = self.act(norm(x))
            offsets, _ = self.offset_head(x, cur_mask)
            # ReLU + per-level scale ensures non-negative distances
            out_offsets += (F.relu(self.scale[l](offsets)),)
        return out_offsets
