"""
Multiscale Transformer Encoder (feature pyramid backbone).

Directly ported from ActionFormer's ConvTransformerBackbone with these changes:
  - No registry decorator (standalone module)
  - Removed multi-input projection path (we always receive a single fused tensor)
  - use_abs_pe defaults to False (no positional encoding per ActionFormer recommendation)
  - Docstring updated for this project's context

Architecture:
  arch = (n_proj_convs, n_stem_blocks, n_branch_blocks)
        = (2,           1,             5)              by default

  embd[0..1]  : 2x MaskedConv1D projection,  fused_dim → d_model
  stem[0]     : 1x TransformerBlock (no downsampling, local attention)
  branch[0..4]: 5x TransformerBlock (each with 2x downsampling)

  → Feature pyramid with 6 levels at resolutions T, T/2, T/4, T/8, T/16, T/32.

Sequence-length constraint for local attention (window_size w, window_overlap = w//2):
  Each level's sequence length must be divisible by 2 * window_overlap.
  For w=19: every T//2^i must be divisible by 18.
  For w=-1 (global): no constraint.

Differences from vanilla ActionFormer backbone.py:
  1. n_in is always a scalar (not a list/tuple); multi-input projection removed.
  2. No registry.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import (
    MaskedConv1D, LayerNorm, TransformerBlock, get_sinusoid_encoding
)


class ConvTransformerBackbone(nn.Module):
    """
    Multiscale Transformer Encoder producing a feature pyramid.

    Args:
        n_in        : Input feature dimension (= CrossModalFusion.fused_dim).
        n_embd      : Internal / output feature dimension (d_model).
        n_head      : Number of attention heads in each TransformerBlock.
        n_embd_ks   : Kernel size for projection conv layers (default 3).
        max_len     : Maximum sequence length (used if use_abs_pe=True).
        arch        : (n_proj_convs, n_stem_blocks, n_branch_blocks).
        mha_win_size: List of window sizes, length = 1 + n_branch_blocks.
                      -1 or 1 → global attention; >1 → local window attention.
        scale_factor: Downsampling factor between pyramid levels (default 2).
        with_ln     : If True, LayerNorm is applied after projection conv layers.
        attn_pdrop  : Dropout on attention maps.
        proj_pdrop  : Dropout on projections / MLP.
        path_pdrop  : Drop-path rate.
        use_abs_pe  : If True, add sinusoidal absolute position embeddings.
        use_rel_pe  : If True, add learnable relative position encodings inside
                      local attention blocks.
    """

    def __init__(
        self,
        n_in,
        n_embd,
        n_head,
        n_embd_ks=3,
        max_len=2304,
        arch=(2, 1, 5),
        mha_win_size=None,
        scale_factor=2,
        with_ln=True,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        use_abs_pe=False,
        use_rel_pe=False,
    ):
        super().__init__()
        assert len(arch) == 3
        n_branch = arch[2]
        if mha_win_size is None:
            mha_win_size = [-1] * (1 + n_branch)
        assert len(mha_win_size) == (1 + n_branch), \
            "mha_win_size must have length 1 + n_branch_blocks"

        self.arch         = arch
        self.mha_win_size = mha_win_size
        self.max_len      = max_len
        self.scale_factor = scale_factor
        self.use_abs_pe   = use_abs_pe
        self.use_rel_pe   = use_rel_pe
        self.relu         = nn.ReLU(inplace=True)

        # ── Projection conv layers (embedding network) ──────────────────────
        self.embd      = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            in_ch = n_in if idx == 0 else n_embd
            self.embd.append(
                MaskedConv1D(
                    in_ch, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks // 2,
                    bias=(not with_ln),
                )
            )
            self.embd_norm.append(LayerNorm(n_embd) if with_ln else nn.Identity())

        # ── Optional absolute position embedding ────────────────────────────
        if use_abs_pe:
            pos_embd = get_sinusoid_encoding(max_len, n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # ── Stem transformer (no downsampling) ──────────────────────────────
        self.stem = nn.ModuleList()
        for _ in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=mha_win_size[0],
                    use_rel_pe=use_rel_pe,
                )
            )

        # ── Branch transformers (each with 2x downsampling) ─────────────────
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(scale_factor, scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=mha_win_size[1 + idx],
                    use_rel_pe=use_rel_pe,
                )
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask):
        """
        Args:
            x   : (B, C, T)  — fused features, channel-first.
            mask: (B, 1, T)  — bool, True for valid positions.

        Returns:
            out_feats: tuple of L feature maps (B, n_embd, T_i)
            out_masks: tuple of L masks (B, 1, T_i)
            where L = 1 + n_branch_blocks (= 6 by default).
        """
        B, C, T = x.size()

        # ── Projection convs ────────────────────────────────────────────────
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # ── Absolute position embeddings (training) ──────────────────────────
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Sequence exceeds max_len"
            x = x + self.pos_embd[:, :, :T] * mask.to(x.dtype)

        # ── Absolute position embeddings (inference, interpolated) ──────────
        if self.use_abs_pe and (not self.training):
            pe = (F.interpolate(self.pos_embd, T, mode='linear', align_corners=False)
                  if T > self.max_len else self.pos_embd)
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # ── Stem ─────────────────────────────────────────────────────────────
        for block in self.stem:
            x, mask = block(x, mask)

        out_feats = (x,)
        out_masks = (mask,)

        # ── Branch (downsampling) ─────────────────────────────────────────────
        for block in self.branch:
            x, mask = block(x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks
