"""
Backbone implementations for temporal action localisation.

1. ConvTransformerBackbone (from ActionFormer)
2. MaxPoolBackbone (from TemporalMaxer)
3. SGPBackbone (from TriDet)

All three share the same interface:
  forward(x, mask) -> (out_feats, out_masks)
  where out_feats is a tuple of L feature maps (B, n_embd, T_i),
  and L = 1 + n_branch_blocks.

Factory function:
  build_backbone(backbone_type, **kwargs) → nn.Module
"""
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import (
    MaskedConv1D, LayerNorm, TransformerBlock, get_sinusoid_encoding,
    TemporalMaxerBlock, SGPBlock,
)


def _init_weights(module):
    """Initialize bias to 0 for all Linear and Conv1d layers."""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def _build_projection_layers(n_proj, n_in, n_embd, n_embd_ks, with_ln):
    """Build projection conv layers (embedding network) shared by all backbones."""
    embd = nn.ModuleList()
    embd_norm = nn.ModuleList()
    for idx in range(n_proj):
        in_ch = n_in if idx == 0 else n_embd
        embd.append(
            MaskedConv1D(
                in_ch, n_embd, n_embd_ks,
                stride=1, padding=n_embd_ks // 2,
                bias=(not with_ln),
            )
        )
        embd_norm.append(LayerNorm(n_embd) if with_ln else nn.Identity())
    return embd, embd_norm


# ActionFormer backbone
class ConvTransformerBackbone(nn.Module):
    """
    Multiscale Transformer Encoder producing a feature pyramid.

    Args:
        n_in        : Input feature dimension.
        n_embd      : Internal / output feature dimension (d_model).
        n_head      : Number of attention heads in each TransformerBlock.
        n_embd_ks   : Kernel size for projection conv layers (default 3).
        max_len     : Maximum sequence length.
        arch        : (n_proj_convs, n_stem_blocks, n_branch_blocks).
        mha_win_size: List of window sizes, length = 1 + n_branch_blocks.
                      -1 or 1 -> global attention; >1 -> local window attention.
        scale_factor: Downsampling factor between pyramid levels (default 2).
        with_ln     : If True, LayerNorm is applied after projection conv layers.
        attn_pdrop  : Dropout on attention maps.
        proj_pdrop  : Dropout on projections / MLP.
        path_pdrop  : Droppath rate.
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
        **kwargs,
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

        # Projection conv layers
        self.embd, self.embd_norm = _build_projection_layers(
            arch[0], n_in, n_embd, n_embd_ks, with_ln
        )

        # Optional absolute position embedding
        if use_abs_pe:
            pos_embd = get_sinusoid_encoding(max_len, n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # Stem transformer (no downsampling)
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

        # Branch transformers (each with 2x downsampling)
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

        self.apply(_init_weights)

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

        # Projection convs
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # Absolute position embeddings (training)
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Sequence exceeds max_len"
            x = x + self.pos_embd[:, :, :T] * mask.to(x.dtype)

        # Absolute position embeddings (inference, interpolated)
        if self.use_abs_pe and (not self.training):
            pe = (F.interpolate(self.pos_embd, T, mode='linear', align_corners=False)
                  if T > self.max_len else self.pos_embd)
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # Stem
        for block in self.stem:
            x, mask = block(x, mask)

        out_feats = (x,)
        out_masks = (mask,)

        # Downsampling
        for block in self.branch:
            x, mask = block(x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks


# TemporalMaxer backbone
class MaxPoolBackbone(nn.Module):
    """
    MaxPool-based backbone from TemporalMaxer.

    Replaces transformer self-attention with parameter-free MaxPool1D blocks.
    The backbone has no learnable parameters beyond the initial projection convs.

    Architecture:
      embd[0..n_proj-1]     : n_proj MaskedConv1D projections  (n_in -> n_embd)
      branch[0..n_branch-1] : n_branch TemporalMaxerBlocks (each scale_factor downsampling)

      -> Feature pyramid with (1 + n_branch) levels.

    Args:
        n_in            : Input feature dimension.
        n_embd          : Internal / output feature dimension.
        n_embd_ks       : Kernel size for projection conv layers.
        max_len         : Maximum sequence length.
        arch            : (n_proj_convs, n_stem_ignored, n_branch_blocks).
                          n_stem is accepted for API compatibility but ignored;
                          all pyramid levels are produced by the MaxPool branch.
        scale_factor    : Downsampling factor per branch block.
        with_ln         : If True, LayerNorm after projection conv layers.
        pool_kernel_size: MaxPool kernel size.
    """

    def __init__(
        self,
        n_in,
        n_embd,
        n_embd_ks=3,
        max_len=2304,
        arch=(1, 1, 2),
        scale_factor=2,
        with_ln=True,
        pool_kernel_size=3,
        **kwargs,
    ):
        super().__init__()
        assert len(arch) == 3
        n_proj   = arch[0]
        n_branch = arch[2]

        self.arch         = arch
        self.max_len      = max_len
        self.scale_factor = scale_factor
        self.relu         = nn.ReLU(inplace=True)

        self.embd, self.embd_norm = _build_projection_layers(
            n_proj, n_in, n_embd, n_embd_ks, with_ln
        )

        self.branch = nn.ModuleList()
        for _ in range(n_branch):
            self.branch.append(
                TemporalMaxerBlock(
                    kernel_size=pool_kernel_size,
                    stride=scale_factor,
                    padding=pool_kernel_size // 2,
                    n_embd=n_embd,
                )
            )

        self.apply(_init_weights)

    def forward(self, x, mask):
        """
        Args:
            x   : (B, C, T)
            mask: (B, 1, T) bool

        Returns:
            out_feats: tuple of (1 + n_branch) feature maps (B, n_embd, T_i)
            out_masks: tuple of (1 + n_branch) masks (B, 1, T_i)
        """
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        out_feats = (x,)
        out_masks = (mask,)

        for block in self.branch:
            x, mask = block(x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks


# SGP backbone from TriDet 
class SGPBackbone(nn.Module):
    """
    SGP-based backbone from TriDet.

    Uses Scalable-Granularity Perception (SGP) layers with dual-branch depthwise
    convolutions.

    Architecture:
      embd[0..n_proj-1]     : n_proj MaskedConv1D projections  (n_in -> n_embd)
      stem[0..n_stem-1]     : n_stem SGPBlocks (stride=1, no downsampling)
      branch[0..n_branch-1] : n_branch SGPBlocks (stride=scale_factor)

      -> Feature pyramid with (1 + n_branch) levels.

    Args:
        n_in            : Input feature dimension.
        n_embd          : Internal / output feature dimension.
        n_embd_ks       : Kernel size for projection conv layers.
        max_len         : Maximum sequence length.
        arch            : (n_proj_convs, n_stem_blocks, n_branch_blocks).
        scale_factor    : Downsampling factor per branch block.
        with_ln         : If True, LayerNorm after projection conv layers.
        path_pdrop      : Drop-path rate for branch SGP blocks.
        sgp_kernel_size : SGP instant-level conv kernel size.
        sgp_mlp_dim     : Hidden dim for FFN MLP in SGP blocks.
        k               : Window-level kernel scale factor in SGP.
        init_conv_vars  : Gaussian init std for SGP depthwise conv weights.
        use_abs_pe      : If True, add sinusoidal absolute position embeddings.
        downsample_type : Downsampling method in branch SGP blocks.
    """

    def __init__(
        self,
        n_in,
        n_embd,
        n_embd_ks=3,
        max_len=2304,
        arch=(1, 1, 2),
        scale_factor=2,
        with_ln=True,
        path_pdrop=0.0,
        sgp_kernel_size=3,
        sgp_mlp_dim=None,
        k=1.5,
        init_conv_vars=1,
        use_abs_pe=False,
        downsample_type='max',
        **kwargs,
    ):
        super().__init__()
        assert len(arch) == 3
        n_proj   = arch[0]
        n_stem   = arch[1]
        n_branch = arch[2]

        self.arch         = arch
        self.max_len      = max_len
        self.scale_factor = scale_factor
        self.use_abs_pe   = use_abs_pe
        self.relu         = nn.ReLU(inplace=True)

        if sgp_mlp_dim is None:
            sgp_mlp_dim = 4 * n_embd

        if use_abs_pe:
            pos_embd = get_sinusoid_encoding(max_len, n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        self.embd, self.embd_norm = _build_projection_layers(
            n_proj, n_in, n_embd, n_embd_ks, with_ln
        )

        self.stem = nn.ModuleList()
        for _ in range(n_stem):
            self.stem.append(
                SGPBlock(
                    n_embd,
                    kernel_size=sgp_kernel_size,
                    n_ds_stride=1,
                    n_hidden=sgp_mlp_dim,
                    k=k,
                    init_conv_vars=init_conv_vars,
                )
            )

        self.branch = nn.ModuleList()
        for _ in range(n_branch):
            self.branch.append(
                SGPBlock(
                    n_embd,
                    kernel_size=sgp_kernel_size,
                    n_ds_stride=scale_factor,
                    path_pdrop=path_pdrop,
                    n_hidden=sgp_mlp_dim,
                    downsample_type=downsample_type,
                    k=k,
                    init_conv_vars=init_conv_vars,
                )
            )

        self.apply(_init_weights)

    def forward(self, x, mask):
        """
        Args:
            x   : (B, C, T)
            mask: (B, 1, T) bool

        Returns:
            out_feats: tuple of (1 + n_branch) feature maps (B, n_embd, T_i)
            out_masks: tuple of (1 + n_branch) masks (B, 1, T_i)
        """
        B, C, T = x.size()

        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Sequence exceeds max_len"
            x = x + self.pos_embd[:, :, :T] * mask.to(x.dtype)

        if self.use_abs_pe and (not self.training):
            pe = (F.interpolate(self.pos_embd, T, mode='linear', align_corners=False)
                  if T > self.max_len else self.pos_embd)
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        for block in self.stem:
            x, mask = block(x, mask)

        out_feats = (x,)
        out_masks = (mask,)

        for block in self.branch:
            x, mask = block(x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks


# Backbone factory
def build_backbone(backbone_type, **kwargs):
    """
    Args:
        backbone_type: One of "transformer", "temporalmaxer", "sgp".
        **kwargs     : Passed directly to the backbone constructor.

    Returns:
        nn.Module backbone instance.
    """
    _registry = {
        'transformer'  : ConvTransformerBackbone,
        'temporalmaxer': MaxPoolBackbone,
        'sgp'          : SGPBackbone,
    }
    if backbone_type not in _registry:
        raise ValueError(
            f"Unknown backbone type '{backbone_type}'. "
            f"Choose from: {list(_registry.keys())}"
        )
    return _registry[backbone_type](**kwargs)
