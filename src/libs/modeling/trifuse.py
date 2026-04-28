"""
Trimodal Cross-Modal Attention Preprocessor.

Architecture (3 stages):
  1. All three modalities are projected to a shared d_model dimension.

  2. Position-wise cross-modal attention .

  3. The three streams are concatenated.

Interface contract (same as every other preprocessor):
    forward(text, audio, video) -> (B, T, 3 * d_model)
    .d_out : int   — output feature dimension (= 3 * d_model)

Config key:
    preprocessor:
      type: "trifuse"
      d_model: 256 
      n_heads: 4
      n_fusion_layers: 1
      dropout: 0.1
      modality_dropout: 0.1
"""
import torch
from torch import nn


class PositionwiseFFN(nn.Module):
    """
    Pre-norm position-wise feed-forward block.

    Applied identically and independently at every timestep.

    Args:
        d_model : Feature dimension.
        dropout : Dropout probability after each linear layer.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.ln  = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.ln(x))


class CrossModalAttentionLayer(nn.Module):
    """
    For each modality, multi-head attention is applied with that modality as
    query and the other two modalities as keys/values, followed by a
    position-wise FFN.  All sub-layers use pre-norm residual connections.

    Args:
        d_model : Common feature dimension.
        n_heads : Number of attention heads (must divide d_model).
        dropout : Dropout probability inside attention and FFN.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        mha = dict(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn_v = nn.MultiheadAttention(**mha)
        self.attn_a = nn.MultiheadAttention(**mha)
        self.attn_x = nn.MultiheadAttention(**mha)
        self.ln_v   = nn.LayerNorm(d_model)
        self.ln_a   = nn.LayerNorm(d_model)
        self.ln_x   = nn.LayerNorm(d_model)
        self.ffn_v  = PositionwiseFFN(d_model, dropout)
        self.ffn_a  = PositionwiseFFN(d_model, dropout)
        self.ffn_x  = PositionwiseFFN(d_model, dropout)

    def forward(
        self,
        v: torch.Tensor,   # (BT, 1, D)
        a: torch.Tensor,   # (BT, 1, D)
        x: torch.Tensor,   # (BT, 1, D)
    ):
        """Returns: (v, a, x) with the same shapes as the inputs."""
        v_n = self.ln_v(v)
        a_n = self.ln_a(a)
        x_n = self.ln_x(x)

        # Use normalised features as KV
        kv_ax = torch.cat([a_n, x_n], dim=1)   # (BT, 2, D) — KV for video
        kv_vx = torch.cat([v_n, x_n], dim=1)   # (BT, 2, D) — KV for audio
        kv_va = torch.cat([v_n, a_n], dim=1)   # (BT, 2, D) — KV for text

        v_attn, _ = self.attn_v(v_n, kv_ax, kv_ax)
        a_attn, _ = self.attn_a(a_n, kv_vx, kv_vx)
        x_attn, _ = self.attn_x(x_n, kv_va, kv_va)

        v = self.ffn_v(v + v_attn)
        a = self.ffn_a(a + a_attn)
        x = self.ffn_x(x + x_attn)

        return v, a, x


class TriFusePreprocessor(nn.Module):
    """
    Trimodal cross-modal attention preprocessor.

    All three modality inputs arrive pre-aligned at (B, T, D_m).

    Args:
        text_dim          : Native text feature dimension.
        audio_dim         : Native audio feature dimension.
        video_dim         : Native video feature dimension.
        d_model           : Shared internal dimension.
        n_heads           : Number of attention heads (must divide d_model).
        n_fusion_layers   : Number of cross-modal attention layers.
        dropout           : Dropout probability throughout.
        modality_dropout  : Probability of zeroing an entire modality per sample
                            during training.

    Output shape: (B, T, 3 * d_model)
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        video_dim: int,
        d_model: int,
        n_heads: int = 8,
        n_fusion_layers: int = 4,
        dropout: float = 0.1,
        modality_dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model           = d_model
        self.d_out             = 3 * d_model
        self.modality_dropout  = modality_dropout

        # Stage 1: projections
        self.proj_v = nn.Linear(video_dim, d_model)
        self.proj_a = nn.Linear(audio_dim, d_model)
        self.proj_x = nn.Linear(text_dim,  d_model)

        # Stage 2: cross-modal attention
        self.fusion_layers = nn.ModuleList([
            CrossModalAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_fusion_layers)
        ])


    def forward(
        self,
        text:  torch.Tensor,   # (B, T, text_dim)
        audio: torch.Tensor,   # (B, T, audio_dim)
        video: torch.Tensor,   # (B, T, video_dim)
    ) -> torch.Tensor:         # (B, T, 3 * d_model)
        B, T, _ = video.shape

        # Modality dropout
        if self.training and self.modality_dropout > 0.0:
            keep = torch.bernoulli(
                torch.full((B, 3), 1.0 - self.modality_dropout, device=video.device)
            )
            keep[keep.sum(dim=1) == 0] = 1.0   # never drop all modalities
            video = video * keep[:, 0].view(B, 1, 1)
            audio = audio * keep[:, 1].view(B, 1, 1)
            text  = text  * keep[:, 2].view(B, 1, 1)

        v = self.proj_v(video)
        a = self.proj_a(audio)
        x = self.proj_x(text)

        # Flatten time into batch so each layer operates position-wise.
        D = self.d_model
        v_pw = v.reshape(B * T, 1, D)
        a_pw = a.reshape(B * T, 1, D)
        x_pw = x.reshape(B * T, 1, D)

        for layer in self.fusion_layers:
            v_pw, a_pw, x_pw = layer(v_pw, a_pw, x_pw)

        v = v_pw.reshape(B, T, D)
        a = a_pw.reshape(B, T, D)
        x = x_pw.reshape(B, T, D)

        return torch.cat([v, a, x], dim=-1)   # (B, T, 3 * d_model)
