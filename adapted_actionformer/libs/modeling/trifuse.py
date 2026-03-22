"""
TriFuse — Trimodal Cross-Modal Attention Preprocessor.

Architecture (4 stages):

  Stage 1 — Linear projection.
             Text is sparse: zero vectors where no speech is present.
             A presence mask is derived from the raw input and used to re-zero
             text features after every stage that could corrupt zero inputs.
             No positional encoding is added — temporal context aggregation is
             left entirely to the backbone.

  Stage 2 — Per-modality position-wise FFN (n_unimodal_layers stacked pre-norm
             blocks).  Applied independently at each timestep so no temporal
             mixing occurs; the backbone owns all temporal modelling.
             Text outputs are re-zeroed at absent positions.

  Stage 3 — Position-wise cross-modal attention (n_fusion_layers layers).
             Time is flattened into the batch dimension so each layer sees
             only (v_t, a_t, x_t) at each individual timestep — no temporal
             mixing.  For each modality, multi-head attention is applied with
             that modality as query and the other two as keys/values, followed
             by a position-wise FFN (pre-norm, residual throughout).  Absent
             text positions are excluded from the key/value sequence via
             key_padding_mask so they cannot contaminate video and audio.

  Stage 4 — Concat aggregation.
             The three enriched streams are concatenated.  Absent text
             positions are already zeroed by Stage 3.

Interface contract (same as every other preprocessor):
    forward(text, audio, video) -> (B, T, 3 * d_model)
    .d_out : int   — output feature dimension (= 3 * d_model)

Config key:
    preprocessor:
      type: "trifuse"
      d_out: 256          # d_model inside TriFuse; backbone n_in = 3 * d_out
      n_heads: 4
      n_unimodal_layers: 2
      n_fusion_layers: 4
      dropout: 0.1
      mask_absent_text: true   # set false to treat text like the other modalities
"""
import torch
from torch import nn


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 building block
# ──────────────────────────────────────────────────────────────────────────────

class PositionwiseFFN(nn.Module):
    """
    Pre-norm position-wise feed-forward block.

    Applied identically and independently at every timestep — no information
    flows between positions.  Multiple instances can be stacked for depth.

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


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 building block
# ──────────────────────────────────────────────────────────────────────────────

class CrossModalAttentionLayer(nn.Module):
    """
    One layer of position-wise cross-modal attention (Stage 3).

    For each modality, multi-head attention is applied with that modality as
    query and the other two modalities as keys/values, followed by a
    position-wise FFN.  All sub-layers use pre-norm residual connections.

    When mask_x is provided (not None), absent text positions are excluded
    from the KV sequence for video/audio queries, and text outputs are
    re-zeroed at absent positions.  Pass mask_x=None to disable all text
    masking and treat text identically to the other modalities.

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
        v: torch.Tensor,                        # (BT, 1, D)
        a: torch.Tensor,                        # (BT, 1, D)
        x: torch.Tensor,                        # (BT, 1, D)
        mask_x: "torch.Tensor | None" = None,  # (BT, 1) float, 1 present / 0 absent
    ):
        """Returns: (v, a, x) with the same shapes as the inputs."""
        # Normalise each modality once; reuse for both Q and KV roles.
        v_n = self.ln_v(v)
        a_n = self.ln_a(a)
        x_n = self.ln_x(x)

        # Use pre-update (normalised) features as KV — no ordering bias.
        kv_ax = torch.cat([a_n, x_n], dim=1)   # (BT, 2, D) — KV for video
        kv_vx = torch.cat([v_n, x_n], dim=1)   # (BT, 2, D) — KV for audio
        kv_va = torch.cat([v_n, a_n], dim=1)   # (BT, 2, D) — KV for text

        if mask_x is not None:
            # Key-padding masks for 2-token KV sequences.
            # key_padding_mask: True = ignore that key.
            # For video/audio queries: mask the text key when absent.
            # For text query: KV = [v, a]; never masked.
            absent = mask_x.squeeze(1).eq(0)             # (BT,) bool
            false_ = torch.zeros_like(absent)
            kv_mask_va = torch.stack([false_, absent], dim=1)   # (BT, 2)
            kv_mask_none = None
        else:
            kv_mask_va = kv_mask_none = None

        v_attn, _ = self.attn_v(v_n, kv_ax, kv_ax, key_padding_mask=kv_mask_va)
        a_attn, _ = self.attn_a(a_n, kv_vx, kv_vx, key_padding_mask=kv_mask_va)
        x_attn, _ = self.attn_x(x_n, kv_va, kv_va, key_padding_mask=kv_mask_none)

        v = self.ffn_v(v + v_attn)
        a = self.ffn_a(a + a_attn)
        x_out = self.ffn_x(x + x_attn)
        if mask_x is not None:
            x_out = x_out * mask_x.unsqueeze(-1)   # re-zero absent text positions

        return v, a, x_out


# ──────────────────────────────────────────────────────────────────────────────
# Full TriFuse preprocessor
# ──────────────────────────────────────────────────────────────────────────────

class TriFusePreprocessor(nn.Module):
    """
    Trimodal cross-modal attention preprocessor.

    All three modality inputs arrive pre-aligned at (B, T, D_m).
    The text (transcript) modality is sparse: timesteps with no speech are
    zero vectors.  TriFuse handles this explicitly at every attention and
    aggregation step.

    No positional encoding is applied — temporal context aggregation is left
    entirely to the backbone, which adds its own PE.

    Args:
        text_dim          : Native text feature dimension.
        audio_dim         : Native audio feature dimension.
        video_dim         : Native video feature dimension.
        d_model           : Shared internal dimension.
        n_heads           : Number of attention heads (must divide d_model).
        n_unimodal_layers : Number of per-modality position-wise FFN layers (Stage 2).
        n_fusion_layers   : Number of cross-modal attention layers (Stage 3).
        dropout           : Dropout probability throughout.
        modality_dropout  : Probability of zeroing an entire modality per sample
                            during training (independent per modality).  If all
                            three would be dropped, all are kept instead.
                            Default 0.0 (disabled).
        mask_absent_text  : If True (default), absent text timesteps (zero input
                            vectors) are tracked with a presence mask and re-zeroed
                            after every stage that could corrupt them, and excluded
                            from KV sequences in Stage 3.  Set to False to treat
                            text identically to video and audio — the projection
                            bias at absent positions will produce non-zero features,
                            which the model must learn to ignore or exploit.

    Output shape: (B, T, 3 * d_model)
    Attribute   : d_out = 3 * d_model
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        video_dim: int,
        d_model: int,
        n_heads: int = 8,
        n_unimodal_layers: int = 2,
        n_fusion_layers: int = 4,
        dropout: float = 0.1,
        modality_dropout: float = 0.0,
        mask_absent_text: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model           = d_model
        self.d_out             = 3 * d_model   # Stage 4 concat — no projection
        self.modality_dropout  = modality_dropout
        self.mask_absent_text  = mask_absent_text

        # ── Stage 1: projections ──────────────────────────────────────────────
        self.proj_v = nn.Linear(video_dim, d_model)
        self.proj_a = nn.Linear(audio_dim, d_model)
        self.proj_x = nn.Linear(text_dim,  d_model)

        # ── Stage 2: per-modality position-wise FFN ───────────────────────────
        self.ffn_v = nn.Sequential(*[PositionwiseFFN(d_model, dropout) for _ in range(n_unimodal_layers)])
        self.ffn_a = nn.Sequential(*[PositionwiseFFN(d_model, dropout) for _ in range(n_unimodal_layers)])
        self.ffn_x = nn.Sequential(*[PositionwiseFFN(d_model, dropout) for _ in range(n_unimodal_layers)])

        # ── Stage 3: cross-modal attention ────────────────────────────────────
        self.fusion_layers = nn.ModuleList([
            CrossModalAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_fusion_layers)
        ])

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        text:  torch.Tensor,   # (B, T, text_dim)
        audio: torch.Tensor,   # (B, T, audio_dim)
        video: torch.Tensor,   # (B, T, video_dim)
    ) -> torch.Tensor:         # (B, T, 3 * d_model)
        B, T, _ = video.shape

        # ── Modality dropout (training only) ──────────────────────────────────
        if self.training and self.modality_dropout > 0.0:
            # Mirrors _apply_modality_dropout in feature_preprocessors.py.
            # Cannot import it here (that module imports us — circular).
            keep = torch.bernoulli(
                torch.full((B, 3), 1.0 - self.modality_dropout, device=video.device)
            )
            keep[keep.sum(dim=1) == 0] = 1.0   # never drop all modalities
            video = video * keep[:, 0].view(B, 1, 1)
            audio = audio * keep[:, 1].view(B, 1, 1)
            text  = text  * keep[:, 2].view(B, 1, 1)

        # ── Stage 1 ───────────────────────────────────────────────────────────
        if self.mask_absent_text:
            # Derive presence mask BEFORE projection so the bias cannot corrupt it.
            mask_x = (text.norm(dim=-1) > 1e-6).float()   # (B, T)

        v = self.proj_v(video)
        a = self.proj_a(audio)
        x = self.proj_x(text)

        if self.mask_absent_text:
            x = x * mask_x.unsqueeze(-1)   # re-zero: proj bias corrupts zero inputs

        # ── Stage 2 ───────────────────────────────────────────────────────────
        v = self.ffn_v(v)
        a = self.ffn_a(a)
        x = self.ffn_x(x)

        if self.mask_absent_text:
            x = x * mask_x.unsqueeze(-1)   # re-zero: LN beta corrupts zero inputs

        # ── Stage 3 ───────────────────────────────────────────────────────────
        # Flatten time into batch so each layer operates position-wise.
        D = self.d_model
        v_pw = v.reshape(B * T, 1, D)
        a_pw = a.reshape(B * T, 1, D)
        x_pw = x.reshape(B * T, 1, D)
        mask_pw = mask_x.reshape(B * T, 1) if self.mask_absent_text else None

        for layer in self.fusion_layers:
            v_pw, a_pw, x_pw = layer(v_pw, a_pw, x_pw, mask_pw)

        v = v_pw.reshape(B, T, D)
        a = a_pw.reshape(B, T, D)
        x = x_pw.reshape(B, T, D)

        # ── Stage 4 ───────────────────────────────────────────────────────────
        return torch.cat([v, a, x], dim=-1)   # (B, T, 3 * d_model)