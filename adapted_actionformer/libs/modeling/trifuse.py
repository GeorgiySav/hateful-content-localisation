"""
TriFuse — Trimodal Bottleneck Fusion Preprocessor.

Architecture (4 stages):

  Stage 1 — Linear projection + modality embeddings + sinusoidal PE.
             Text is sparse: zero vectors where no speech is present.
             A presence mask is derived from the raw input and used to re-zero
             text features after every stage that could corrupt zero inputs.

  Stage 2 — Per-modality self-attention (L1 layers, default 2).
             Text self-attention uses a key-padding mask so absent timesteps
             neither attend to others nor are attended to.

  Stage 3 — Bottleneck cross-modal fusion (L2 layers, default 4).
             Learnable bottleneck tokens gather information from all modalities,
             refine it via self-attention, then distribute updates back through
             per-modality gated residuals.

  Stage 4 — Presence-conditioned gated aggregation.
             A small MLP weights the three enriched streams; the text weight is
             forced to 0 at absent timesteps via masked softmax.

Interface contract (same as every other preprocessor):
    forward(text, audio, video) -> (B, T, d_model)
    .d_out : int   — output feature dimension (= d_model)

Config key:
    preprocessor:
      type: "trifuse"
      d_out: 256          # d_model inside TriFuse and input to the backbone
      n_heads: 4
      n_bottleneck: 4
      n_unimodal_layers: 2
      n_fusion_layers: 4
      dropout: 0.1
"""
import torch
import torch.nn.functional as F
from torch import nn

from .blocks import get_sinusoid_encoding


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 building block
# ──────────────────────────────────────────────────────────────────────────────

class BottleneckFusionLayer(nn.Module):
    """
    One layer of the bottleneck cross-modal fusion (Stage 3).

    Each layer has three sub-steps:

      3a) Gather  — bottleneck tokens cross-attend to each modality.
                    Text cross-attention is mask-aware: positions where text is
                    absent are excluded from the key set.
      3b) Refine  — self-attention + FFN on the bottleneck tokens.
      3c) Distribute — each modality cross-attends back to the bottleneck;
                       updates are applied through a learned sigmoid gate.
                       Text is hard-re-zeroed after distribution.

    Args:
        d_model   : Common feature dimension.
        n_heads   : Number of attention heads.
        dropout   : Dropout probability inside attention layers and FFN.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        D = d_model

        # ── 3a Gather ─────────────────────────────────────────────────────────
        self.gather_v = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.gather_a = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.gather_x = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.ln_gather = nn.LayerNorm(D)

        # ── 3b Refine ─────────────────────────────────────────────────────────
        self.self_attn_b = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.ln_b1       = nn.LayerNorm(D)
        self.ffn         = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * D, D),
        )
        self.ln_b2 = nn.LayerNorm(D)

        # ── 3c Distribute ─────────────────────────────────────────────────────
        self.dist_v  = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.dist_a  = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.dist_x  = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.gate_v  = nn.Linear(2 * D, D)
        self.gate_a  = nn.Linear(2 * D, D)
        self.gate_x  = nn.Linear(2 * D, D)

    def forward(
        self,
        b: torch.Tensor,       # (B, N_b, D)
        v: torch.Tensor,       # (B, T, D)
        a: torch.Tensor,       # (B, T, D)
        x: torch.Tensor,       # (B, T, D)  — sparse text
        mask_x: torch.Tensor,  # (B, T)     — 1 where text is present, 0 absent
    ):
        """
        Returns: (b, v, a, x) all with the same shapes as the inputs.
        """
        absent = ~mask_x.bool()   # True where text is absent — used as key_padding_mask

        # Safety: for samples where ALL text positions are absent (fully silent
        # videos), key_padding_mask=all_True causes softmax(−∞, …) = NaN.
        # nan_to_num(0.0) fixes the forward value, but the NaN statistics stored
        # inside LayerNorm produce NaN gradients for the LN gamma parameter,
        # which progressively corrupts training.  Clearing the mask for those
        # samples lets attention run on the (all-zero) features; the mandatory
        # `x * mask_x` re-zero at the end of this layer ensures no leakage.
        all_absent = absent.all(dim=1)   # (B,)
        if all_absent.any():
            absent = absent.clone()
            absent[all_absent] = False   # drop mask for fully-silent samples

        # ── 3a Gather ─────────────────────────────────────────────────────────
        b = b + self.gather_v(b, v, v)[0]
        b = b + self.gather_a(b, a, a)[0]

        delta_x = self.gather_x(b, x, x, key_padding_mask=absent)[0]
        b = b + delta_x.nan_to_num(0.0)

        b = self.ln_gather(b)

        # ── 3b Refine ─────────────────────────────────────────────────────────
        b = self.ln_b1(b + self.self_attn_b(b, b, b)[0])
        b = self.ln_b2(b + self.ffn(b))

        # ── 3c Distribute ─────────────────────────────────────────────────────
        # Video
        dv   = self.dist_v(v, b, b)[0]
        gate = torch.sigmoid(self.gate_v(torch.cat([v, dv], dim=-1)))
        v    = v + gate * dv

        # Audio
        da   = self.dist_a(a, b, b)[0]
        gate = torch.sigmoid(self.gate_a(torch.cat([a, da], dim=-1)))
        a    = a + gate * da

        # Text — gate + hard re-zero for absent positions
        dx   = self.dist_x(x, b, b)[0]
        gate = torch.sigmoid(self.gate_x(torch.cat([x, dx], dim=-1)))
        x    = x + gate * dx
        x    = x * mask_x.unsqueeze(-1)   # re-zero absent timesteps

        return b, v, a, x


# ──────────────────────────────────────────────────────────────────────────────
# Full TriFuse preprocessor
# ──────────────────────────────────────────────────────────────────────────────

class TriFusePreprocessor(nn.Module):
    """
    Trimodal bottleneck fusion preprocessor.

    All three modality inputs arrive pre-aligned at (B, T, D_m).
    The text (transcript) modality is sparse: timesteps with no speech are
    zero vectors.  TriFuse handles this explicitly at every attention and
    aggregation step.

    Args:
        text_dim          : Native text feature dimension.
        audio_dim         : Native audio feature dimension.
        video_dim         : Native video feature dimension.
        d_model           : Shared internal and output dimension.
        n_heads           : Number of attention heads (must divide d_model).
        n_bottleneck      : Number of learnable bottleneck tokens.
        n_unimodal_layers : Number of per-modality self-attention layers (Stage 2).
        n_fusion_layers   : Number of bottleneck fusion layers (Stage 3).
        dropout           : Dropout probability throughout.

    Output shape: (B, T, d_model)
    Attribute   : d_out = d_model
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        video_dim: int,
        d_model: int,
        n_heads: int = 8,
        n_bottleneck: int = 4,
        n_unimodal_layers: int = 2,
        n_fusion_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model  = d_model
        self.d_out    = d_model
        self._pe_cache: torch.Tensor | None = None   # lazy sinusoidal PE

        # ── Stage 1: projections + embeddings ─────────────────────────────────
        self.proj_v = nn.Linear(video_dim, d_model)
        self.proj_a = nn.Linear(audio_dim, d_model)
        self.proj_x = nn.Linear(text_dim,  d_model)

        self.mod_emb_v = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mod_emb_a = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mod_emb_x = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ── Stage 2: per-modality self-attention ─────────────────────────────
        enc_layer_v = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=False,
        )
        enc_layer_a = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=False,
        )
        enc_layer_x = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=False,
        )
        self.self_attn_v = nn.TransformerEncoder(enc_layer_v, num_layers=n_unimodal_layers)
        self.self_attn_a = nn.TransformerEncoder(enc_layer_a, num_layers=n_unimodal_layers)
        # enable_nested_tensor=False: the nested-tensor fast path crashes when ALL
        # text positions are masked (empty nested tensor). This is safe to disable
        # and has no effect on correctness.
        self.self_attn_x = nn.TransformerEncoder(
            enc_layer_x, num_layers=n_unimodal_layers, enable_nested_tensor=False,
        )

        # ── Stage 3: bottleneck fusion ────────────────────────────────────────
        self.bottleneck   = nn.Parameter(torch.randn(1, n_bottleneck, d_model) * 0.02)
        self.fusion_layers = nn.ModuleList([
            BottleneckFusionLayer(d_model, n_heads, dropout)
            for _ in range(n_fusion_layers)
        ])

        # ── Stage 4: presence-conditioned gated aggregation ───────────────────
        self.agg_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    # ── Sinusoidal PE (lazy, cached) ──────────────────────────────────────────

    def _get_pe(self, T: int, device: torch.device) -> torch.Tensor:
        """Return sinusoidal PE of shape (1, T, d_model), recomputed when T grows."""
        if self._pe_cache is None or self._pe_cache.size(1) < T:
            # get_sinusoid_encoding returns (1, d_model, n_position) → permute
            pe = get_sinusoid_encoding(T, self.d_model)   # (1, d_model, T)
            pe = pe.permute(0, 2, 1)                       # (1, T, d_model)
            self._pe_cache = pe.to(device)
        return self._pe_cache[:, :T, :].to(device)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        text:  torch.Tensor,   # (B, T, text_dim)
        audio: torch.Tensor,   # (B, T, audio_dim)
        video: torch.Tensor,   # (B, T, video_dim)
    ) -> torch.Tensor:         # (B, T, d_model)
        B, T, _ = video.shape

        # ── Stage 1 ───────────────────────────────────────────────────────────
        # Derive text presence mask BEFORE projection.
        mask_x = (text.norm(dim=-1) > 1e-6).float()   # (B, T)

        pe = self._get_pe(T, video.device)             # (1, T, d_model)

        v = self.proj_v(video) + self.mod_emb_v + pe
        a = self.proj_a(audio) + self.mod_emb_a + pe
        x = self.proj_x(text)  + self.mod_emb_x + pe

        # Re-zero text: projection bias / mod_emb / PE would produce non-zero
        # features even for zero-input timesteps.
        x = x * mask_x.unsqueeze(-1)

        # ── Stage 2 ───────────────────────────────────────────────────────────
        absent = ~mask_x.bool()   # True where text is absent

        # Safety: fully-silent samples have absent=all_True → softmax(−∞) = NaN.
        # NaN forward values are safe (nan_to_num + re-zero), but NaN is stored
        # inside LayerNorm as statistics, making g_gamma = 0 * NaN = NaN during
        # backward and corrupting LN weights.  For those samples, drop the mask;
        # the re-zero below (`x * mask_x`) still produces zero output, so the
        # model behaviour is identical for fully-silent videos.
        all_absent = absent.all(dim=1)   # (B,)
        absent_stage2 = absent.clone()
        if all_absent.any():
            absent_stage2[all_absent] = False

        v = self.self_attn_v(v)
        a = self.self_attn_a(a)
        x = self.self_attn_x(x, src_key_padding_mask=absent_stage2)
        x = x.nan_to_num(0.0)
        x = x * mask_x.unsqueeze(-1)   # re-zero after attention

        # ── Stage 3 ───────────────────────────────────────────────────────────
        b = self.bottleneck.expand(B, -1, -1)   # (B, N_b, d_model)
        for layer in self.fusion_layers:
            b, v, a, x = layer(b, v, a, x, mask_x)

        # ── Stage 4 ───────────────────────────────────────────────────────────
        logits = self.agg_mlp(torch.cat([v, a, x], dim=-1))   # (B, T, 3)

        # Force the text weight to −∞ (→ 0 after softmax) where text is absent,
        # so the aggregation degrades gracefully to bimodal video+audio.
        logits = logits.clone()
        logits[:, :, 2] = logits[:, :, 2].masked_fill(absent, float('-inf'))

        alpha = F.softmax(logits, dim=-1).nan_to_num(0.0)   # (B, T, 3)

        stacked = torch.stack([v, a, x], dim=2)              # (B, T, 3, d_model)
        fused   = (alpha.unsqueeze(-1) * stacked).sum(dim=2) # (B, T, d_model)

        return fused
