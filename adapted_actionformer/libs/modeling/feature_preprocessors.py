"""
Feature preprocessors — transform raw modality features before the backbone.

The preprocessor sits between the dataset-loaded raw features (text, audio, video)
and the backbone encoder.  Swapping preprocessors lets you run:

  - Fully multimodal experiments  (GuidedCMAPreprocessor — the default)
  - Unimodal ablations            (UnimodalPreprocessor  — single modality)
  - Simple fusion baselines       (ConcatPreprocessor    — concatenate then project)

All preprocessors share the same interface:

    forward(text, audio, video) -> (B, T, d_out)
    .d_out : int   — output feature dimension passed to the backbone

Config key: ``preprocessor``

    preprocessor:
      type: "cma"                      # GuidedCMAPreprocessor
      d_out: 256
      num_heads: 4
      dropout: 0.0
      query_modality: "text"           # "text" | "audio" | "video"  (default: "text")
      kv_modalities: ["audio", "video"]  # default: the other two modalities
      zero_out_missing_query: true     # zero output where query feature is absent

    preprocessor:
      type: "unimodal"  # UnimodalPreprocessor
      modality: "video" # one of "text" | "audio" | "video"
      d_out: 256

    preprocessor:
      type: "concat"    # ConcatPreprocessor
      modalities: ["audio", "video"]
      d_out: 256

Backward compatibility
----------------------
If the config has a ``fusion`` key but no ``preprocessor`` key, the ``fusion``
block is treated as a CMA preprocessor config (using ``d_cma`` as ``d_out``).
"""
import torch
from torch import nn


# ──────────────────────────────────────────────────────────────────────────────
# Shared modality-dropout helper
# ──────────────────────────────────────────────────────────────────────────────

def _apply_modality_dropout(text, audio, video, active_modalities, p):
    """
    Zero entire modalities independently with probability *p* (training only).

    Args:
        text / audio / video : Raw feature tensors, each (B, T, D_m).
        active_modalities    : Ordered list of modality names actually used by
                               the preprocessor (subset of "text","audio","video").
        p                    : Drop probability per modality per sample.

    Returns:
        (text, audio, video) with dropped modalities zeroed out.
        Safe fallback: if all active modalities would be dropped for a sample,
        all are kept instead.
    """
    B = text.shape[0]
    feat_map = {'text': text, 'audio': audio, 'video': video}
    n = len(active_modalities)
    keep = torch.bernoulli(
        torch.full((B, n), 1.0 - p, device=text.device)
    )
    # Prevent all-modalities-dropped for any sample
    all_dropped = keep.sum(dim=1) == 0          # (B,)
    keep[all_dropped] = 1.0
    for i, m in enumerate(active_modalities):
        feat_map[m] = feat_map[m] * keep[:, i].view(B, 1, 1)
    return feat_map['text'], feat_map['audio'], feat_map['video']


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessor modules
# ──────────────────────────────────────────────────────────────────────────────

class GuidedCMAPreprocessor(nn.Module):
    """
    Cross-modal attention fusion with a configurable query modality.

    For each timestep t:
        Q  = proj_query(query_feat[t])             — 1 query token
        KV = [proj_m(feat_m[t]) for m in kv_modalities]  — K key/value tokens
        out[t] = MultiHeadAttention(Q, KV, KV)
        out[t] *= query_presence_mask[t]           — optional: zero-out absent query

    The batch operation folds T into the batch axis to avoid Python loops.

    Args:
        text_dim              : Input dimension of text features  (default 768).
        audio_dim             : Input dimension of audio features (default 1024).
        video_dim             : Input dimension of video features (default 768).
        d_out                 : Output dimension (= backbone input dimension).
        num_heads             : Number of attention heads for the MHA layer.
        dropout               : Dropout probability inside MHA.
        query_modality        : Which modality acts as the query: "text" | "audio" | "video".
                                Defaults to "text".
        kv_modalities         : List of modalities used as key/value.
                                Defaults to the two modalities that are not the query.
        zero_out_missing_query: If True, zero the output at timesteps where the query
                                feature vector is all-zeros (e.g. silent frames when
                                text is the query).  Defaults to True.
        modality_dropout      : Probability of zeroing an entire modality for a sample
                                during training.  Default 0.0 (disabled).

    Output shape: (B, T, d_out)
    """

    def __init__(
        self,
        text_dim,
        audio_dim,
        video_dim,
        d_out,
        num_heads,
        dropout=0.1,
        query_modality="text",
        kv_modalities=None,
        zero_out_missing_query=True,
        modality_dropout=0.0,
    ):
        super().__init__()
        assert query_modality in ("text", "audio", "video"), (
            f"query_modality must be 'text', 'audio', or 'video', got '{query_modality}'"
        )
        self.d_out = d_out
        self.query_modality = query_modality
        self.kv_modalities = (
            list(kv_modalities)
            if kv_modalities is not None
            else [m for m in ("text", "audio", "video") if m != query_modality]
        )
        assert len(self.kv_modalities) >= 1, "kv_modalities must not be empty"
        self.zero_out_missing_query = zero_out_missing_query
        self.modality_dropout = modality_dropout

        dim_map = {"text": text_dim, "audio": audio_dim, "video": video_dim}
        all_modalities = set([query_modality] + self.kv_modalities)
        # Stable ordering for modality_dropout indexing
        self.active_modalities = sorted(all_modalities)
        self.projs = nn.ModuleDict({
            m: nn.Linear(dim_map[m], d_out) for m in all_modalities
        })

        self.cma = nn.MultiheadAttention(
            d_out, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, text, audio, video):
        """
        Args:
            text  : (B, T, text_dim)
            audio : (B, T, audio_dim)
            video : (B, T, video_dim)
        Returns:
            fused : (B, T, d_out)
        """
        if self.training and self.modality_dropout > 0.0:
            text, audio, video = _apply_modality_dropout(
                text, audio, video, self.active_modalities, self.modality_dropout
            )

        feat_map = {"text": text, "audio": audio, "video": video}
        B, T, _ = feat_map[self.query_modality].shape

        # Optional mask: 1 where query feature is non-zero, 0 where absent
        if self.zero_out_missing_query:
            query_raw = feat_map[self.query_modality]
            query_mask = (query_raw.abs().sum(dim=-1) > 0).float()  # (B, T)

        # Project query and KV modalities to the common CMA space
        q_proj = self.projs[self.query_modality](feat_map[self.query_modality])  # (B, T, d_out)
        kv_projs = [self.projs[m](feat_map[m]) for m in self.kv_modalities]      # each (B, T, d_out)

        # Fold T into batch for per-timestep attention
        n_kv = len(self.kv_modalities)
        Q  = q_proj.reshape(B * T, 1, self.d_out)                               # (B*T, 1, d_out)
        KV = torch.stack(kv_projs, dim=2).reshape(B * T, n_kv, self.d_out)      # (B*T, n_kv, d_out)

        out, _ = self.cma(Q, KV, KV)                                             # (B*T, 1, d_out)
        out = out.reshape(B, T, self.d_out)                                      # (B, T, d_out)

        # Zero-out timesteps where query is absent
        if self.zero_out_missing_query:
            out = out * query_mask.unsqueeze(-1)
        return out


class UnimodalPreprocessor(nn.Module):
    """
    Passes a single chosen modality to the backbone.

    The raw feature is linearly projected to ``d_out`` if its native dimension
    differs.  Use this for unimodal ablations — set ``modality`` to one of
    ``"text"``, ``"audio"``, or ``"video"``.

    Args:
        modality  : Which modality to use: ``"text"`` | ``"audio"`` | ``"video"``.
        text_dim  : Native dimension of text features.
        audio_dim : Native dimension of audio features.
        video_dim : Native dimension of video features.
        d_out     : Output dimension (= backbone input dimension).
                    If equal to the native feature dim, no projection is applied.

    Output shape: (B, T, d_out)
    """

    def __init__(self, modality, text_dim, audio_dim, video_dim, d_out):
        super().__init__()
        assert modality in ('text', 'audio', 'video'), (
            f"modality must be 'text', 'audio', or 'video', got '{modality}'"
        )
        self.modality = modality
        self.d_out    = d_out

        dim_map = {'text': text_dim, 'audio': audio_dim, 'video': video_dim}
        in_dim  = dim_map[modality]
        self.proj = nn.Linear(in_dim, d_out) if in_dim != d_out else nn.Identity()

    def forward(self, text, audio, video):
        """
        Args:
            text  : (B, T, text_dim)
            audio : (B, T, audio_dim)
            video : (B, T, video_dim)
        Returns:
            out : (B, T, d_out)
        """
        feat_map = {'text': text, 'audio': audio, 'video': video}
        return self.proj(feat_map[self.modality])


class MultiHateLocPreprocessor(nn.Module):
    """
    MA-TE + DCM-Fusion pipeline from MultiHateLoc (Sun et al., WWW 2026).

    Stage 1 — Modality-Aware Temporal Encoding (MA-TE), Section 3.2:
        For each modality m ∈ {video, audio, text}:
          1. Linear project raw features → common dim D.
          2. Pre-norm self-attention + residual.
          3. Pre-norm FFN (Linear(D,4D)→ReLU→Linear(4D,D)) + residual.
        All weights are modality-specific and learned independently.

    Stage 2a — Dynamic Modality Selection (DMS), Eq. 3-4:
        α_m[t] = sigmoid(W_α^m · F'_m[t])   # scalar gate per timestep
        F_m^weighted[t] = α_m[t] · F'_m[t]

    Stage 2b — Cross-Modal Attention (CMA), Eq. 5-7:
        F_concat = Concat(F_v^w, F_a^w, F_l^w)   ∈ R^{T×3D}
        Q = W_q · F_concat  (external proj 3D→D)
        F_fused = MHA(Q, F_concat, F_concat)      ∈ R^{T×D}
        (MHA has embed_dim=D, kdim=3D, vdim=3D)

    Stage 3 — Output:
        out = Linear(Concat(F'_v, F'_a, F'_l, F_fused))   ∈ R^{T×d_out}

    Args:
        text_dim         : Native text feature dimension (e.g. 768 for BERT/HateBERT).
        audio_dim        : Native audio feature dimension (e.g. 1024 for Wav2Vec2).
        video_dim        : Native video feature dimension (e.g. 768 for ViT-B/16).
        d_out            : Output dimension passed to the backbone.
        n_heads          : Attention heads for both MA-TE self-attention and CMA.
                           Must divide d_inner evenly.
        dropout          : Dropout probability inside all attention layers.
        d_inner          : Common internal dimension D used throughout the module.
                           Default 256.
        modality_dropout : Probability of zeroing an entire modality for a sample
                           during training.  Applied independently per modality per
                           sample.  If all three modalities would be dropped for a
                           sample, all are kept instead (safe fallback).
                           Default 0.0 (disabled).

    Output shape: (B, T, d_out)
    """

    def __init__(
        self,
        text_dim,
        audio_dim,
        video_dim,
        d_out,
        n_heads=4,
        dropout=0.1,
        d_inner=256,
        modality_dropout=0.0,
    ):
        super().__init__()
        assert d_inner % n_heads == 0, (
            f"d_inner ({d_inner}) must be divisible by n_heads ({n_heads})"
        )
        self.d_out            = d_out
        self.d_inner          = d_inner
        self.modality_dropout = modality_dropout
        D = d_inner

        # ── Stage 1: MA-TE input projections ──────────────────────────────────
        self.proj_text  = nn.Linear(text_dim,  D)
        self.proj_audio = nn.Linear(audio_dim, D)
        self.proj_video = nn.Linear(video_dim, D)

        # Per-modality pre-norm self-attention (text)
        self.ln_attn_text  = nn.LayerNorm(D)
        self.attn_text     = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.ln_ffn_text   = nn.LayerNorm(D)
        self.ffn_text      = nn.Sequential(nn.Linear(D, 4 * D), nn.ReLU(), nn.Linear(4 * D, D))

        # Per-modality pre-norm self-attention (audio)
        self.ln_attn_audio = nn.LayerNorm(D)
        self.attn_audio    = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.ln_ffn_audio  = nn.LayerNorm(D)
        self.ffn_audio     = nn.Sequential(nn.Linear(D, 4 * D), nn.ReLU(), nn.Linear(4 * D, D))

        # Per-modality pre-norm self-attention (video)
        self.ln_attn_video = nn.LayerNorm(D)
        self.attn_video    = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.ln_ffn_video  = nn.LayerNorm(D)
        self.ffn_video     = nn.Sequential(nn.Linear(D, 4 * D), nn.ReLU(), nn.Linear(4 * D, D))

        # ── Stage 2a: DMS — scalar gate per timestep per modality ──────────────
        self.dms_text  = nn.Linear(D, 1)
        self.dms_audio = nn.Linear(D, 1)
        self.dms_video = nn.Linear(D, 1)

        # ── Stage 2b: CMA — cross-modal attention over concatenated features ───
        # External Q projection: 3D → D  (W_q in the paper)
        self.q_proj = nn.Linear(3 * D, D)
        # MHA with kdim=3D, vdim=3D handles W_k and W_v (3D → D internally)
        self.cma = nn.MultiheadAttention(
            D, n_heads, kdim=3 * D, vdim=3 * D, dropout=dropout, batch_first=True
        )

        # ── Stage 3: output projection ─────────────────────────────────────────
        # Concat(F'_v, F'_a, F'_l, F_fused) has dim 4D → d_out
        self.out_proj = nn.Linear(4 * D, d_out) if 4 * D != d_out else nn.Identity()

    def _mate_block(self, F, ln_attn, attn, ln_ffn, ffn):
        """Single modality MA-TE block: pre-norm attention + pre-norm FFN."""
        F_norm = ln_attn(F)
        F_attn, _ = attn(F_norm, F_norm, F_norm)
        F = F_attn + F                          # residual
        F_norm2 = ln_ffn(F)
        F_prime = ffn(F_norm2) + F              # residual
        return F_prime

    def forward(self, text, audio, video):
        """
        Args:
            text  : (B, T, text_dim)
            audio : (B, T, audio_dim)
            video : (B, T, video_dim)
        Returns:
            out : (B, T, d_out)
        """
        # ── Modality dropout (training only) ─────────────────────────────────
        if self.training and self.modality_dropout > 0.0:
            text, audio, video = _apply_modality_dropout(
                text, audio, video, ['audio', 'text', 'video'], self.modality_dropout
            )

        # ── Stage 1: MA-TE ────────────────────────────────────────────────────
        Ft = self.proj_text(text)    # (B, T, D)
        Fa = self.proj_audio(audio)  # (B, T, D)
        Fv = self.proj_video(video)  # (B, T, D)

        Ft_prime = self._mate_block(Ft, self.ln_attn_text,  self.attn_text,
                                        self.ln_ffn_text,   self.ffn_text)   # (B, T, D)
        Fa_prime = self._mate_block(Fa, self.ln_attn_audio, self.attn_audio,
                                        self.ln_ffn_audio,  self.ffn_audio)  # (B, T, D)
        Fv_prime = self._mate_block(Fv, self.ln_attn_video, self.attn_video,
                                        self.ln_ffn_video,  self.ffn_video)  # (B, T, D)

        # ── Stage 2a: DMS ─────────────────────────────────────────────────────
        alpha_t = torch.sigmoid(self.dms_text(Ft_prime))   # (B, T, 1)
        alpha_a = torch.sigmoid(self.dms_audio(Fa_prime))  # (B, T, 1)
        alpha_v = torch.sigmoid(self.dms_video(Fv_prime))  # (B, T, 1)

        Ft_w = alpha_t * Ft_prime   # (B, T, D)
        Fa_w = alpha_a * Fa_prime   # (B, T, D)
        Fv_w = alpha_v * Fv_prime   # (B, T, D)

        # ── Stage 2b: CMA ─────────────────────────────────────────────────────
        F_concat = torch.cat([Fv_w, Fa_w, Ft_w], dim=-1)           # (B, T, 3D)
        Q = self.q_proj(F_concat)                                    # (B, T, D)
        F_fused, _ = self.cma(Q, F_concat, F_concat)                # (B, T, D)

        # ── Stage 3: output projection ─────────────────────────────────────────
        F_all = torch.cat([Fv_prime, Fa_prime, Ft_prime, F_fused], dim=-1)  # (B, T, 4D)
        return self.out_proj(F_all)                                           # (B, T, d_out)


class ConcatPreprocessor(nn.Module):
    """
    Concatenates the chosen modalities and projects to ``d_out``.

    A simple fusion baseline that does not use attention — concatenation + a
    single learned linear projection.  Set ``modalities`` to any non-empty
    subset of ``["text", "audio", "video"]``.

    Args:
        modalities       : List of modality names to concatenate, e.g. ``["audio", "video"]``.
        text_dim         : Native dimension of text features.
        audio_dim        : Native dimension of audio features.
        video_dim        : Native dimension of video features.
        d_out            : Output dimension (= backbone input dimension).
        modality_dropout : Probability of zeroing an entire modality for a sample
                           during training.  Default 0.0 (disabled).

    Output shape: (B, T, d_out)
    """

    def __init__(self, modalities, text_dim, audio_dim, video_dim, d_out,
                 modality_dropout=0.0):
        super().__init__()
        assert len(modalities) >= 1, "modalities must not be empty"
        for m in modalities:
            assert m in ('text', 'audio', 'video'), (
                f"Unknown modality '{m}'. Expected one of: 'text', 'audio', 'video'."
            )
        self.modalities       = list(modalities)
        self.d_out            = d_out
        self.modality_dropout = modality_dropout

        dim_map = {'text': text_dim, 'audio': audio_dim, 'video': video_dim}
        in_dim  = sum(dim_map[m] for m in modalities)
        self.proj = nn.Linear(in_dim, d_out)

    def forward(self, text, audio, video):
        """
        Args:
            text  : (B, T, text_dim)
            audio : (B, T, audio_dim)
            video : (B, T, video_dim)
        Returns:
            out : (B, T, d_out)
        """
        if self.training and self.modality_dropout > 0.0:
            text, audio, video = _apply_modality_dropout(
                text, audio, video, self.modalities, self.modality_dropout
            )
        feat_map = {'text': text, 'audio': audio, 'video': video}
        feats = [feat_map[m] for m in self.modalities]
        return self.proj(torch.cat(feats, dim=-1))


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_preprocessor(cfg, text_dim, audio_dim, video_dim):
    """
    Instantiate the correct preprocessor from the model config dict.

    Looks for a ``preprocessor`` key first.  If absent, falls back to the
    legacy ``fusion`` key and treats it as a CMA preprocessor config.

    Parameters
    ----------
    cfg       : Full model config dict (as loaded from YAML).
    text_dim  : Native dimension of text features.
    audio_dim : Native dimension of audio features.
    video_dim : Native dimension of video features.

    Returns
    -------
    module : nn.Module
        One of GuidedCMAPreprocessor, UnimodalPreprocessor, ConcatPreprocessor,
        MultiHateLocPreprocessor.
    d_out : int
        Output feature dimension (passed as ``n_in`` to the backbone).
    """
    prep_cfg = cfg.get('preprocessor', None)

    # ── Backward-compatible: legacy ``fusion`` block ──────────────────────────
    if prep_cfg is None:
        fus_cfg = cfg.get('fusion', {})
        d_out   = fus_cfg.get('d_cma', 128)
        module  = GuidedCMAPreprocessor(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            d_out=d_out,
            num_heads=fus_cfg.get('num_heads', 2),
            dropout=fus_cfg.get('dropout', 0.1),
            query_modality=fus_cfg.get('query_modality', 'text'),
            kv_modalities=fus_cfg.get('key_modalities', None),
            zero_out_missing_query=fus_cfg.get('zero_out_missing_query', True),
        )
        return module, d_out

    # ── New ``preprocessor`` block ─────────────────────────────────────────────
    ptype = prep_cfg.get('type', 'cma')
    d_out = prep_cfg['d_out']

    if ptype == 'cma':
        module = GuidedCMAPreprocessor(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            d_out=d_out,
            num_heads=prep_cfg.get('num_heads', 2),
            dropout=prep_cfg.get('dropout', 0.1),
            query_modality=prep_cfg.get('query_modality', 'text'),
            kv_modalities=prep_cfg.get('kv_modalities', None),
            zero_out_missing_query=prep_cfg.get('zero_out_missing_query', True),
            modality_dropout=prep_cfg.get('modality_dropout', 0.0),
        )

    elif ptype == 'unimodal':
        module = UnimodalPreprocessor(
            modality=prep_cfg['modality'],
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            d_out=d_out,
        )

    elif ptype == 'concat':
        module = ConcatPreprocessor(
            modalities=prep_cfg['modalities'],
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            d_out=d_out,
            modality_dropout=prep_cfg.get('modality_dropout', 0.0),
        )

    elif ptype == 'multihateloc':
        module = MultiHateLocPreprocessor(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            d_out=d_out,
            n_heads=prep_cfg.get('n_heads', 4),
            dropout=prep_cfg.get('dropout', 0.1),
            d_inner=prep_cfg.get('d_inner', 256),
            modality_dropout=prep_cfg.get('modality_dropout', 0.0),
        )

    elif ptype == 'trifuse':
        from .trifuse import TriFusePreprocessor
        module = TriFusePreprocessor(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            d_model=d_out,
            n_heads=prep_cfg.get('n_heads', 8),
            n_fusion_layers=prep_cfg.get('n_fusion_layers', 4),
            dropout=prep_cfg.get('dropout', 0.1),
            modality_dropout=prep_cfg.get('modality_dropout', 0.0),
        )

    else:
        raise ValueError(
            f"Unknown preprocessor type '{ptype}'. "
            "Expected one of: 'cma', 'unimodal', 'concat', 'multihateloc', 'trifuse'."
        )

    return module, module.d_out
