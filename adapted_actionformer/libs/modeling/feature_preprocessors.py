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

        dim_map = {"text": text_dim, "audio": audio_dim, "video": video_dim}
        all_modalities = set([query_modality] + self.kv_modalities)
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


class ConcatPreprocessor(nn.Module):
    """
    Concatenates the chosen modalities and projects to ``d_out``.

    A simple fusion baseline that does not use attention — concatenation + a
    single learned linear projection.  Set ``modalities`` to any non-empty
    subset of ``["text", "audio", "video"]``.

    Args:
        modalities : List of modality names to concatenate, e.g. ``["audio", "video"]``.
        text_dim   : Native dimension of text features.
        audio_dim  : Native dimension of audio features.
        video_dim  : Native dimension of video features.
        d_out      : Output dimension (= backbone input dimension).

    Output shape: (B, T, d_out)
    """

    def __init__(self, modalities, text_dim, audio_dim, video_dim, d_out):
        super().__init__()
        assert len(modalities) >= 1, "modalities must not be empty"
        for m in modalities:
            assert m in ('text', 'audio', 'video'), (
                f"Unknown modality '{m}'. Expected one of: 'text', 'audio', 'video'."
            )
        self.modalities = list(modalities)
        self.d_out      = d_out

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
        One of GuidedCMAPreprocessor, UnimodalPreprocessor, ConcatPreprocessor.
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
        )

    else:
        raise ValueError(
            f"Unknown preprocessor type '{ptype}'. "
            "Expected one of: 'cma', 'unimodal', 'concat'."
        )

    return module, d_out
