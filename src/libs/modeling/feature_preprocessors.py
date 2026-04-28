"""
Feature preprocessors

The preprocessor sits between the dataset-loaded raw features (text, audio, video)
and the backbone encoder.

All preprocessors share the same interface:
    forward(text, audio, video) -> (B, T, d_out)
    .d_out : int   — output feature dimension passed to the backbone

Config key: ``preprocessor``
    preprocessor:
      type: "unimodal"  # UnimodalPreprocessor
      modality: "video" # one of "text" | "audio" | "video"

    preprocessor:
      type: "concat"    # ConcatPreprocessor
      modalities: ["audio", "video"]
      modality_dropout: 0.1

    preprocessor:
      type: "trifuse"   # TriFusePreprocessor
      d_model: 256
      n_heads: 4
      n_fusion_layers: 1
      dropout: 0.1
      modality_dropout: 0.1
"""
import torch
from torch import nn


def _apply_modality_dropout(text, audio, video, active_modalities, p):
    """
    Zero entire modalities independently with probability *p*.

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
    # Prevent all modalities from being dropped
    all_dropped = keep.sum(dim=1) == 0          # (B,)
    keep[all_dropped] = 1.0
    for i, m in enumerate(active_modalities):
        feat_map[m] = feat_map[m] * keep[:, i].view(B, 1, 1)
    return feat_map['text'], feat_map['audio'], feat_map['video']


# Preprocessor modules
class UnimodalPreprocessor(nn.Module):
    """
    Passes a single chosen modality to the backbone.

    Args:
        modality  : Which modality to use: ``"text"`` | ``"audio"`` | ``"video"``.
        text_dim  : Native dimension of text features.
        audio_dim : Native dimension of audio features.
        video_dim : Native dimension of video features.

    Output shape: (B, T, d_out)
    """

    def __init__(self, modality, text_dim, audio_dim, video_dim):
        super().__init__()
        assert modality in ('text', 'audio', 'video'), (
            f"modality must be 'text', 'audio', or 'video', got '{modality}'"
        )
        self.modality = modality

        dim_map = {'text': text_dim, 'audio': audio_dim, 'video': video_dim}
        in_dim  = dim_map[modality]
        self.d_out = in_dim

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
        return feat_map[self.modality]


class ConcatPreprocessor(nn.Module):
    """
    Concatenates the chosen modalities.

    Args:
        modalities       : List of modality names to concatenate, e.g. ``["audio", "video"]``.
        text_dim         : Native dimension of text features.
        audio_dim        : Native dimension of audio features.
        video_dim        : Native dimension of video features.
        modality_dropout : Probability of zeroing an entire modality for a sample
                           during training.  Default 0.0 (disabled).

    Output shape: (B, T, d_out)
    """

    def __init__(self, modalities, text_dim, audio_dim, video_dim,
                 modality_dropout=0.0):
        super().__init__()
        assert len(modalities) >= 1, "modalities must not be empty"
        for m in modalities:
            assert m in ('text', 'audio', 'video'), (
                f"Unknown modality '{m}'. Expected one of: 'text', 'audio', 'video'."
            )
        self.modalities       = list(modalities)
        self.modality_dropout = modality_dropout

        dim_map = {'text': text_dim, 'audio': audio_dim, 'video': video_dim}
        in_dim  = sum(dim_map[m] for m in modalities)
        self.d_out = in_dim

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
        return torch.cat(feats, dim=-1)


# Factory
def build_preprocessor(cfg, text_dim, audio_dim, video_dim):
    """
    Instantiate the correct preprocessor from the model config dict.

    Parameters
    ----------
    cfg       : Full model config dict (as loaded from YAML).
    text_dim  : Native dimension of text features.
    audio_dim : Native dimension of audio features.
    video_dim : Native dimension of video features.

    Returns
    -------
    module : nn.Module
        One of UnimodalPreprocessor, ConcatPreprocessor, TriFusePreprocessor.
    d_out : int
        Output feature dimension (passed as ``n_in`` to the backbone).
    """
    prep_cfg = cfg.get('preprocessor', None)
    if prep_cfg is None:
        raise ValueError(
            "Config is missing a 'preprocessor' key. "
            "Add a 'preprocessor' block with a 'type' field "
            "('unimodal', 'concat', or 'trifuse')."
        )

    ptype = prep_cfg['type']

    if ptype == 'unimodal':
        module = UnimodalPreprocessor(
            modality=prep_cfg['modality'],
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
        )

    elif ptype == 'concat':
        module = ConcatPreprocessor(
            modalities=prep_cfg['modalities'],
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            modality_dropout=prep_cfg.get('modality_dropout', 0.0),
        )

    elif ptype == 'trifuse':
        from .trifuse import TriFusePreprocessor
        module = TriFusePreprocessor(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            d_model=prep_cfg['d_model'],
            n_heads=prep_cfg.get('n_heads', 4),
            n_fusion_layers=prep_cfg.get('n_fusion_layers', 1),
            dropout=prep_cfg.get('dropout', 0.1),
            modality_dropout=prep_cfg.get('modality_dropout', 0.0),
        )

    else:
        raise ValueError(
            f"Unknown preprocessor type '{ptype}'. "
            "Expected one of: 'unimodal', 'concat', 'trifuse'."
        )

    return module, module.d_out
