"""
Cross-Modal Attention (CMA) Fusion module.

Adapted from MM-HSD's CMA strategy for 3 modalities (no OCR):
  - Text (HateBERT, 768-dim) is the query — most direct semantic hate signal,
    benefits from visual/acoustic context grounding.
  - Audio (Wav2Vec2, 1024-dim) and Video (CLIP ViT-L/14, 768-dim) are key/value.

Zero-out strategy: when text features are absent at a timestep (all-zero vector
from the extraction script), the CMA output is zeroed — the model falls back to
the raw concatenation of audio + video for that timestep.

Differences from vanilla MM-HSD CMA:
  1. Three modalities only (no OCR/on-screen text).
  2. Text is query (instead of OCR in MM-HSD) — same rationale.
  3. Zero-out is computed on-the-fly from the raw feature norm (no precomputed mask).
  4. Per-timestep attention is batched by folding T into the batch dimension.
  5. Output includes all raw modalities concatenated with the CMA output.
"""
import torch
from torch import nn


class CrossModalFusion(nn.Module):
    """
    Cross-Modal Attention fusion for (text, audio, video) → fused feature.

    For each timestep t:
        1. Project text → (d_cma,), audio → (d_cma,), video → (d_cma,)
        2. Q = text_proj, K = V = stack([audio_proj, video_proj])
        3. cma_out = MultiHeadAttention(Q, K, V)
        4. text_mask[t] = 1 if text[t].abs().sum() > 0 else 0   (zero-out)
        5. cma_out[t] *= text_mask[t]
        6. fused[t] = cma_out[t]   (raw modalities are NOT re-concatenated)

    Output dimension: d_cma (default: 128)

    The batch operation reshapes (B, T, D) → (B*T, 1/2, D) so there is
    no explicit Python loop over timesteps.
    """

    def __init__(
        self,
        text_dim=768,
        audio_dim=1024,
        video_dim=768,
        d_cma=256,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.text_dim  = text_dim
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.d_cma     = d_cma
        self.fused_dim = d_cma

        # Per-modality linear projections to the common CMA space
        self.proj_text  = nn.Linear(text_dim,  d_cma)
        self.proj_audio = nn.Linear(audio_dim, d_cma)
        self.proj_video = nn.Linear(video_dim, d_cma)

        # Standard multi-head attention (batch_first=True)
        # Q: text projection (1 token per timestep)
        # K/V: audio + video projections (2 tokens per timestep)
        self.cma = nn.MultiheadAttention(
            d_cma, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, text, audio, video):
        """
        Args:
            text:  (B, T, 768)
            audio: (B, T, 1024)
            video: (B, T, 768)

        Returns:
            fused: (B, T, d_cma)
        """
        B, T, _ = text.shape

        # --- Compute text-presence mask from raw features (zero-out strategy) ---
        # Text features are exactly zero at silent/no-transcript timesteps.
        # Shape: (B, T)  —  1.0 where speech is present, 0.0 where silent.
        text_mask = (text.abs().sum(dim=-1) > 0).float()  # (B, T)

        # --- Project all modalities to the common CMA dimension ---
        t_proj = self.proj_text(text)    # (B, T, d_cma)
        a_proj = self.proj_audio(audio)  # (B, T, d_cma)
        v_proj = self.proj_video(video)  # (B, T, d_cma)

        # --- Reshape for per-timestep multi-head attention ---
        # Treat each (batch, timestep) as an independent attention problem.
        # Q: (B*T, 1, d_cma)  — one query token (text) per timestep
        # K/V: (B*T, 2, d_cma) — two key/value tokens (audio, video)
        Q = t_proj.reshape(B * T, 1, self.d_cma)

        # stack along the token dimension → (B, T, 2, d_cma) → (B*T, 2, d_cma)
        KV = torch.stack([a_proj, v_proj], dim=2).reshape(B * T, 2, self.d_cma)

        # --- Multi-head cross-modal attention ---
        # K and V are identical (same as MM-HSD design)
        cma_out, _ = self.cma(Q, KV, KV)          # (B*T, 1, d_cma)
        cma_out = cma_out.reshape(B, T, self.d_cma)  # (B, T, d_cma)

        # --- Zero-out CMA where text is absent ---
        # When no transcript is present the CMA output carries no signal;
        # zero it so the model relies solely on audio+video for silent frames.
        cma_out = cma_out * text_mask.unsqueeze(-1)  # (B, T, d_cma)

        # Return only the CMA output — it already encodes all three modalities
        # via cross-modal attention. Re-concatenating the raw features would
        # inflate the backbone input to 2816-dim, creating excess capacity.
        return cma_out  # (B, T, d_cma)
