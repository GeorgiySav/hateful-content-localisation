# Implementation Prompt: Adapted ActionFormer for Hateful Content Temporal Localization

## Goal

Implement a hybrid model that combines ActionFormer's temporal action localization architecture with MM-HSD's cross-modal attention (CMA) fusion strategy. The model takes pre-extracted per-timestep multi-modal features (video, audio, text) and outputs temporally localized hateful segments — each with a start time, end time, and confidence score.

Use the **zero-out strategy** for missing modalities: when a modality is absent at a given timestep (e.g. silent frames with no transcript), the CMA contribution for that timestep is zeroed out.

---

## Directory Structure

You will be working in a directory with the following layout:

```
project/
├── actionformer/                    # Cloned from https://github.com/happyharrycn/actionformer_release
│   ├── libs/
│   │   ├── modeling/                # blocks.py, backbones.py, meta_archs.py, etc.
│   │   ├── datasets/
│   │   └── utils/
│   ├── configs/
│   ├── train.py
│   ├── eval.py
│   └── ...
├── mm-hsd/                          # Cloned from https://github.com/idiap/mm-hsd
│   ├── src/mm_hsd/
│   │   ├── configs/
│   │   ├── scripts/
│   │   └── ...
│   └── ...
└── data/
    ├── extract_features/
    │   └── extract_features.py      # My existing feature extraction script (DO NOT MODIFY)
    └── hatemm/
        ├── dataset/                 # Pre-extracted .npz feature files live here
        │   ├── hate_video_1.npz
        │   ├── non_hate_video_1.npz
        │   └── ...
        └── scripts/                 # YOUR OUTPUT — the adapted model goes here
            └── (empty, to be created by you)
```

**Read and understand `actionformer/`, `mm-hsd/`, and `data/extract_features/extract_features.py` before writing any code.** All new code goes under `data/hatemm/scripts/`. Do not modify anything outside that directory.

---

## Pre-Extracted Features (ALREADY HANDLED — do not reimplement)

Feature extraction is handled by `data/extract_features/extract_features.py`, which I have already run. You do not need to implement any feature extraction code. Just build the model and dataset classes that consume the output.

The script extracts **three modalities** at 1 FPS, aligned to a common timeline of T timesteps (T = number of sampled video frames). The existing script saves individual `.pt` files; I have converted these into per-video `.npz` bundles in `data/hatemm/dataset/`.

### .npz file format

Each video has one `.npz` file containing:

```python
data = np.load("data/hatemm/dataset/<video_id>.npz")
data["video"]       # (T, 768)   — float32, CLIP ViT-L/14 frame-level features
data["audio"]       # (T, 1024)  — float32, Wav2Vec2 Large, linearly interpolated to T
data["text"]        # (T, 768)   — float32, Whisper ASR → sentence-wise HateBERT CLS
data["fps"]         # scalar     — feature extraction FPS (1.0)
data["duration"]    # scalar     — video duration in seconds
```

Important properties:
- T varies per video (depends on video duration at 1 FPS).
- `audio` is a zero array `(T, 1024)` when the video has no audio track.
- `text` has zero vectors at timesteps where no speech was detected by Whisper.
- `video` is always present (never zero) for all T frames.

### Feature extractor models

| Modality | Model | Dimensions |
|----------|-------|-----------|
| Video | CLIP ViT-L/14 (per-frame [CLS]) | 768 |
| Audio | Wav2Vec2 Large (`facebook/wav2vec2-large-960h`), linearly interpolated to T | 1024 |
| Text | Whisper ASR → sentence-wise HateBERT (`GroNLP/hateBERT`) [CLS], spread over timestamp spans | 768 |

**There is no OCR / on-screen text modality.** The CMA design must work with three modalities only.

---

## Architecture Specification

### Overview

The pipeline has three stages:
1. **Cross-modal fusion** producing one fused feature vector per timestep (adapted from MM-HSD's CMA)
2. **Projection + multiscale transformer encoder** building a temporal feature pyramid (from ActionFormer)
3. **Classification + regression decoder** producing per-timestep hate scores and segment boundaries (from ActionFormer)

### Stage 1: Cross-Modal Attention Fusion (Trainable)

Implement as a PyTorch module `CrossModalFusion`.

**CMA design (adapted from MM-HSD for 3 modalities, no OCR):**

In MM-HSD, on-screen text was the query because it was the weakest standalone modality but carried unique signals when grounded against other modalities. In our 3-modality setup, text (transcript) is the natural query: it carries the most direct semantic signal for hate speech but benefits from attending to what is visually shown and acoustically conveyed — the same word can be hateful or benign depending on visual/audio context.

At each timestep `t`:

1. **Project each modality to a common dimension `d_cma`** (default 256):
   ```
   t_proj = linear_text(text[t])     # (d_cma,)
   a_proj = linear_audio(audio[t])   # (d_cma,)
   v_proj = linear_video(video[t])   # (d_cma,)
   ```

2. **Cross-modal attention**:
   ```
   Q = t_proj  → shape (1, d_cma)
   K = V = stack([a_proj, v_proj])  → shape (2, d_cma)
   cma_out = MultiHeadAttention(Q, K, V)  → shape (d_cma,)
   ```

3. **Compute text presence mask**: Determine whether text features are present at this timestep. Since silent frames are zero vectors from the extraction script, compute:
   ```
   text_mask[t] = 1.0 if text[t].abs().sum() > 0 else 0.0
   ```
   This mask is computed on-the-fly from the loaded features, no separate mask file needed.

4. **Zero-out**: Multiply the CMA output by the text presence mask:
   ```
   cma_out[t] = cma_out[t] * text_mask[t]
   ```
   When there is no transcript at a timestep, the CMA output is zeroed — the model relies solely on the individual modality features for those timesteps.

5. **Concatenate** all features along the feature dimension:
   ```
   fused[t] = concat([text[t], audio[t], video[t], cma_out[t]])
   ```
   Total fused dimension = 768 + 1024 + 768 + d_cma = 2560 + d_cma (default: 2816).

6. Do this for all T timesteps. The full operation can be batched efficiently by reshaping timesteps into the batch dimension — do NOT use an explicit Python loop over timesteps.

**Output**: `fused` tensor of shape `(B, T, fused_dim)`.

### Stage 2: ActionFormer Temporal Localization Backbone (Trainable)

Port ActionFormer's architecture with these components:

#### Projection
- 2x Conv1D layers (from ActionFormer), mapping from `fused_dim` → `d_model` (default 512).
- ReLU activation between layers.
- No positional encoding (ActionFormer found it hurts performance).

#### Multiscale Transformer Encoder
- Follow ActionFormer's design exactly.
- Default: 6 transformer blocks total. First 1 without downsampling, remaining 5 with 2x downsampling each.
- Local self-attention with configurable window size (default 19).
- Each block: LayerNorm → Local Multi-Head Self-Attention (with learnable per-channel scaling α) → residual → LayerNorm → MLP with GELU (with learnable scaling ᾱ) → residual → optional 2x downsample via strided depthwise Conv1D.
- Output: feature pyramid `Z = {Z^1, Z^2, ..., Z^L}` at resolutions T, T/2, T/4, ..., T/32.

### Stage 3: Decoder Heads (Trainable)

#### Classification head
- 3x Conv1D layers (kernel=3), with LayerNorm on the first 2 layers, ReLU activation.
- Shared across all pyramid levels.
- Output dimension = 1 (binary hate/no-hate, single sigmoid output).
- For a future multi-class variant (racial, religious, sexist, etc.): output dimension = C with independent sigmoids. Make this configurable via `num_classes` in config.

#### Regression head
- Same architecture as classification head (3x Conv1D, kernel=3, LayerNorm, ReLU).
- Output dimension = 2 (d_start, d_end distances to segment boundaries).
- ReLU at the end to ensure non-negative distances.
- Regression range per pyramid level: level 1 = [0, 4), level 2 = [4, 8), level 3 = [8, 16), level 4 = [16, 32), level 5 = [32, 64), level 6 = [64, +∞). These are in units of feature strides (seconds at 1 FPS).

#### Loss function
Directly from ActionFormer:
```
L = sum_t (L_cls + λ_reg * 1_{positive} * L_reg) / T_+
```
- `L_cls`: Focal loss (α=0.25, γ=2.0). Critical for handling the massive foreground/background imbalance — most timesteps are non-hateful.
- `L_reg`: DIoU loss for boundary regression, only applied to positive (foreground) timesteps.
- `λ_reg = 1.0` default.
- **Center sampling** during training: only timesteps within `α * stride` of a hateful segment's center are labeled positive. α = 1.5.
- Loss is applied across all pyramid levels and averaged over positive samples.

#### Inference
- Feed the full unpadded sequence (no sliding window needed since there are no positional encodings).
- Each point on each pyramid level decodes a candidate: `start = t - d_s`, `end = t + d_e`, `score = sigmoid(cls_output)`.
- Apply Soft-NMS (σ=0.4) to remove overlapping detections.
- Output: list of `(start_time, end_time, confidence)` tuples.

---

## Code Structure

Create the following under `data/hatemm/scripts/`:

```
data/hatemm/scripts/
├── README.md                        # Setup instructions, usage, architecture overview
├── requirements.txt                 # torch, numpy, pyyaml, pandas, tensorboard, scipy
├── configs/
│   └── default.yaml                 # Default config (binary hate/no-hate, HateMM)
├── libs/
│   ├── __init__.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── cross_modal_fusion.py    # CrossModalFusion module (CMA + zero-out + concat)
│   │   ├── backbone.py              # Multiscale transformer encoder producing feature pyramid
│   │   ├── blocks.py                # MaskedConv1D, LayerNorm, LocalMHA, TransformerBlock (from ActionFormer)
│   │   ├── heads.py                 # Classification and regression convolutional heads
│   │   └── meta_arch.py             # Full model: CrossModalFusion → Projection → Encoder → Heads
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── hatemm.py                # Dataset class loading .npz features + temporal annotations
│   └── utils/
│       ├── __init__.py
│       ├── train_utils.py           # Training loop, optimizer (Adam + warmup + cosine), scheduler, EMA
│       ├── eval_utils.py            # Temporal mAP evaluation at various tIoU thresholds
│       └── nms.py                   # Soft-NMS (port from ActionFormer)
├── train.py                         # Main training entry point
├── eval.py                          # Main evaluation entry point
└── tests/
    └── test_forward_pass.py         # Synthetic test verifying shapes and zero-out behaviour
```

---

## Config Format (YAML)

```yaml
# Dataset
dataset:
  name: "hatemm"
  # Path to directory containing per-video .npz files
  feat_dir: "../../dataset"               # Relative to scripts/, i.e. data/hatemm/dataset/
  annotation_file: "../../dataset/annotations.json"
  num_classes: 1                           # 1 for binary hate/no-hate
  feature_fps: 1.0                         # Features per second (from extraction script)
  max_seq_len: 2304                        # Max input length in timesteps (pad or crop)
  # Input feature dimensions (must match extraction script output)
  input_dims:
    text: 768                              # HateBERT CLS
    audio: 1024                            # Wav2Vec2 Large
    video: 768                             # CLIP ViT-L/14
  # Keys inside each .npz file
  npz_keys:
    text: "text"
    audio: "audio"
    video: "video"
    fps: "fps"
    duration: "duration"

# Cross-modal fusion
fusion:
  d_cma: 256                               # CMA internal / output dimension
  num_heads: 4                             # Multi-head attention heads in CMA
  dropout: 0.1
  query_modality: "text"                   # Which modality serves as query
  key_modalities: ["audio", "video"]       # Which modalities serve as key/value
  zero_out_missing_query: true             # Zero CMA output when query modality is absent

# ActionFormer backbone
backbone:
  d_model: 512                             # Internal feature dimension after projection
  n_proj_layers: 2                         # Number of Conv1D projection layers
  n_layers: 6                              # Number of transformer blocks
  n_heads: 4                               # Self-attention heads per transformer block
  window_size: 19                          # Local self-attention window size
  downsample_start: 1                      # Index of first layer with 2x downsampling (0-indexed)
  downsample_ratio: 2                      # Downsampling factor

# Decoder heads
heads:
  n_layers: 3                              # Number of Conv1D layers in each head
  kernel_size: 3
  use_layer_norm: true                     # LayerNorm on first (n_layers - 1) layers

# Regression ranges per pyramid level (auto-computed if omitted)
# Format: list of [min, max) ranges in units of feature strides
# Default for 6 levels: [[0,4], [4,8], [8,16], [16,32], [32,64], [64,inf]]

# Loss
loss:
  focal_alpha: 0.25
  focal_gamma: 2.0
  lambda_reg: 1.0
  center_sampling: true
  center_sampling_radius: 1.5

# Training
training:
  epochs: 50
  batch_size: 2
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  warmup_epochs: 5
  lr_scheduler: "cosine"
  use_ema: true
  ema_decay: 0.999
  clip_grad_norm: 1.0

# Inference
inference:
  score_threshold: 0.001
  nms_method: "soft_nms"
  nms_sigma: 0.4
  nms_threshold: 0.1
  max_detections: 200
```

---

## Annotation Format

Use a JSON format similar to ActivityNet annotations. This file lives at `data/hatemm/dataset/annotations.json`:

```json
{
  "database": {
    "hate_video_1": {
      "duration": 120.0,
      "subset": "train",
      "annotations": [
        { "segment": [12.5, 28.3], "label": "hate" },
        { "segment": [55.0, 62.1], "label": "hate" }
      ]
    },
    "non_hate_video_1": {
      "duration": 45.0,
      "subset": "val",
      "annotations": []
    }
  }
}
```

**Weak supervision fallback for HateMM:** The HateMM dataset has only video-level binary labels (hate / non-hate), not temporal segment annotations. The dataset class MUST support a fallback mode: if a video is labeled "hate" and has no segment-level annotations in the JSON, treat the **entire video** as a single hateful segment `[0.0, duration]`. Non-hate videos get an empty annotation list. Document this behaviour clearly in the dataset class docstring and in the README. This allows initial training even without frame-level ground truth.

---

## Key Implementation Details

### CrossModalFusion module — detailed pseudocode

```
class CrossModalFusion(nn.Module):
    def __init__(self, text_dim=768, audio_dim=1024, video_dim=768,
                 d_cma=256, num_heads=4, dropout=0.1):
        # Linear projections for each modality → d_cma
        self.proj_text  = nn.Linear(text_dim, d_cma)
        self.proj_audio = nn.Linear(audio_dim, d_cma)
        self.proj_video = nn.Linear(video_dim, d_cma)
        # Standard nn.MultiheadAttention (batch_first=True)
        self.cma = nn.MultiheadAttention(d_cma, num_heads, dropout=dropout,
                                         batch_first=True)

    def forward(self, text, audio, video):
        # text:  (B, T, 768)
        # audio: (B, T, 1024)
        # video: (B, T, 768)
        B, T, _ = text.shape

        # Project all modalities
        t_proj = self.proj_text(text)    # (B, T, d_cma)
        a_proj = self.proj_audio(audio)  # (B, T, d_cma)
        v_proj = self.proj_video(video)  # (B, T, d_cma)

        # Compute text presence mask from raw features
        # Text features are zero vectors at silent timesteps
        text_mask = (text.abs().sum(dim=-1) > 0).float()  # (B, T)

        # Reshape for per-timestep attention:
        # Treat each timestep as an independent attention problem
        # Q: (B*T, 1, d_cma) — one query token per timestep
        # K/V: (B*T, 2, d_cma) — audio and video as two key/value tokens
        Q = t_proj.reshape(B * T, 1, -1)
        K = torch.stack([a_proj, v_proj], dim=2)  # (B, T, 2, d_cma)
        K = K.reshape(B * T, 2, -1)
        V = K.clone()  # K and V are identical (same as MM-HSD)

        # Multi-head attention
        cma_out, _ = self.cma(Q, K, V)  # (B*T, 1, d_cma)
        cma_out = cma_out.reshape(B, T, -1)  # (B, T, d_cma)

        # Zero-out where text is absent
        cma_out = cma_out * text_mask.unsqueeze(-1)  # (B, T, d_cma)

        # Concatenate all features
        fused = torch.cat([text, audio, video, cma_out], dim=-1)
        # fused: (B, T, 768 + 1024 + 768 + d_cma)
        return fused
```

### Dataset class — loading .npz features

```
class HateMMDataset(Dataset):
    def __init__(self, feat_dir, annotation_file, max_seq_len,
                 feature_fps=1.0, npz_keys=None, is_training=True):
        """
        Loads pre-extracted .npz features for each video.

        Each .npz file in feat_dir contains:
            "video"    : (T, 768)   — CLIP ViT-L/14 frame features
            "audio"    : (T, 1024)  — Wav2Vec2 Large features
            "text"     : (T, 768)   — HateBERT sentence embeddings
            "fps"      : scalar     — feature extraction FPS
            "duration" : scalar     — video duration in seconds

        Weak supervision fallback:
            If a video is labeled 'hate' in the annotations but has no
            segment-level annotations, the entire video [0, duration] is
            treated as one hateful segment.
        """

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]

        # Load the .npz bundle
        npz_path = os.path.join(self.feat_dir, f"{video_id}.npz")
        data = np.load(npz_path)

        video_feat = torch.from_numpy(data[self.npz_keys["video"]])   # (T, 768)
        audio_feat = torch.from_numpy(data[self.npz_keys["audio"]])   # (T, 1024)
        text_feat  = torch.from_numpy(data[self.npz_keys["text"]])    # (T, 768)
        duration   = float(data[self.npz_keys["duration"]])

        T = video_feat.shape[0]  # video defines the canonical length

        # Verify / handle length mismatches (audio and text should match
        # video T from extraction, but add a safety check)
        audio_feat = self._match_length(audio_feat, T, dim=1024)
        text_feat  = self._match_length(text_feat, T, dim=768)

        # Pad or crop to max_seq_len, create padding mask
        # ...

        # Load temporal annotations for this video
        # ...

        return {
            "video_id": video_id,
            "video_feat": video_feat,   # (max_seq_len, 768)
            "audio_feat": audio_feat,   # (max_seq_len, 1024)
            "text_feat": text_feat,     # (max_seq_len, 768)
            "mask": mask,               # (max_seq_len,) — 1.0 for real, 0.0 for padded
            "segments": segments,       # list of (start, end) in seconds
            "labels": labels,           # list of class indices
        }
```

### Handling Variable-Length Sequences

Follow ActionFormer's approach exactly:
- During training, pad (or randomly crop) sequences to `max_seq_len`. Create a binary padding mask.
- Apply the mask to all attention operations and loss computation — padded positions must not attend or contribute to loss.
- At inference, feed the full unpadded sequence (possible because there are no positional encodings).

### Meta-architecture wiring (meta_arch.py)

The forward pass flows:
```
Load .npz → (text, audio, video) numpy arrays → torch tensors
    → CrossModalFusion → fused (B, T, fused_dim)
    → Conv1D Projection → (B, T, d_model=512)
    → Multiscale Transformer Encoder → feature pyramid [Z^1, ..., Z^L]
    → Classification Head → per-timestep hate scores on each level
    → Regression Head → per-timestep (d_start, d_end) on each level
    → Loss (training) or Decode + Soft-NMS (inference)
```

---

## Differences from vanilla ActionFormer (document in comments)

1. **Input**: Three separate modality arrays from a single `.npz` file, fused via CMA, instead of a single pre-fused feature (e.g. I3D).
2. **CrossModalFusion module**: New module inserted before ActionFormer's projection layer. Implements MM-HSD-style CMA with text as query, audio+video as key/value, plus zero-out for missing text.
3. **Classification**: Binary hate/no-hate (1 sigmoid) instead of 20+ action categories (C sigmoids). Configurable via `num_classes`.
4. **Feature stride**: 1 second (1 FPS) instead of ActionFormer's ~0.13s (I3D at stride 4 on 30fps). Regression ranges may need different scaling — make configurable.
5. **Weak supervision fallback**: Dataset handles video-level labels as full-video segments when no temporal annotations are available.

## Differences from vanilla MM-HSD (document in comments)

1. **Temporal**: Per-timestep features and per-timestep classification + boundary regression, instead of one label per video.
2. **Three modalities, no OCR**: Text (HateBERT), audio (Wav2Vec2), video (CLIP ViT-L/14). No PaddleOCR on-screen text extraction.
3. **No per-modality encoders**: ActionFormer's transformer encoder replaces MM-HSD's LSTM (video) and FC (text, audio) encoders. CMA operates on raw projected embeddings.
4. **Output**: Temporal segments with boundaries, not a binary video-level label.
5. **Zero-out strategy**: Uses zero-detection on raw features rather than a precomputed mask file.

---

## Implementation priority

Build in this order:
1. `libs/modeling/blocks.py` — Port ActionFormer's MaskedConv1D, LayerNorm, local self-attention, TransformerBlock. Read `actionformer/libs/modeling/blocks.py` carefully.
2. `libs/modeling/cross_modal_fusion.py` — The CMA + zero-out + concat module as specified above.
3. `libs/modeling/backbone.py` — Multiscale transformer encoder producing the feature pyramid. Port from `actionformer/libs/modeling/backbones.py`.
4. `libs/modeling/heads.py` — Classification and regression Conv1D heads. Port from `actionformer/libs/modeling/meta_archs.py` (the heads are defined there).
5. `libs/modeling/meta_arch.py` — Full model wiring: fusion → projection → encoder → heads → loss/decode.
6. `libs/datasets/hatemm.py` — Dataset class loading `.npz` feature bundles.
7. `libs/utils/nms.py` — Soft-NMS. Port from `actionformer/libs/utils/`.
8. `libs/utils/train_utils.py` — Training loop, Adam optimizer with warmup + cosine decay, optional EMA. Port patterns from `actionformer/libs/utils/`.
9. `libs/utils/eval_utils.py` — Temporal mAP at tIoU thresholds.
10. `train.py` and `eval.py` — Entry point scripts.
11. `tests/test_forward_pass.py` — Synthetic tests.
12. `configs/default.yaml` and `README.md`.

---

## Testing requirements

After implementation, `tests/test_forward_pass.py` must:

1. **Shape test**: Generate random features of correct shapes (B=2, T=120, dims as above), run a full forward pass, verify output shapes at each pyramid level.
2. **Zero-out test**: Set `text_feat` to all zeros for all timesteps. Verify the CMA output tensor is exactly zero everywhere. Verify the model still runs and produces valid outputs from the audio+video features alone (via the direct concatenation path).
3. **Pyramid test**: Verify the feature pyramid has the expected number of levels (6) with temporal resolutions T, T/2, T/4, T/8, T/16, T/32.
4. **Mask test**: Create a sequence where the last 40 timesteps are padding (mask=0). Verify the loss is zero for those positions.
5. **Gradient test**: Run a forward + backward pass and verify that all parameters in CrossModalFusion, the backbone, and the heads receive non-zero gradients.
6. **npz round-trip test**: Create a synthetic `.npz` file with known shapes, load it through the dataset class, and verify the tensors arrive at the model with correct shapes and dtypes.

---

## References

- ActionFormer paper: https://arxiv.org/abs/2202.07925
- ActionFormer code: https://github.com/happyharrycn/actionformer_release
- MM-HSD paper: https://arxiv.org/abs/2508.20546
- MM-HSD code: https://github.com/idiap/mm-hsd
- HateMM dataset: Das et al., "HateMM: A Multi-Modal Dataset for Hate Video Classification", ICWSM 2023
- Feature extraction: `data/extract_features/extract_features.py` (CLIP ViT-L/14 + Wav2Vec2 Large + Whisper/HateBERT)