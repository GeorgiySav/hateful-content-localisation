# Config File Reference

Each experiment is a single YAML file with eight top-level sections.
The sections below list every supported key, its type, valid values, and what it controls.

---

## 1. `dataset`

Paths to features and dataset-level settings.

| Key | Type | Description |
|-----|------|-------------|
| `name` | str | Dataset name. Only `"hatemm"` is supported. |
| `video_feat_dir` | str | Path to per-video `.pt` files for video features. |
| `audio_feat_dir` | str | Path to per-video `.pt` files for audio features. |
| `text_feat_dir` | str | Path to per-video `.pt` files for text features. |
| `annotation_file` | str | Path to the `annotations.json` file. |
| `num_classes` | int | `1` for binary hate/no-hate classification. |
| `feature_fps` | float | Features extracted per second. Default `1.0`. |
| `max_seq_len` | int | Maximum sequence length in frames. Sequences are padded or truncated to this. |
| `input_dims.text` | int | Native dimension of text features (e.g. `768` for HateBERT CLS). |
| `input_dims.audio` | int | Native dimension of audio features (e.g. `1024` for Wav2Vec2 Large). |
| `input_dims.video` | int | Native dimension of video features (e.g. `768` for CLIP ViT-L/14). |

---

## 2. `preprocessor`

Fuses multi-modal features before the backbone. Five types are available.

### `type: "cma"` — Guided Cross-Modal Attention (default)

Per-timestep cross-modal attention: one modality is the query, the others are key/value.
Use this for the full multi-modal model.

```yaml
preprocessor:
  type: "cma"
  d_out: 128           # output dim fed to the backbone (must equal backbone.d_model)
  num_heads: 2         # attention heads; d_out must be divisible by num_heads
  dropout: 0.1
  query_modality: "text"              # "text" | "audio" | "video"
  kv_modalities: ["audio", "video"]  # the modalities used as key/value
  zero_out_missing_query: true        # zero output where query feature is all-zeros
```

### `type: "unimodal"` — Single modality ablation

Passes one modality through a linear projection, discarding the others.

```yaml
preprocessor:
  type: "unimodal"
  modality: "video"   # "text" | "audio" | "video"
  d_out: 256
```

### `type: "concat"` — Concatenation baseline

Concatenates the chosen modalities and projects to `d_out` with a single linear layer.
No attention, no cross-modal interaction.

```yaml
preprocessor:
  type: "concat"
  modalities: ["audio", "video"]   # any non-empty subset of the three modalities
  d_out: 256                        # input dim = sum of chosen modalities' native dims
```

### `type: "multihateloc"` — MA-TE + DCM-Fusion (MultiHateLoc paper)

Full tri-modal pipeline from Sun et al., WWW 2026.  Three stages:

1. **MA-TE** — per-modality pre-norm Transformer block (self-attention + FFN) in a shared internal dimension `d_inner`.
2. **DCM-Fusion** — Dynamic Modality Selection (sigmoid scalar gate per timestep) followed by Cross-Modal Attention over the concatenated weighted features.
3. **Output** — linear projection from the four branches (F'_v, F'_a, F'_l, F_fused) to `d_out`.

```yaml
preprocessor:
  type: "multihateloc"
  d_out: 256        # output dim fed to the backbone
  d_inner: 256      # internal common modality dim D; must be divisible by n_heads
  n_heads: 4        # attention heads for both MA-TE and CMA
  dropout: 0.1
```

### `type: "trifuse"` — TriFuse Trimodal Bottleneck Fusion

Four-stage architecture with learnable bottleneck tokens and mask-aware sparse transcript handling.

1. **Stage 1** — per-modality linear projection to `d_out` + learnable modality embeddings + sinusoidal positional encoding. A presence mask `mask_x = (‖text‖₂ > 1e-6)` is derived from the raw text input and applied to zero-out absent text timesteps after projection.
2. **Stage 2** — separate `nn.TransformerEncoder` per modality (`n_unimodal_layers` layers). Text uses a key-padding mask so absent timesteps neither attend to others nor are attended to.
3. **Stage 3** — `n_fusion_layers` bottleneck fusion layers. Each layer: (a) bottleneck tokens cross-attend to all modalities (text cross-attention is mask-aware), (b) bottleneck self-attention + FFN, (c) each modality cross-attends back to the bottleneck through a learned sigmoid gate; text is re-zeroed for absent positions.
4. **Stage 4** — a 2-layer MLP produces per-timestep weights for the three enriched streams; the text weight is forced to 0 where text is absent, so the fusion degrades gracefully to bimodal video+audio.

```yaml
preprocessor:
  type: "trifuse"
  d_out: 256            # d_model inside TriFuse and n_in to the backbone
  n_heads: 4            # attention heads; d_out must be divisible by n_heads
  n_bottleneck: 4       # number of learnable bottleneck tokens
  n_unimodal_layers: 2  # per-modality self-attention layers (Stage 2)
  n_fusion_layers: 4    # bottleneck cross-modal fusion layers (Stage 3)
  dropout: 0.1
```

> **Note:** `preprocessor.d_out` is `n_in` to the backbone. The backbone's projection convs (`n_proj_layers` Conv1Ds) project from `n_in` → `d_model`, so the two values can differ freely. The only exception is `n_proj_layers: 0` — with no projection layers, features flow directly into the stem blocks expecting `d_model`, so `d_out` must equal `d_model` in that case.

---

## 3. `backbone`

The temporal encoder. Three types are available.

### `type: "transformer"` — ActionFormer (local self-attention)

```yaml
backbone:
  type: "transformer"
  d_model: 128          # internal feature dim; must equal preprocessor.d_out
  n_proj_layers: 1      # Conv1D projection layers before the transformer pyramid
  n_layers: 3           # total pyramid levels (stem + branch blocks)
  n_heads: 2            # self-attention heads; d_model must be divisible by n_heads
  window_size: 17       # local attention window (odd number); -1 for global attention
  downsample_start: 1   # 0-based index of the first downsampling block
  downsample_ratio: 2   # stride multiplier per branch block
```

### `type: "temporalmaxer"` — TemporalMaxer (parameter-free max pooling)

Replaces self-attention with local max pooling. Faster, fewer parameters.
`n_heads` and `window_size` are ignored but must still be present for API compatibility.

```yaml
backbone:
  type: "temporalmaxer"
  d_model: 128
  n_proj_layers: 1
  n_layers: 3
  n_heads: 2            # ignored
  window_size: 17       # ignored
  downsample_start: 1
  downsample_ratio: 2
  pool_kernel_size: 3   # MaxPool1D kernel size (TemporalMaxer paper uses 3)
```

### `type: "sgp"` — TriDet / SGP (depthwise conv, no attention)

Scalable-Granularity Perception: dual-branch depthwise conv replacing self-attention.
`n_heads` and `window_size` are ignored.

```yaml
backbone:
  type: "sgp"
  d_model: 128
  n_proj_layers: 1
  n_layers: 3
  n_heads: 2              # ignored
  window_size: 17         # ignored
  downsample_start: 1
  downsample_ratio: 2
  sgp_kernel_size: 3      # instant-level depthwise conv kernel
  sgp_mlp_dim: 512        # FFN hidden dim (typically 4× d_model)
  sgp_k: 1.5              # window-level kernel scale factor
  sgp_init_conv_vars: 1   # Gaussian init std for SGP depthwise weights (0 = zeros)
  sgp_downsample_type: "max"   # "max" | "avg" pooling for branch downsampling
```

### Pyramid levels

The number of pyramid levels is `n_layers`. The stem has `downsample_start` blocks at stride 1; the remaining `n_layers - downsample_start` branch blocks each multiply the stride by `downsample_ratio`.

Examples:
- `n_layers=3, downsample_start=1` → strides `[1, 2, 4]` → **3 levels**
- `n_layers=6, downsample_start=1` → strides `[1, 2, 4, 8, 16, 32]` → **6 levels**

> **Constraint:** The number of entries in `regression_ranges` must equal `n_layers`.

---

## 4. `neck`

Feature aggregation across pyramid levels.

### `type: "identity"` (default)

Applies only layer normalisation per level. No cross-scale fusion.

```yaml
neck:
  type: "identity"
```

### `type: "fpn"` — Feature Pyramid Network

Adds lateral 1×1 convs and a top-down path so each level receives context from coarser scales.

```yaml
neck:
  type: "fpn"
  with_ln: true   # apply layer norm after each lateral conv
```

---

## 5. `heads`

Decode backbone features into segment predictions.

### `type: "standard"` — classification + direct offset regression

```yaml
heads:
  type: "standard"
  n_layers: 1          # number of conv layers per head (including output layer)
  kernel_size: 3
  use_layer_norm: false  # set true when n_layers > 1
```

### `type: "trident"` — Trident boundary distribution head (TriDet)

Predicts relative boundary probability distributions instead of direct offsets.
Better for ambiguous boundaries.

```yaml
heads:
  type: "trident"
  n_layers: 3
  kernel_size: 3
  use_layer_norm: true
  boundary_kernel_size: 3  # kernel for the start/end boundary sub-heads
  num_bins: 16             # number of distribution bins (excluding the zero-offset bin)
  iou_weight_power: 2      # exponent for IoU-weighted classification loss
```

---

## 6. `regression_ranges`

One entry per pyramid level. Each entry `[min, max]` defines the half-duration range
(in seconds at `feature_fps`) assigned to that level.

Label assignment compares `max(t - start, end - t)` (half the segment duration) against
these ranges, so a range of `[0, 16]` covers segments up to 32 seconds long.

The last entry's upper bound is effectively infinity — use a large number like `100000`.

```yaml
regression_ranges:
  - [0,   16]   # stride-1 level: short segments (D < 32 s)
  - [16,  64]   # stride-2 level: medium segments (D < 128 s)
  - [64,  100000]  # stride-4 level: long segments
```

> **Constraint:** Must have exactly as many entries as `backbone.n_layers`.

---

## 7. `loss`

| Key | Type | Description |
|-----|------|-------------|
| `focal_alpha` | float | Focal loss alpha (foreground weight).  |
| `focal_gamma` | float | Focal loss focusing parameter. Typically `2.0`–`3.0`. |
| `lambda_reg` | float | Weight of the regression loss relative to the classification loss. |
| `center_sampling` | bool | If `true`, only anchors within `center_sampling_radius` of a segment centre are treated as positives. |
| `center_sampling_radius` | float | Radius in seconds for centre sampling. `2.0` gives ~33% more positives than `1.5`. |

---

## 8. `training`

| Key | Type | Description |
|-----|------|-------------|
| `batch_size` | int | Training batch size. |
| `epochs` | int | Total training epochs. |
| `learning_rate` | float | Peak learning rate (after warmup). |
| `weight_decay` | float | AdamW weight decay. |
| `warmup_epochs` | int | Linear warmup from 0 to `learning_rate` over this many epochs. |
| `lr_scheduler` | str | `"cosine"` or `"multistep"`. |
| `use_ema` | bool | Exponential moving average of model weights for evaluation. |
| `ema_decay` | float | EMA decay factor (e.g. `0.999`). |
| `clip_grad_norm` | float | Gradient clipping norm. |
| `patience` | int | Early stopping patience in epochs (optional; omit to disable). |
| `cls_prior_prob` | float | Prior probability for the classifier bias initialisation. Set to approximate positive anchor rate. |
| `dropout` | float | Dropout probability in backbone and heads. |
| `droppath` | float | Stochastic depth (drop-path) rate. |
| `label_smoothing` | float | Label smoothing for the classification loss (`0.0` to disable). |
| `weighted_sampling` | bool | Oversample positive videos during training (optional). |

---

## 9. `augmentation`

Applied during training only.

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Master switch. Set `false` to disable all augmentation. |
| `feature_noise_std` | float | Standard deviation of Gaussian noise added to all modality features. |
| `temporal_mask_prob` | float | Probability of applying temporal masking to a sample. |
| `temporal_mask_num` | int | Number of masking windows when masking is triggered. |
| `temporal_mask_max_len` | int | Maximum length in frames of each masking window. Keep ≤ p10 segment duration to avoid masking entire short segments. |
| `segment_jitter_sec` | float | Max ±jitter in seconds applied to segment boundaries. |

---

## 10. `inference`

| Key | Type | Description |
|-----|------|-------------|
| `score_threshold` | float | Minimum confidence score to keep a candidate segment. |
| `nms_method` | str | `"soft_nms"` (recommended) or `"nms"`. |
| `nms_sigma` | float | Gaussian sigma for Soft-NMS score decay. |
| `nms_threshold` | float | IoU threshold for NMS suppression. |
| `max_detections` | int | Maximum number of output segments per video. |

---

## Key constraints summary

| Constraint | Rule |
|-----------|------|
| Preprocessor → backbone dimension | Can differ; projection convs handle the mapping. Exception: if `n_proj_layers: 0` then `preprocessor.d_out` must equal `backbone.d_model`. |
| Regression ranges | Number of entries == `backbone.n_layers` |
| Attention heads | `backbone.d_model` divisible by `backbone.n_heads` (transformer only) |
| CMA/TriFuse heads | `preprocessor.d_out` divisible by `preprocessor.num_heads` (or `n_heads` for trifuse) |
| Window size | Must be an odd integer (transformer backbone) |

---

## Minimal experiment template

```yaml
dataset:
  name: "hatemm"
  video_feat_dir: "../../data/hatemm/video_features"
  audio_feat_dir: "../../data/hatemm/audio_features"
  text_feat_dir:  "../../data/hatemm/text_features"
  annotation_file: "../../data/hatemm/dataset/annotations.json"
  num_classes: 1
  feature_fps: 1.0
  max_seq_len: 512
  input_dims:
    text: 768
    audio: 1024
    video: 768

preprocessor:
  type: "cma"
  d_out: 128           # must equal backbone.d_model
  num_heads: 2
  dropout: 0.1
  query_modality: "text"
  kv_modalities: ["audio", "video"]
  zero_out_missing_query: true

backbone:
  type: "transformer"  # "transformer" | "temporalmaxer" | "sgp"
  d_model: 128         # must equal preprocessor.d_out
  n_proj_layers: 1
  n_layers: 3          # determines number of pyramid levels
  n_heads: 2
  window_size: 17
  downsample_start: 1
  downsample_ratio: 2

neck:
  type: "identity"     # "identity" | "fpn"

heads:
  type: "standard"     # "standard" | "trident"
  n_layers: 1
  kernel_size: 3
  use_layer_norm: false

regression_ranges:     # one entry per backbone level (n_layers = 3 → 3 entries)
  - [0,   16]
  - [16,  64]
  - [64,  100000]

loss:
  focal_alpha: 0.40
  focal_gamma: 3.0
  lambda_reg: 1.0
  center_sampling: true
  center_sampling_radius: 2.0

training:
  batch_size: 8
  epochs: 50
  learning_rate: 1.0e-4
  weight_decay: 5.0e-4
  warmup_epochs: 5
  lr_scheduler: "cosine"
  use_ema: true
  ema_decay: 0.999
  clip_grad_norm: 1.0
  cls_prior_prob: 0.01
  dropout: 0.5
  droppath: 0.3
  label_smoothing: 0.1

augmentation:
  enabled: true
  feature_noise_std: 0.01
  temporal_mask_prob: 0.5
  temporal_mask_num: 2
  temporal_mask_max_len: 3
  segment_jitter_sec: 0.5

inference:
  score_threshold: 0.001
  nms_method: "soft_nms"
  nms_sigma: 0.4
  nms_threshold: 0.1
  max_detections: 200
```
