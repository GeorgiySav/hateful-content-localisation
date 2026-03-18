# HateMM Temporal Hateful Content Localization

Adapted **ActionFormer** (temporal action localization) combined with
**MM-HSD**'s cross-modal attention (CMA) fusion, trained on the
[HateMM dataset](https://github.com/hate-alert/HateMM).

---

## Architecture Overview

```
.npz features
 ├── text  (T, 768)   ─┐
 ├── audio (T, 1024)  ─┼──► CrossModalFusion (CMA) ──► fused (T, 2816)
 └── video (T, 768)   ─┘
                               ↓
                    2× Conv1D projection → (T, 512)
                               ↓
         Multiscale Transformer Encoder (6 blocks, 2× downsampling each)
                               ↓
              Feature pyramid:  T, T/2, T/4, T/8, T/16, T/32
                               ↓
          ┌──────────────────────────────────────────┐
          │  Classification head  →  hate scores     │
          │  Regression head      →  (d_start, d_end)│
          └──────────────────────────────────────────┘
                               ↓
                  Focal loss + DIoU loss (training)
                  Soft-NMS decoding       (inference)
                               ↓
          List of (start_time, end_time, confidence)
```

### Stage 1 — Cross-Modal Fusion (adapted from MM-HSD CMA)

- **Query**: text (HateBERT) — most direct semantic hate signal
- **Key/Value**: audio (Wav2Vec2) + video (CLIP ViT-L/14)
- **Zero-out strategy**: CMA output is zeroed at timesteps where text features
  are all-zero (silent frames / no transcript).  The model falls back to raw
  audio+video concatenation for those frames.  No precomputed mask file needed.

### Stage 2 — ActionFormer Temporal Backbone

- 2× Conv1D projection layers
- 6 transformer blocks: 1 without downsampling, 5 with 2× downsampling
- Local self-attention with configurable window size (default 19)
- No positional encoding (ActionFormer found it hurts performance)
- Outputs feature pyramid at 6 resolutions

### Stage 3 — Decoder Heads

- **Classification**: 3× Conv1D → 1-channel sigmoid output (binary hate/no-hate)
- **Regression**: 3× Conv1D → 2-channel non-negative output (d_start, d_end)
- **Loss**: Focal loss (α=0.25, γ=2.0) + DIoU regression loss, with center
  sampling (radius=1.5 strides)

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Data Preparation

### 1. Extract features

Run the existing extraction script (already done):

```bash
python data/extract_features.py --input_dir <video_dir> --output_dir data/hatemm/dataset/
```

This creates per-video `.npz` files with keys `video`, `audio`, `text`, `fps`, `duration`.

### 2. Create annotations file

Create `data/hatemm/dataset/annotations.json` in ActivityNet format:

```json
{
  "database": {
    "video_001": {
      "duration": 120.0,
      "subset": "train",
      "annotations": [
        {"segment": [12.5, 28.3], "label": "hate"}
      ]
    },
    "video_002": {
      "duration": 45.0,
      "subset": "val",
      "annotations": []
    }
  }
}
```

**Weak supervision fallback** (HateMM video-level labels):
The HateMM dataset has only video-level binary labels (hate / non-hate), not
temporal segment annotations.  If a video entry has an empty `"annotations"`
list **and** includes `"video_label": "hate"`, the dataset class automatically
treats the *entire* video `[0, duration]` as one hateful segment.  Non-hate
videos get empty annotations.  This allows initial training without frame-level
ground truth.

```json
{
  "database": {
    "hate_video_001": {
      "duration": 60.0,
      "subset": "train",
      "annotations": [],
      "video_label": "hate"
    }
  }
}
```

---

## Training

```bash
cd data/hatemm/scripts
python train.py --config configs/default.yaml --output_dir runs/exp1
```

Options:

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config (required) |
| `--output_dir` | Directory for checkpoints and TensorBoard logs |
| `--seed` | Random seed (default 42) |
| `--resume` | Resume from checkpoint |
| `--no_tb` | Disable TensorBoard |

---

## Evaluation

```bash
python eval.py --config configs/default.yaml \
               --checkpoint runs/exp1/model_best.pth.tar \
               --subset val
```

Reports mAP at tIoU thresholds 0.3, 0.5, 0.7.

---

## Tests

```bash
cd data/hatemm/scripts
python -m pytest tests/test_forward_pass.py -v
# or directly:
python tests/test_forward_pass.py
```

Six tests are run:

| Test | Description |
|------|-------------|
| `test_shape` | Full forward pass; verify output shape |
| `test_zero_out` | CMA output is exactly zero when text is all-zero |
| `test_pyramid` | Feature pyramid has expected levels and resolutions |
| `test_mask` | Loss is zero for fully-padded (all-mask=0) batches |
| `test_gradients` | All trainable parameters receive non-zero gradients |
| `test_npz_round_trip` | Synthetic .npz → Dataset → model pipeline |

> **Note on sequence lengths**: Tests use `max_seq_len=128` and `window_size=-1`
> (global attention).  The production config uses `max_seq_len=2304` and
> `window_size=19`.  For local attention, every pyramid level's sequence length
> must be divisible by `(window_size // 2) * 2`.  For `window_size=19` this
> means each level must be divisible by 18; `max_seq_len=2304` satisfies this
> down to level 6 (`2304 / 32 = 72`, `72 % 18 = 0`).

---

## Differences from Vanilla ActionFormer

1. **Input**: Three separate modality arrays (text, audio, video) from `.npz`
   files, fused via CMA, instead of a single pre-fused feature (e.g. I3D).
2. **CrossModalFusion**: New module inserted before the projection layers.
   Implements MM-HSD-style CMA with text as query, audio+video as key/value,
   plus zero-out for missing text.
3. **Classification**: Binary hate/no-hate (1 sigmoid) instead of 20+ action
   categories.  Configurable via `num_classes`.
4. **Feature stride**: 1 second (1 FPS) instead of ~0.13 s.
5. **Weak supervision**: Dataset handles video-level labels as full-video
   segments when no temporal annotations are available.
6. **NMS**: Pure-Python Soft-NMS (no compiled C++ extension required).

## Differences from Vanilla MM-HSD

1. **Temporal**: Per-timestep features and per-timestep classification + boundary
   regression, instead of one label per video.
2. **Three modalities, no OCR**: HateBERT, Wav2Vec2, CLIP.  No on-screen text.
3. **No per-modality encoders**: ActionFormer's transformer encoder replaces
   MM-HSD's LSTM (video) and FC (text, audio) encoders.
4. **Output**: Temporal segments with boundaries, not a binary video-level label.
5. **Zero-out strategy**: Computed on-the-fly from feature norms, no precomputed
   mask file.

---

## Directory Structure

```
data/hatemm/scripts/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── libs/
│   ├── modeling/
│   │   ├── blocks.py            # MaskedConv1D, LayerNorm, LocalMHCA, TransformerBlock
│   │   ├── cross_modal_fusion.py # CMA + zero-out + concat
│   │   ├── backbone.py          # Multiscale transformer encoder
│   │   ├── heads.py             # Cls + Reg conv heads
│   │   └── meta_arch.py         # Full model + losses + inference
│   ├── datasets/
│   │   └── hatemm.py            # Dataset + DataLoader
│   └── utils/
│       ├── nms.py               # Pure-Python Soft-NMS
│       ├── train_utils.py       # Optimizer, scheduler, EMA, training loop
│       └── eval_utils.py        # Temporal mAP evaluation
├── train.py
├── eval.py
└── tests/
    └── test_forward_pass.py
```
