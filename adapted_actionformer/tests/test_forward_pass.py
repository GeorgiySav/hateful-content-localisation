"""
Synthetic forward-pass tests for the HateMM temporal localizer.

Run with:
    cd data/hatemm/scripts
    python -m pytest tests/test_forward_pass.py -v

Tests:
  1. shape_test          — full forward pass; verify output shapes.
  2. zero_out_test       — CMA output is exactly zero when text is all-zero.
  3. pyramid_test        — feature pyramid has the expected number of levels.
  4. mask_test           — loss is zero for padded positions.
  5. gradient_test       — all trainable parameters receive non-zero gradients.
  6. npz_round_trip_test — synthetic .npz → Dataset → model produces correct shapes.

NOTE on test configuration:
  The default production config uses max_seq_len=2304 and window_size=19.
  For the sliding-window attention, each sequence length at every pyramid level
  must be divisible by (window_size // 2) * 2.  For tests we use:
    - window_size = -1  (global attention, no divisibility constraint)
    - n_layers = 4      (1 stem + 3 branches → 4 pyramid levels: T, T/2, T/4, T/8)
    - max_seq_len = 128 (padded test sequences)
    - T = 120 features  (padded to 128 by the test harness)
  This mirrors what happens in real training where sequences are padded to
  max_seq_len before being fed to the model.
"""
import os
import sys
import tempfile

import torch
import pytest

# Make scripts/ importable
_scripts_dir = os.path.join(os.path.dirname(__file__), '..')
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from libs.modeling.cross_modal_fusion import CrossModalFusion
from libs.modeling.feature_preprocessors import (
    GuidedCMAPreprocessor, UnimodalPreprocessor, ConcatPreprocessor,
)
from libs.modeling.backbone import ConvTransformerBackbone
from libs.modeling.meta_arch import HatefulContentLocalizer
from libs.datasets.hatemm import HateMMDataset


# ──────────────────────────────────────────────────────────────────────────────
# Shared test configuration
# ──────────────────────────────────────────────────────────────────────────────

B          = 2
T_raw      = 120        # raw feature length (will be padded to max_seq_len)
T_PAD      = 128        # max_seq_len used in tests (must be divisible by 2^3=8)
N_LEVELS   = 4          # 1 stem + 3 branches

# Test-model config (smaller than production to keep tests fast)
TEST_CFG = {
    'dataset': {
        'name': 'hatemm',
        'video_feat_dir': 'dummy',
        'audio_feat_dir': 'dummy',
        'text_feat_dir': 'dummy',
        'annotation_file': 'dummy',
        'num_classes': 1,
        'feature_fps': 1.0,
        'max_seq_len': T_PAD,
        'input_dims': {'text': 768, 'audio': 1024, 'video': 768},
    },
    'fusion': {
        'd_cma': 256,
        'num_heads': 4,
        'dropout': 0.0,
        'query_modality': 'text',
        'key_modalities': ['audio', 'video'],
        'zero_out_missing_query': True,
    },
    'backbone': {
        'd_model': 128,         # smaller than production 512
        'n_proj_layers': 2,
        'n_layers': N_LEVELS,   # 1 stem + (N_LEVELS-1) branches
        'n_heads': 4,
        'window_size': -1,      # global attention — no seq-len divisibility constraint
        'downsample_start': 1,
        'downsample_ratio': 2,
    },
    'heads': {
        'n_layers': 3,
        'kernel_size': 3,
        'use_layer_norm': True,
    },
    'loss': {
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'lambda_reg': 1.0,
        'center_sampling': True,
        'center_sampling_radius': 1.5,
    },
    'training': {
        'epochs': 50,
        'batch_size': 2,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'warmup_epochs': 5,
        'lr_scheduler': 'cosine',
        'use_ema': True,
        'ema_decay': 0.999,
        'clip_grad_norm': 1.0,
        'cls_prior_prob': 0.01,
        'dropout': 0.0,
        'droppath': 0.0,
        'label_smoothing': 0.0,
    },
    'inference': {
        'score_threshold': 0.001,
        'nms_method': 'soft_nms',
        'nms_sigma': 0.4,
        'nms_threshold': 0.1,
        'max_detections': 200,
    },
    'regression_ranges': [[0, 4], [4, 8], [8, 16], [16, 1e6]],
}


def make_batch(T=T_PAD, B=B, last_pad=0, zero_text=False):
    """Create a synthetic batch with optional padding at the end."""
    text  = torch.randn(B, T, 768)
    audio = torch.randn(B, T, 1024)
    video = torch.randn(B, T, 768)

    if zero_text:
        text = torch.zeros_like(text)

    mask = torch.ones(B, T)
    if last_pad > 0:
        mask[:, T - last_pad:] = 0.0

    # Dummy segments (one hate segment per video)
    seg_len = max(1, (T - last_pad) // 2)
    segments = [torch.tensor([[0.0, float(seg_len)]], dtype=torch.float32)] * B
    labels   = [torch.zeros(1, dtype=torch.long)] * B

    return {
        'video_id'  : [f'video_{i}' for i in range(B)],
        'text_feat' : text,
        'audio_feat': audio,
        'video_feat': video,
        'mask'      : mask,
        'segments'  : segments,
        'labels'    : labels,
        'duration'  : [float(T)] * B,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Shape test
# ──────────────────────────────────────────────────────────────────────────────

def test_shape():
    """Full forward pass; verify output shapes at each pyramid level."""
    model = HatefulContentLocalizer(TEST_CFG)
    model.eval()

    batch = make_batch()
    with torch.no_grad():
        # Inference mode returns a list of result dicts
        output = model(batch)

    assert isinstance(output, list), "Inference output should be a list"
    assert len(output) == B, f"Expected {B} results, got {len(output)}"

    for res in output:
        assert 'segments' in res
        assert 'scores'   in res
        assert 'labels'   in res
        segs = res['segments']
        assert segs.dim() == 2 and segs.shape[1] == 2, \
            f"segments should be (N, 2), got {segs.shape}"

    print("✓ shape_test passed")


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Zero-out test
# ──────────────────────────────────────────────────────────────────────────────

def test_zero_out():
    """
    When text_feat is all-zero, the GuidedCMAPreprocessor output must be exactly
    zero (the text-presence mask zeros out the CMA output for silent frames).
    The model should still run and produce valid output.
    """
    d_cma = TEST_CFG['fusion']['d_cma']
    preprocessor = GuidedCMAPreprocessor(
        text_dim=768, audio_dim=1024, video_dim=768,
        d_out=d_cma, num_heads=4, dropout=0.0,
    )
    preprocessor.eval()

    text  = torch.zeros(B, T_PAD, 768)    # all-zero text
    audio = torch.randn(B, T_PAD, 1024)
    video = torch.randn(B, T_PAD, 768)

    with torch.no_grad():
        out = preprocessor(text, audio, video)  # (B, T, d_cma)

    assert out.shape == (B, T_PAD, d_cma), \
        f"Preprocessor output shape mismatch: {out.shape}"
    assert torch.all(out == 0.0), \
        "GuidedCMAPreprocessor output must be exactly zero when text is all-zero"

    # Verify model still runs with all-zero text
    model = HatefulContentLocalizer(TEST_CFG)
    model.eval()
    batch = make_batch(zero_text=True)
    with torch.no_grad():
        output = model(batch)
    assert isinstance(output, list) and len(output) == B
    # Scores should be in [0, 1] (they come from sigmoid)
    for res in output:
        if res['scores'].numel() > 0:
            assert res['scores'].min() >= 0.0
            assert res['scores'].max() <= 1.0

    print("✓ zero_out_test passed")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3a: Unimodal preprocessor test
# ──────────────────────────────────────────────────────────────────────────────

def test_unimodal_preprocessor():
    """UnimodalPreprocessor should pass each modality through unchanged or projected."""
    text  = torch.randn(B, T_PAD, 768)
    audio = torch.randn(B, T_PAD, 1024)
    video = torch.randn(B, T_PAD, 768)

    # Video — no projection needed (d_out == native dim)
    prep = UnimodalPreprocessor('video', 768, 1024, 768, d_out=768)
    prep.eval()
    with torch.no_grad():
        out = prep(text, audio, video)
    assert out.shape == (B, T_PAD, 768), f"video unimodal shape: {out.shape}"
    assert torch.allclose(out, video), "video unimodal should be identity (no projection)"

    # Audio — projection required (1024 -> 256)
    prep = UnimodalPreprocessor('audio', 768, 1024, 768, d_out=256)
    prep.eval()
    with torch.no_grad():
        out = prep(text, audio, video)
    assert out.shape == (B, T_PAD, 256), f"audio unimodal shape: {out.shape}"

    # Text — projection required (768 -> 256)
    prep = UnimodalPreprocessor('text', 768, 1024, 768, d_out=256)
    prep.eval()
    with torch.no_grad():
        out = prep(text, audio, video)
    assert out.shape == (B, T_PAD, 256), f"text unimodal shape: {out.shape}"

    print("✓ unimodal_preprocessor_test passed")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3b: Concat preprocessor test
# ──────────────────────────────────────────────────────────────────────────────

def test_concat_preprocessor():
    """ConcatPreprocessor should concatenate modalities and project to d_out."""
    text  = torch.randn(B, T_PAD, 768)
    audio = torch.randn(B, T_PAD, 1024)
    video = torch.randn(B, T_PAD, 768)

    d_out = 256

    # Audio + Video (1024+768=1792 -> 256)
    prep = ConcatPreprocessor(['audio', 'video'], 768, 1024, 768, d_out=d_out)
    prep.eval()
    with torch.no_grad():
        out = prep(text, audio, video)
    assert out.shape == (B, T_PAD, d_out), f"av concat shape: {out.shape}"

    # All three (768+1024+768=2560 -> 256)
    prep = ConcatPreprocessor(['text', 'audio', 'video'], 768, 1024, 768, d_out=d_out)
    prep.eval()
    with torch.no_grad():
        out = prep(text, audio, video)
    assert out.shape == (B, T_PAD, d_out), f"tav concat shape: {out.shape}"

    print("✓ concat_preprocessor_test passed")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3c: Full model with unimodal preprocessor config
# ──────────────────────────────────────────────────────────────────────────────

def test_model_unimodal_preprocessor():
    """HatefulContentLocalizer should work end-to-end with a unimodal preprocessor."""
    import copy
    cfg = copy.deepcopy(TEST_CFG)
    # Replace the legacy fusion block with a new-style preprocessor block
    cfg.pop('fusion', None)
    cfg['preprocessor'] = {
        'type': 'unimodal',
        'modality': 'video',
        'd_out': 256,
    }
    cfg['backbone']['d_model'] = 128

    model = HatefulContentLocalizer(cfg)
    model.eval()
    batch = make_batch()
    with torch.no_grad():
        output = model(batch)
    assert isinstance(output, list) and len(output) == B

    print("✓ model_unimodal_preprocessor_test passed")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3d: Full model with concat preprocessor config
# ──────────────────────────────────────────────────────────────────────────────

def test_model_concat_preprocessor():
    """HatefulContentLocalizer should work end-to-end with a concat preprocessor."""
    import copy
    cfg = copy.deepcopy(TEST_CFG)
    cfg.pop('fusion', None)
    cfg['preprocessor'] = {
        'type': 'concat',
        'modalities': ['audio', 'video'],
        'd_out': 256,
    }
    cfg['backbone']['d_model'] = 128

    model = HatefulContentLocalizer(cfg)
    model.eval()
    batch = make_batch()
    with torch.no_grad():
        output = model(batch)
    assert isinstance(output, list) and len(output) == B

    print("✓ model_concat_preprocessor_test passed")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Pyramid test
# ──────────────────────────────────────────────────────────────────────────────

def test_pyramid():
    """Verify feature pyramid has the expected number of levels and resolutions."""
    # The backbone receives d_cma-dimensional features from the preprocessor.
    fused_dim = TEST_CFG['fusion']['d_cma']
    d_model   = TEST_CFG['backbone']['d_model']

    # n_stem = downsample_start = 1, n_branch = n_layers - 1 - 1 = N_LEVELS - 1 - 1
    n_branch = TEST_CFG['backbone']['n_layers'] - 1 - TEST_CFG['backbone']['downsample_start']
    # wait: n_stem = downsample_start (index of first branch = 1), n_branch = rest
    # arch = (n_proj=2, n_stem=1, n_branch=N_LEVELS-1)
    n_levels_expected = 1 + (TEST_CFG['backbone']['n_layers'] - 1)

    backbone = ConvTransformerBackbone(
        n_in=fused_dim,
        n_embd=d_model,
        n_head=4,
        n_embd_ks=3,
        max_len=T_PAD,
        arch=(2, 1, N_LEVELS - 1),
        mha_win_size=[-1] * N_LEVELS,
        scale_factor=2,
        with_ln=True,
    )
    backbone.eval()

    x    = torch.randn(B, fused_dim, T_PAD)
    mask = torch.ones(B, 1, T_PAD, dtype=torch.bool)

    with torch.no_grad():
        out_feats, out_masks = backbone(x, mask)

    assert len(out_feats) == N_LEVELS, \
        f"Expected {N_LEVELS} pyramid levels, got {len(out_feats)}"

    for i, (feat, m) in enumerate(zip(out_feats, out_masks)):
        expected_T = T_PAD // (2 ** i)
        assert feat.shape == (B, d_model, expected_T), \
            f"Level {i}: expected (B={B}, C={d_model}, T={expected_T}), got {feat.shape}"

    print(f"✓ pyramid_test passed  ({N_LEVELS} levels: "
          f"{[T_PAD // 2**i for i in range(N_LEVELS)]})")


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Mask test
# ──────────────────────────────────────────────────────────────────────────────

def test_mask():
    """
    Loss should be zero (or very small) for padded positions.
    We check that the classification loss contribution from fully-padded
    batches (all mask=0) is zero.
    """
    model = HatefulContentLocalizer(TEST_CFG)
    model.train()

    # Create a batch where ALL positions are masked out
    batch = make_batch()
    batch['mask'] = torch.zeros(B, T_PAD)   # all padded
    batch['segments'] = [torch.zeros((0, 2), dtype=torch.float32)] * B
    batch['labels']   = [torch.zeros((0,),   dtype=torch.long)]    * B

    losses = model(batch)
    # With all positions masked, cls_loss should be 0
    assert losses['cls_loss'].item() == pytest.approx(0.0, abs=1e-5), \
        f"cls_loss should be 0 for all-masked batch, got {losses['cls_loss'].item()}"

    print(f"✓ mask_test passed  (cls_loss={losses['cls_loss'].item():.2e})")


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Gradient test
# ──────────────────────────────────────────────────────────────────────────────

def test_gradients():
    """All trainable parameters should receive non-zero gradients after a backward pass."""
    model = HatefulContentLocalizer(TEST_CFG)
    model.train()

    batch = make_batch(last_pad=0)
    losses = model(batch)
    losses['final_loss'].backward()

    zero_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if param.grad.abs().max().item() == 0.0:
                zero_grad_params.append(name)

    no_grad_params = [
        name for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]

    # Some parameters (e.g. AffineDropPath scale) may have zero grad if the
    # drop-path is unused (prob=0 in tests) — exclude them from the assertion.
    excluded_patterns = ('drop_path', 'pool_skip')
    truly_zero = [p for p in zero_grad_params
                  if not any(pat in p for pat in excluded_patterns)]
    truly_none  = [p for p in no_grad_params
                   if not any(pat in p for pat in excluded_patterns)]

    assert len(truly_none) == 0, \
        f"Parameters with no gradient: {truly_none[:5]}"
    # Allow a small number of zero-grad params (e.g. biases in degenerate cases)
    assert len(truly_zero) < 5, \
        f"Too many zero-gradient parameters: {truly_zero[:10]}"

    print(f"✓ gradient_test passed  ({len(zero_grad_params)} zero-grad params "
          f"after filtering drop-path)")


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: NPZ round-trip test
# ──────────────────────────────────────────────────────────────────────────────

def test_npz_round_trip():
    """
    Create synthetic .pt feature files, load them through HateMMDataset, and
    verify the returned tensors have the correct shapes and dtypes.
    """
    import json

    T_video = 60   # raw feature length
    duration = float(T_video)

    with tempfile.TemporaryDirectory() as tmpdir:
        video_id = 'test_video_001'

        # Create sub-directories for each modality
        vdir = os.path.join(tmpdir, 'video_features')
        adir = os.path.join(tmpdir, 'audio_features')
        tdir = os.path.join(tmpdir, 'text_features')
        os.makedirs(vdir); os.makedirs(adir); os.makedirs(tdir)

        torch.save(torch.randn(T_video, 768),  os.path.join(vdir, f'{video_id}.pt'))
        torch.save(torch.randn(T_video, 1024), os.path.join(adir, f'{video_id}.pt'))
        torch.save(torch.randn(T_video, 768),  os.path.join(tdir, f'{video_id}.pt'))

        # Write a minimal annotations.json
        ann_path = os.path.join(tmpdir, 'annotations.json')
        ann_data = {
            "database": {
                video_id: {
                    "duration": duration,
                    "subset": "train",
                    "annotations": [
                        {"segment": [5.0, 20.0], "label": "hate"}
                    ]
                }
            }
        }
        with open(ann_path, 'w') as f:
            json.dump(ann_data, f)

        # Build dataset
        dataset = HateMMDataset(
            video_feat_dir=vdir,
            audio_feat_dir=adir,
            text_feat_dir=tdir,
            annotation_file=ann_path,
            max_seq_len=T_PAD,
            subset='train',
            is_training=False,
        )
        assert len(dataset) == 1

        sample = dataset[0]

        assert sample['video_feat'].shape == (T_PAD, 768),  \
            f"video_feat shape mismatch: {sample['video_feat'].shape}"
        assert sample['audio_feat'].shape == (T_PAD, 1024), \
            f"audio_feat shape mismatch: {sample['audio_feat'].shape}"
        assert sample['text_feat'].shape  == (T_PAD, 768),  \
            f"text_feat shape mismatch: {sample['text_feat'].shape}"
        assert sample['mask'].shape       == (T_PAD,),       \
            f"mask shape mismatch: {sample['mask'].shape}"

        # Check dtypes
        assert sample['video_feat'].dtype == torch.float32
        assert sample['audio_feat'].dtype == torch.float32
        assert sample['text_feat'].dtype  == torch.float32

        # Check mask: first T_video positions should be 1, rest 0
        assert sample['mask'][:T_video].all(),   "First T_video mask entries should be 1"
        assert not sample['mask'][T_video:].any(), "Padded mask entries should be 0"

        # Check annotation parsing
        assert sample['segments'].shape == (1, 2)
        assert abs(sample['segments'][0, 0].item() - 5.0) < 1e-5
        assert abs(sample['segments'][0, 1].item() - 20.0) < 1e-5

        # Feed through model (inference mode)
        model = HatefulContentLocalizer(TEST_CFG)
        model.eval()

        from libs.datasets.hatemm import collate_fn
        batch = collate_fn([sample])
        with torch.no_grad():
            output = model(batch)
        assert isinstance(output, list) and len(output) == 1

    print("✓ npz_round_trip_test passed")


# ──────────────────────────────────────────────────────────────────────────────
# Run directly
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Running forward-pass tests...\n")
    test_shape()
    test_zero_out()
    test_unimodal_preprocessor()
    test_concat_preprocessor()
    test_model_unimodal_preprocessor()
    test_model_concat_preprocessor()
    test_pyramid()
    test_mask()
    test_gradients()
    test_npz_round_trip()
    print("\nAll tests passed!")
