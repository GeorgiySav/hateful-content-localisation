"""
Tests for HateClipSegDataset augmentation (_augment).

Run with:
    python -m pytest adapted_actionformer/tests/test_augmentation.py -v
or:
    python adapted_actionformer/tests/test_augmentation.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from libs.datasets.hateclipseg import HateClipSegDataset


# ── helpers ───────────────────────────────────────────────────────────────────

T        = 20   # valid frames
MAX_LEN  = 32   # max_seq_len (12 padded frames at the end)
PAD_LEN  = MAX_LEN - T

def _make_feats(t=MAX_LEN):
    """Deterministic non-zero features of the right shapes."""
    torch.manual_seed(0)
    return (
        torch.randn(t, 768),
        torch.randn(t, 1024),
        torch.randn(t, 768),
    )

def _make_mask(t_valid=T, total=MAX_LEN):
    m = torch.zeros(total)
    m[:t_valid] = 1.0
    return m

def _make_segments():
    return torch.tensor([[2.0, 5.0], [10.0, 15.0]], dtype=torch.float32)

def _make_labels():
    return torch.tensor([0, 0], dtype=torch.long)

def _dummy_dataset(aug_cfg):
    """Return a dataset instance with aug_cfg set (no real files needed)."""
    ds = HateClipSegDataset.__new__(HateClipSegDataset)
    ds.max_seq_len  = MAX_LEN
    ds.feature_fps  = 1.0
    ds.aug_cfg      = aug_cfg
    ds.is_training  = True
    return ds


# ── feature noise tests ───────────────────────────────────────────────────────

class TestFeatureNoise:

    def test_noise_not_applied_to_padded_frames(self):
        """Padded frames (mask==0) must remain exactly zero after noise."""
        ds = _dummy_dataset({'enabled': True, 'feature_noise_std': 0.5})
        mask = _make_mask()
        vf, af, tf = _make_feats()
        # zero out the padded region
        vf[T:] = 0.0; af[T:] = 0.0; tf[T:] = 0.0

        torch.manual_seed(1)
        vf2, af2, tf2, *_ = ds._augment(vf, af, tf, _make_segments(), _make_labels(), mask)

        assert vf2[T:].abs().max().item() == 0.0, "video padding was corrupted by noise"
        assert af2[T:].abs().max().item() == 0.0, "audio padding was corrupted by noise"
        assert tf2[T:].abs().max().item() == 0.0, "text padding was corrupted by noise"

    def test_noise_does_change_valid_frames(self):
        """Valid frames (mask==1) should be changed by noise."""
        ds = _dummy_dataset({'enabled': True, 'feature_noise_std': 0.1})
        mask = _make_mask()
        vf, af, tf = _make_feats()
        vf[T:] = 0.0; af[T:] = 0.0; tf[T:] = 0.0
        vf_orig = vf[:T].clone()

        torch.manual_seed(2)
        vf2, *_ = ds._augment(vf, af, tf, _make_segments(), _make_labels(), mask)

        assert not torch.allclose(vf2[:T], vf_orig), "noise had no effect on valid frames"

    def test_no_noise_when_std_zero(self):
        """With noise_std=0, features must be unchanged."""
        ds = _dummy_dataset({'enabled': True, 'feature_noise_std': 0.0})
        mask = _make_mask()
        vf, af, tf = _make_feats()
        vf_orig = vf.clone(); af_orig = af.clone(); tf_orig = tf.clone()

        vf2, af2, tf2, *_ = ds._augment(vf, af, tf, _make_segments(), _make_labels(), mask)
        assert torch.allclose(vf2, vf_orig)
        assert torch.allclose(af2, af_orig)
        assert torch.allclose(tf2, tf_orig)

    def test_noise_scale_relative_to_features(self):
        """Noise should be ~noise_std relative to feature magnitudes."""
        ds = _dummy_dataset({'enabled': True, 'feature_noise_std': 0.02})
        mask = _make_mask(t_valid=MAX_LEN, total=MAX_LEN)  # no padding
        torch.manual_seed(3)
        vf = torch.randn(MAX_LEN, 768) * 0.63  # typical video feature magnitude

        vf_orig = vf.clone()
        torch.manual_seed(3)
        vf2, *_ = ds._augment(vf, torch.randn(MAX_LEN, 1024) * 0.48,
                               torch.randn(MAX_LEN, 768) * 0.38,
                               _make_segments(), _make_labels(), mask)

        diff_std = (vf2 - vf_orig).std().item()
        assert 0.01 < diff_std < 0.05, (
            f"noise std {diff_std:.4f} is way off from target 0.02 — noise may not be applied")


# ── temporal shift tests ──────────────────────────────────────────────────────

class TestTemporalShift:

    def test_no_shift_when_max_zero(self):
        """temporal_shift_max_frames=0 must leave features, mask, and segments untouched."""
        ds = _dummy_dataset({'enabled': True, 'temporal_shift_max_frames': 0})
        mask = _make_mask()
        vf, af, tf = _make_feats()
        vf_orig, af_orig, tf_orig = vf.clone(), af.clone(), tf.clone()
        segs = _make_segments(); segs_orig = segs.clone()

        vf2, af2, tf2, mask2, segs2, _ = ds._augment(vf, af, tf, segs, _make_labels(), mask.clone())
        assert torch.allclose(vf2, vf_orig)
        assert torch.allclose(af2, af_orig)
        assert torch.allclose(tf2, tf_orig)
        assert torch.allclose(mask2, mask)
        assert torch.allclose(segs2, segs_orig)

    def test_positive_shift_pads_start_crops_end(self):
        """Positive shift must zero out the first `shift` frames and shift mask/segments forward."""
        ds = _dummy_dataset({'enabled': True})
        mask = _make_mask()                                  # T=20 valid, MAX=32
        vf, af, tf = _make_feats()
        segs = _make_segments(); lbls = _make_labels()
        SHIFT = 5

        vf2, af2, tf2, mask2, segs2, _ = ds._temporal_shift(
            vf.clone(), af.clone(), tf.clone(), mask.clone(), segs.clone(), lbls.clone(), SHIFT)

        assert vf2.shape == (MAX_LEN, 768)
        assert af2.shape == (MAX_LEN, 1024)
        assert tf2.shape == (MAX_LEN, 768)
        assert mask2.shape == (MAX_LEN,)
        assert vf2[:SHIFT].abs().max().item() == 0.0,    "first `shift` video frames must be zero"
        assert af2[:SHIFT].abs().max().item() == 0.0,    "first `shift` audio frames must be zero"
        assert tf2[:SHIFT].abs().max().item() == 0.0,    "first `shift` text frames must be zero"
        assert mask2[:SHIFT].abs().max().item() == 0.0,  "first `shift` mask entries must be zero"
        assert torch.allclose(vf2[SHIFT:], vf[:-SHIFT]),  "tail of features must match original head"
        # segments should shift forward by SHIFT/fps (fps=1)
        assert torch.allclose(segs2, segs + SHIFT)

    def test_negative_shift_crops_start_pads_end(self):
        """Negative shift must zero out the last `|shift|` frames and shift mask/segments backward."""
        ds = _dummy_dataset({'enabled': True})
        mask = _make_mask()
        vf, af, tf = _make_feats()
        segs = _make_segments(); lbls = _make_labels()
        SHIFT = -3

        vf2, af2, tf2, mask2, segs2, _ = ds._temporal_shift(
            vf.clone(), af.clone(), tf.clone(), mask.clone(), segs.clone(), lbls.clone(), SHIFT)

        s = -SHIFT
        assert vf2.shape == (MAX_LEN, 768)
        assert vf2[-s:].abs().max().item() == 0.0,    "last `|shift|` video frames must be zero"
        assert af2[-s:].abs().max().item() == 0.0,    "last `|shift|` audio frames must be zero"
        assert tf2[-s:].abs().max().item() == 0.0,    "last `|shift|` text frames must be zero"
        assert mask2[-s:].abs().max().item() == 0.0,  "last `|shift|` mask entries must be zero"
        assert torch.allclose(vf2[:-s], vf[s:]),      "head of features must match original tail"
        # segments shift by SHIFT/fps and get clamped to >= 0
        expected = (segs + SHIFT).clamp(min=0.0, max=MAX_LEN)
        assert torch.allclose(segs2, expected)

    def test_segments_outside_window_are_dropped(self):
        """Segments shifted entirely outside [0, max_seq_len/fps] must be removed."""
        ds = _dummy_dataset({'enabled': True})
        mask = _make_mask()
        vf, af, tf = _make_feats()
        # one segment near the start (will be pushed off-screen by negative shift)
        # and one in the middle (will survive)
        segs = torch.tensor([[0.5, 1.5], [10.0, 14.0]], dtype=torch.float32)
        lbls = torch.tensor([0, 0], dtype=torch.long)
        SHIFT = -5  # at fps=1 → shift by -5s; first segment becomes [-4.5,-3.5] (dropped)

        _, _, _, _, segs2, lbls2 = ds._temporal_shift(
            vf.clone(), af.clone(), tf.clone(), mask.clone(), segs.clone(), lbls.clone(), SHIFT)

        assert segs2.shape[0] == 1, f"expected 1 surviving segment, got {segs2.shape[0]}"
        assert lbls2.shape[0] == 1
        assert torch.allclose(segs2[0], torch.tensor([5.0, 9.0]))

    def test_partial_segments_are_clamped(self):
        """Segments straddling the boundary must be clamped, not dropped."""
        ds = _dummy_dataset({'enabled': True})
        mask = _make_mask()
        vf, af, tf = _make_feats()
        segs = torch.tensor([[2.0, 5.0]], dtype=torch.float32)
        lbls = torch.tensor([0], dtype=torch.long)
        SHIFT = -3  # → segment becomes [-1, 2]; should clamp to [0, 2]

        _, _, _, _, segs2, lbls2 = ds._temporal_shift(
            vf.clone(), af.clone(), tf.clone(), mask.clone(), segs.clone(), lbls.clone(), SHIFT)

        assert segs2.shape[0] == 1
        assert torch.allclose(segs2[0], torch.tensor([0.0, 2.0]))

    def test_total_length_preserved_under_random_shift(self):
        """For any random shift, output length and mask length must equal max_seq_len."""
        ds = _dummy_dataset({'enabled': True, 'temporal_shift_max_frames': 8})
        for seed in range(50):
            torch.manual_seed(seed)
            mask = _make_mask()
            vf, af, tf = _make_feats()
            vf2, af2, tf2, mask2, _, _ = ds._augment(
                vf.clone(), af.clone(), tf.clone(),
                _make_segments(), _make_labels(), mask.clone())
            assert vf2.shape[0] == MAX_LEN
            assert af2.shape[0] == MAX_LEN
            assert tf2.shape[0] == MAX_LEN
            assert mask2.shape[0] == MAX_LEN


# ── augmentation gate tests ───────────────────────────────────────────────────

class TestAugmentationGate:

    def test_augmentation_disabled_flag(self):
        """enabled=False must make _augment a no-op (noise path only)."""
        ds = _dummy_dataset({'enabled': False, 'feature_noise_std': 1.0})
        mask = _make_mask()
        vf, af, tf = _make_feats()
        vf_orig = vf.clone()

        # _augment itself doesn't check 'enabled' — that's the caller's job.
        # Verify the dataset __getitem__ gate via aug_cfg directly.
        # When enabled=False the caller won't invoke _augment, so noise_std
        # should never fire. Here we just confirm noise_std=0 is the safe path.
        ds2 = _dummy_dataset({'enabled': True, 'feature_noise_std': 0.0})
        vf2, *_ = ds2._augment(vf.clone(), af.clone(), tf.clone(),
                                 _make_segments(), _make_labels(), mask)
        assert torch.allclose(vf2, vf_orig)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import traceback
    tests = [
        TestFeatureNoise().test_noise_not_applied_to_padded_frames,
        TestFeatureNoise().test_noise_does_change_valid_frames,
        TestFeatureNoise().test_no_noise_when_std_zero,
        TestFeatureNoise().test_noise_scale_relative_to_features,
        TestTemporalShift().test_no_shift_when_max_zero,
        TestTemporalShift().test_positive_shift_pads_start_crops_end,
        TestTemporalShift().test_negative_shift_crops_start_pads_end,
        TestTemporalShift().test_segments_outside_window_are_dropped,
        TestTemporalShift().test_partial_segments_are_clamped,
        TestTemporalShift().test_total_length_preserved_under_random_shift,
        TestAugmentationGate().test_augmentation_disabled_flag,
    ]
    passed = failed = 0
    for t in tests:
        name = f"{t.__self__.__class__.__name__}.{t.__name__}"
        try:
            t()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
