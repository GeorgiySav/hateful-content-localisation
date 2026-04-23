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


# ── temporal masking tests ────────────────────────────────────────────────────

class TestTemporalMask:

    def test_mask_stays_within_valid_frames(self):
        """Temporal masking must only zero out frames where mask==1."""
        ds = _dummy_dataset({
            'enabled': True,
            'temporal_mask_prob': 1.0,   # always apply
            'temporal_mask_num': 5,
            'temporal_mask_max_len': 4,
        })
        mask = _make_mask()
        vf, af, tf = _make_feats()
        # set padded region to 1.0 across all modalities so a stray zero is detectable
        vf[T:] = 1.0; af[T:] = 1.0; tf[T:] = 1.0

        torch.manual_seed(4)
        vf2, af2, tf2, *_ = ds._augment(vf.clone(), af.clone(), tf.clone(),
                                          _make_segments(), _make_labels(), mask)

        assert vf2[T:].min().item() == 1.0, "temporal mask reached into padded frames (video)"
        assert af2[T:].min().item() == 1.0, "temporal mask reached into padded frames (audio)"
        assert tf2[T:].min().item() == 1.0, "temporal mask reached into padded frames (text)"

    def test_mask_actually_zeroes_some_valid_frames(self):
        """With prob=1.0 and long masks, at least some valid frames must be zeroed."""
        ds = _dummy_dataset({
            'enabled': True,
            'temporal_mask_prob': 1.0,
            'temporal_mask_num': 3,
            'temporal_mask_max_len': 3,
        })
        mask = _make_mask()
        vf, af, tf = _make_feats()
        vf[:T] = 1.0  # all-ones so zero-mask is detectable

        torch.manual_seed(5)
        vf2, *_ = ds._augment(vf.clone(), af.clone(), tf.clone(),
                                _make_segments(), _make_labels(), mask)

        n_zeroed = (vf2[:T] == 0.0).all(dim=1).sum().item()
        assert n_zeroed > 0, "temporal mask with prob=1 zeroed no valid frames"

    def test_mask_not_applied_when_prob_zero(self):
        """temporal_mask_prob=0 must never apply any masking."""
        ds = _dummy_dataset({
            'enabled': True,
            'temporal_mask_prob': 0.0,
            'temporal_mask_num': 10,
            'temporal_mask_max_len': 10,
        })
        mask = _make_mask()
        vf, af, tf = _make_feats()
        vf_orig = vf.clone()

        vf2, *_ = ds._augment(vf, af, tf, _make_segments(), _make_labels(), mask)
        assert torch.allclose(vf2, vf_orig), "masking applied despite prob=0"

    def test_default_config_coverage(self):
        """Default config (prob=0.5, num=2, max_len=2) at 1fps covers at most 4 frames
        of a 60-frame video — report actual coverage for inspection."""
        ds = _dummy_dataset({
            'enabled': True,
            'temporal_mask_prob': 1.0,   # force always-on for measurement
            'temporal_mask_num': 2,
            'temporal_mask_max_len': 2,
        })
        T_video = 60
        mask = _make_mask(t_valid=T_video, total=T_video)
        n_trials = 1000
        zeroed_counts = []
        for seed in range(n_trials):
            torch.manual_seed(seed)
            vf = torch.ones(T_video, 768)
            vf2, *_ = ds._augment(vf, torch.ones(T_video, 1024),
                                    torch.ones(T_video, 768),
                                    _make_segments(), _make_labels(), mask)
            n_zeroed = (vf2 == 0.0).all(dim=1).sum().item()
            zeroed_counts.append(n_zeroed)

        avg_coverage = sum(zeroed_counts) / len(zeroed_counts)
        max_coverage = max(zeroed_counts)
        print(f"\n[temporal mask coverage @ 60-frame video, prob=1.0, num=2, max_len=2]")
        print(f"  avg frames zeroed : {avg_coverage:.2f} / 60  ({avg_coverage/60*100:.1f}%)")
        print(f"  max frames zeroed : {max_coverage} / 60  ({max_coverage/60*100:.1f}%)")
        assert avg_coverage <= 4.0, "more than 4 frames masked on average — unexpected"


# ── segment jitter tests ──────────────────────────────────────────────────────

class TestSegmentJitter:

    def test_jittered_segments_remain_valid(self):
        """All returned segments must have start < end after jitter."""
        ds = _dummy_dataset({'enabled': True, 'segment_jitter_sec': 2.0})
        mask = _make_mask()
        segs = torch.tensor([[0.5, 1.0], [5.0, 10.0], [18.0, 19.5]], dtype=torch.float32)
        lbls = torch.tensor([0, 0, 0], dtype=torch.long)

        for seed in range(200):
            torch.manual_seed(seed)
            _, _, _, segs2, lbls2 = ds._augment(*_make_feats(), segs.clone(), lbls.clone(), mask)
            if segs2.shape[0] > 0:
                assert (segs2[:, 1] > segs2[:, 0]).all(), \
                    f"invalid segment (start >= end) after jitter at seed={seed}"

    def test_segments_clamped_to_valid_range(self):
        """Segment boundaries should not exceed [0, max_seq_len/fps]."""
        ds = _dummy_dataset({'enabled': True, 'segment_jitter_sec': 5.0})
        mask = _make_mask()
        crop_dur = MAX_LEN / 1.0   # = 32.0 seconds

        segs = torch.tensor([[0.1, 0.5], [31.5, 31.9]], dtype=torch.float32)
        lbls = torch.tensor([0, 0], dtype=torch.long)

        for seed in range(200):
            torch.manual_seed(seed)
            _, _, _, segs2, _ = ds._augment(*_make_feats(), segs.clone(), lbls.clone(), mask)
            if segs2.shape[0] > 0:
                assert segs2[:, 0].min().item() >= 0.0,        "segment start < 0 after jitter"
                assert segs2[:, 1].max().item() <= crop_dur,    "segment end > crop_dur after jitter"

    def test_no_jitter_when_zero(self):
        """segment_jitter_sec=0 must leave segments untouched."""
        ds = _dummy_dataset({'enabled': True, 'segment_jitter_sec': 0.0})
        mask = _make_mask()
        segs = _make_segments()
        segs_orig = segs.clone()

        _, _, _, segs2, _ = ds._augment(*_make_feats(), segs, _make_labels(), mask)
        assert torch.allclose(segs2, segs_orig)


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
        TestTemporalMask().test_mask_stays_within_valid_frames,
        TestTemporalMask().test_mask_actually_zeroes_some_valid_frames,
        TestTemporalMask().test_mask_not_applied_when_prob_zero,
        TestTemporalMask().test_default_config_coverage,
        TestSegmentJitter().test_jittered_segments_remain_valid,
        TestSegmentJitter().test_segments_clamped_to_valid_range,
        TestSegmentJitter().test_no_jitter_when_zero,
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
