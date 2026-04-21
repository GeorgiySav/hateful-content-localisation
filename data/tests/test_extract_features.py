"""
Tests for data/extract_features.py — one test class per pipeline stage.

Unit tests bypass heavy model loading by constructing extractor instances via
__new__ and injecting lightweight mocks as instance attributes.  This avoids
downloading / loading CLIP, wav2vec, Whisper or HateBERT during normal CI.

Integration tests (marked @pytest.mark.integration) load real models and run
on bit_0EHvMSiEHVoc.mp4 from the HateCliPSeg dataset.  They are skipped
automatically when HAS_EXTRACTION_DEPS is False.

Run unit tests only:
    cd data
    python -m pytest tests/test_extract_features.py -v -m "not integration"

Run integration tests:
    cd data
    python -m pytest tests/test_extract_features.py -v -m integration
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ── make data/ importable ─────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

import extract_features
from extract_features import (
    AudioFeatureExtractor,
    TextFeatureExtractor,
    VideoFeatureExtractor,
    extract_all,
    HAS_EXTRACTION_DEPS,
    _WAV2VEC_SR,
)

# Real video used for integration tests
_VIDEO = str(
    _DATA_DIR / "hateclipseg" / "dataset" / "videos" / "bit_0EHvMSiEHVoc.mp4"
)
_FPS = 1.0

# ── shared pytest mark ────────────────────────────────────────────────────────
integration = pytest.mark.skipif(
    not HAS_EXTRACTION_DEPS or not os.path.isfile(_VIDEO),
    reason="Extraction deps not installed or test video not found",
)


# ══════════════════════════════════════════════════════════════════════════════
# Mock helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cap_mock(n_frames: int, native_fps: float = 1.0):
    """Return a cv2.VideoCapture mock that streams n_frames random BGR frames."""
    cap = MagicMock()
    cap.get.return_value = native_fps
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(n_frames)]
    cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
    return cap


def _clip_pair():
    """Lightweight (model, preprocess) pair mimicking clip.load()."""
    model = MagicMock()
    model.encode_image.side_effect = lambda b: torch.randn(b.shape[0], 768)
    preprocess = MagicMock(return_value=torch.zeros(3, 224, 224))
    return model, preprocess


def _wav2vec_pair(n_hidden: int = 40):
    """(processor_instance, model_instance) mocks for wav2vec 2.0 Large."""
    proc = MagicMock()
    proc.return_value = MagicMock(input_values=torch.zeros(1, _WAV2VEC_SR))

    mdl = MagicMock()
    mdl.eval.return_value = mdl
    mdl.to.return_value = mdl
    out = MagicMock()
    out.last_hidden_state = torch.zeros(1, n_hidden, 1024)
    mdl.return_value = out
    return proc, mdl


def _bert_pair():
    """(tokenizer_instance, bert_instance) mocks for HateBERT."""
    # tokenizer returns something whose .to() yields a plain dict for **-unpacking
    tok_dict = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    encoding = MagicMock()
    encoding.to.return_value = tok_dict

    tok = MagicMock(return_value=encoding)

    bert = MagicMock()
    bert.eval.return_value = bert
    bert.to.return_value = bert
    bert_out = MagicMock()
    bert_out.last_hidden_state = torch.ones(1, 5, 768)
    bert.return_value = bert_out
    return tok, bert


def _whisper_mock(segments=None):
    if segments is None:
        segments = [
            {"start": 0.0, "end": 3.0, "text": "Hello world"},
            {"start": 3.0, "end": 7.0, "text": "More speech here"},
        ]
    asr = MagicMock()
    asr.transcribe.return_value = {"segments": segments}
    return asr


# ── context manager: patch cv2 + PIL.Image for VideoFeatureExtractor.extract ─

class _MockCv2:
    """Patches extract_features.cv2 and extract_features.Image together."""
    def __init__(self, cap):
        self._cap = cap
        self._p_cv2 = patch.object(extract_features, "cv2", create=True)
        self._p_img = patch.object(extract_features, "Image", create=True)

    def __enter__(self):
        mock_cv2 = self._p_cv2.__enter__()
        mock_img = self._p_img.__enter__()
        mock_cv2.VideoCapture.return_value = self._cap
        mock_cv2.CAP_PROP_FPS = 5          # value passed to cap.get(); ignored by mock
        mock_cv2.COLOR_BGR2RGB = 4
        mock_cv2.cvtColor.side_effect = lambda f, _: f   # BGR→RGB is a no-op here
        mock_img.fromarray.return_value = MagicMock()
        return mock_cv2, mock_img

    def __exit__(self, *args):
        self._p_img.__exit__(*args)
        self._p_cv2.__exit__(*args)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — VideoFeatureExtractor
# ══════════════════════════════════════════════════════════════════════════════

class TestVideoFeatureExtractor:

    def _extractor(self):
        model, preprocess = _clip_pair()
        ext = VideoFeatureExtractor.__new__(VideoFeatureExtractor)
        ext.device = "cpu"
        ext.model = model
        ext.preprocess = preprocess
        return ext

    def test_output_shape(self):
        """extract() returns (T, 768) where T equals the number of sampled frames."""
        ext = self._extractor()
        with _MockCv2(_cap_mock(n_frames=10, native_fps=1.0)):
            feats = ext.extract("fake.mp4", fps=1.0)
        assert feats.ndim == 2
        assert feats.shape == (10, 768)

    def test_frame_interval_subsamples(self):
        """At fps < native_fps, only every nth frame is sampled."""
        ext = self._extractor()
        # native=4 fps, extract at 2 fps → frame_interval=2 → T = ceil(4/2) = 2
        with _MockCv2(_cap_mock(n_frames=4, native_fps=4.0)):
            feats = ext.extract("fake.mp4", fps=2.0)
        assert feats.shape == (2, 768)

    def test_full_fps_samples_all_frames(self):
        """At fps == native_fps, every frame is included."""
        ext = self._extractor()
        with _MockCv2(_cap_mock(n_frames=8, native_fps=8.0)):
            feats = ext.extract("fake.mp4", fps=8.0)
        assert feats.shape[0] == 8

    def test_empty_video_raises(self):
        """extract() raises ValueError when the video contains no frames."""
        ext = self._extractor()
        empty_cap = MagicMock()
        empty_cap.get.return_value = 25.0
        empty_cap.read.side_effect = [(False, None)]
        with _MockCv2(empty_cap):
            with pytest.raises(ValueError, match="No frames"):
                ext.extract("empty.mp4")

    def test_batch_processing_shape(self):
        """Flushing in batches of batch_size still gives correct (T, 768) output."""
        ext = self._extractor()
        # 9 frames at native fps == target fps → T=9; 3 full batches of 3
        with _MockCv2(_cap_mock(n_frames=9, native_fps=1.0)):
            feats = ext.extract("fake.mp4", fps=1.0, batch_size=3)
        assert feats.shape == (9, 768)

    def test_output_dtype_and_device(self):
        """Output tensor is float32 on CPU."""
        ext = self._extractor()
        with _MockCv2(_cap_mock(n_frames=5, native_fps=1.0)):
            feats = ext.extract("fake.mp4", fps=1.0)
        assert feats.dtype == torch.float32
        assert feats.device.type == "cpu"

    @integration
    def test_real_video_shape(self):
        """Integration: CLIP ViT-L/14 features from bit_0EHvMSiEHVoc.mp4."""
        ext = VideoFeatureExtractor(device="cpu")
        feats = ext.extract(_VIDEO, fps=_FPS)
        assert feats.ndim == 2
        assert feats.shape[1] == 768
        assert feats.shape[0] > 0

    @integration
    def test_real_video_no_nan_or_inf(self):
        """Integration: features must be finite."""
        ext = VideoFeatureExtractor(device="cpu")
        feats = ext.extract(_VIDEO, fps=_FPS)
        assert not torch.isnan(feats).any()
        assert not torch.isinf(feats).any()


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — AudioFeatureExtractor
# ══════════════════════════════════════════════════════════════════════════════

class TestAudioFeatureExtractor:

    def _extractor(self, n_hidden=40):
        proc, mdl = _wav2vec_pair(n_hidden=n_hidden)
        ext = AudioFeatureExtractor.__new__(AudioFeatureExtractor)
        ext.device = "cpu"
        ext.processor = proc
        ext.model = mdl
        return ext

    def test_output_shape(self):
        """extract() returns exactly (target_len, 1024)."""
        ext = self._extractor()
        waveform = torch.zeros(_WAV2VEC_SR)   # 1 second of silence
        with patch.object(extract_features, "torchaudio", create=True) as mta:
            mta.load.return_value = (waveform.unsqueeze(0), _WAV2VEC_SR)
            mta.functional.resample.side_effect = lambda w, *_: w
            feats = ext.extract("fake.mp4", target_len=20)
        assert feats.shape == (20, 1024)

    def test_zero_fallback_when_silent(self):
        """Returns an all-zero (target_len, 1024) tensor for silent/audio-free video."""
        ext = self._extractor()
        ext._load_waveform = MagicMock(return_value=torch.zeros(0))
        feats = ext.extract("silent.mp4", target_len=15)
        assert feats.shape == (15, 1024)
        assert feats.sum().item() == 0.0

    def test_target_len_always_matches(self):
        """Interpolation produces exactly target_len regardless of waveform length."""
        ext = self._extractor()
        for target in (5, 17, 100):
            waveform = torch.zeros(target * _WAV2VEC_SR)
            with patch.object(extract_features, "torchaudio", create=True) as mta:
                mta.load.return_value = (waveform.unsqueeze(0), _WAV2VEC_SR)
                mta.functional.resample.side_effect = lambda w, *_: w
                feats = ext.extract("fake.mp4", target_len=target)
            assert feats.shape == (target, 1024), f"failed for target_len={target}"

    def test_load_waveform_resamples_non_16k(self):
        """_load_waveform calls torchaudio.functional.resample when sr != 16 kHz."""
        ext = self._extractor()
        waveform_44k = torch.zeros(44100)
        with patch.object(extract_features, "torchaudio", create=True) as mta:
            mta.load.return_value = (waveform_44k.unsqueeze(0), 44100)
            mta.functional.resample.return_value = torch.zeros(_WAV2VEC_SR)
            ext._load_waveform("fake.mp4")
            mta.functional.resample.assert_called_once()
            call_args = mta.functional.resample.call_args
            assert torch.equal(call_args.args[0], waveform_44k)
            assert call_args.args[1] == 44100
            assert call_args.args[2] == _WAV2VEC_SR

    def test_load_waveform_no_resample_at_16k(self):
        """_load_waveform does NOT resample when audio is already 16 kHz."""
        ext = self._extractor()
        waveform = torch.zeros(_WAV2VEC_SR)
        with patch.object(extract_features, "torchaudio", create=True) as mta:
            mta.load.return_value = (waveform.unsqueeze(0), _WAV2VEC_SR)
            ext._load_waveform("fake.mp4")
            mta.functional.resample.assert_not_called()

    def test_output_dtype(self):
        """Audio features are float32."""
        ext = self._extractor()
        waveform = torch.zeros(_WAV2VEC_SR)
        with patch.object(extract_features, "torchaudio", create=True) as mta:
            mta.load.return_value = (waveform.unsqueeze(0), _WAV2VEC_SR)
            mta.functional.resample.side_effect = lambda w, *_: w
            feats = ext.extract("fake.mp4", target_len=10)
        assert feats.dtype == torch.float32

    @integration
    def test_real_video_shape(self):
        """Integration: wav2vec 2.0 features from bit_0EHvMSiEHVoc.mp4."""
        import cv2
        cap = cv2.VideoCapture(_VIDEO)
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        target = max(1, round(total_frames / native_fps * _FPS))

        ext = AudioFeatureExtractor(device="cpu")
        feats = ext.extract(_VIDEO, target_len=target)
        assert feats.shape == (target, 1024)
        assert feats.dtype == torch.float32


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — TextFeatureExtractor
# ══════════════════════════════════════════════════════════════════════════════

class TestTextFeatureExtractor:

    def _extractor(self, segments=None):
        tok, bert = _bert_pair()
        ext = TextFeatureExtractor.__new__(TextFeatureExtractor)
        ext.device = "cpu"
        ext.asr = _whisper_mock(segments)
        ext.tokenizer = tok
        ext.bert = bert
        return ext

    def test_output_shape(self):
        """extract() returns (target_len, 768)."""
        ext = self._extractor()
        feats = ext.extract("fake.mp4", target_len=10, fps=1.0)
        assert feats.shape == (10, 768)

    def test_empty_transcript_returns_zeros(self):
        """All-zero output when Whisper finds no speech segments."""
        ext = self._extractor(segments=[])
        feats = ext.extract("silent.mp4", target_len=8, fps=1.0)
        assert feats.shape == (8, 768)
        assert feats.sum().item() == 0.0

    def test_frame_expansion(self):
        """Sentence embedding is repeated over its timestamp span; other frames are zero."""
        # Segment covers seconds [2, 5) → frames 2, 3, 4 at fps=1.0
        segments = [{"start": 2.0, "end": 5.0, "text": "hello"}]
        ext = self._extractor(segments=segments)

        # Override bert to return a known non-zero feature
        tok_dict = {
            "input_ids": torch.zeros(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        encoding = MagicMock()
        encoding.to.return_value = tok_dict
        ext.tokenizer = MagicMock(return_value=encoding)

        bert_out = MagicMock()
        bert_out.last_hidden_state = torch.ones(1, 3, 768) * 2.0
        ext.bert = MagicMock(return_value=bert_out)

        feats = ext.extract("fake.mp4", target_len=8, fps=1.0)

        # frames 0-1: no segment → zeros
        assert feats[:2].sum().item() == 0.0, "frames before segment must be zero"
        # frames 2-4: covered by segment → non-zero
        assert feats[2:5].sum().item() != 0.0, "frames within segment must be non-zero"
        # frames 5-7: after segment → zeros
        assert feats[5:].sum().item() == 0.0, "frames after segment must be zero"

    def test_target_len_respected(self):
        """Output shape matches target_len for various values."""
        ext = self._extractor()
        for target in (3, 13, 50):
            feats = ext.extract("fake.mp4", target_len=target, fps=1.0)
            assert feats.shape == (target, 768), f"failed for target_len={target}"

    def test_segment_beyond_target_len_clipped(self):
        """Segments that extend past target_len do not raise IndexError."""
        segments = [{"start": 0.0, "end": 100.0, "text": "very long"}]
        ext = self._extractor(segments=segments)
        feats = ext.extract("fake.mp4", target_len=5, fps=1.0)
        assert feats.shape == (5, 768)

    def test_output_dtype(self):
        """Text features are float32."""
        ext = self._extractor()
        feats = ext.extract("fake.mp4", target_len=10, fps=1.0)
        assert feats.dtype == torch.float32

    @integration
    def test_real_video_shape(self):
        """Integration: HateBERT features from bit_0EHvMSiEHVoc.mp4."""
        import shutil
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not in PATH (required by Whisper for audio decoding)")
        import cv2
        cap = cv2.VideoCapture(_VIDEO)
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        target = max(1, round(total_frames / native_fps * _FPS))

        ext = TextFeatureExtractor(device="cpu", whisper_model="base")
        feats = ext.extract(_VIDEO, target_len=target, fps=_FPS)
        assert feats.shape == (target, 768)
        assert feats.dtype == torch.float32


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline — extract_all
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractAll:

    def _run(self, video_dir, out_dir, T=10, overwrite=False):
        """
        Run extract_all with all three extractor classes replaced by mocks.
        Returns the feature tensors that were saved.
        """
        v_feat = torch.randn(T, 768)
        a_feat = torch.randn(T, 1024)
        t_feat = torch.randn(T, 768)

        mock_vfe = MagicMock()
        mock_vfe.extract.return_value = v_feat
        mock_afe = MagicMock()
        mock_afe.extract.return_value = a_feat
        mock_tfe = MagicMock()
        mock_tfe.extract.return_value = t_feat

        with (
            patch.object(extract_features, "HAS_EXTRACTION_DEPS", True),
            patch.object(extract_features, "VideoFeatureExtractor",
                         return_value=mock_vfe) as VFE,
            patch.object(extract_features, "AudioFeatureExtractor",
                         return_value=mock_afe) as AFE,
            patch.object(extract_features, "TextFeatureExtractor",
                         return_value=mock_tfe) as TFE,
        ):
            extract_all(str(video_dir), str(out_dir), fps=_FPS,
                        device="cpu", overwrite=overwrite)
            return v_feat, a_feat, t_feat, VFE, AFE, TFE

    def _video_file(self, video_dir, name="clip_abc.mp4"):
        p = Path(video_dir) / name
        p.write_bytes(b"")   # stub; glob just needs the extension
        return p

    def test_creates_all_three_feature_files(self, tmp_path):
        """extract_all writes .pt files for video, audio, and text."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        self._video_file(video_dir)
        self._run(video_dir, tmp_path)
        assert (tmp_path / "video_features" / "clip_abc.pt").exists()
        assert (tmp_path / "audio_features" / "clip_abc.pt").exists()
        assert (tmp_path / "text_features"  / "clip_abc.pt").exists()

    def test_saved_tensors_have_correct_shapes(self, tmp_path):
        """Saved .pt files contain tensors with the expected dimensions."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        self._video_file(video_dir)
        T = 7
        self._run(video_dir, tmp_path, T=T)

        v = torch.load(tmp_path / "video_features" / "clip_abc.pt",
                       weights_only=True)
        a = torch.load(tmp_path / "audio_features" / "clip_abc.pt",
                       weights_only=True)
        t = torch.load(tmp_path / "text_features"  / "clip_abc.pt",
                       weights_only=True)
        assert v.shape == (T, 768)
        assert a.shape == (T, 1024)
        assert t.shape == (T, 768)

    def test_temporal_dims_are_consistent(self, tmp_path):
        """All three modalities share the same temporal dimension T."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        self._video_file(video_dir)
        T = 13
        self._run(video_dir, tmp_path, T=T)

        v = torch.load(tmp_path / "video_features" / "clip_abc.pt",
                       weights_only=True)
        a = torch.load(tmp_path / "audio_features" / "clip_abc.pt",
                       weights_only=True)
        t = torch.load(tmp_path / "text_features"  / "clip_abc.pt",
                       weights_only=True)
        assert v.shape[0] == a.shape[0] == t.shape[0] == T

    def test_skip_when_files_already_exist(self, tmp_path):
        """Videos with existing .pt files are skipped when overwrite=False."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        self._video_file(video_dir)

        # pre-create all three feature files
        for sub, dim in [("video_features", 768), ("audio_features", 1024),
                          ("text_features", 768)]:
            (tmp_path / sub).mkdir(parents=True, exist_ok=True)
            torch.save(torch.zeros(5, dim), tmp_path / sub / "clip_abc.pt")

        v_feat = torch.randn(10, 768)
        mock_vfe = MagicMock()
        mock_vfe.extract.return_value = v_feat

        with (
            patch.object(extract_features, "HAS_EXTRACTION_DEPS", True),
            patch.object(extract_features, "VideoFeatureExtractor",
                         return_value=mock_vfe) as VFE,
            patch.object(extract_features, "AudioFeatureExtractor",
                         return_value=MagicMock()),
            patch.object(extract_features, "TextFeatureExtractor",
                         return_value=MagicMock()),
        ):
            extract_all(str(video_dir), str(tmp_path), fps=_FPS,
                        device="cpu", overwrite=False)
            VFE.return_value.extract.assert_not_called()

    def test_overwrite_replaces_existing_files(self, tmp_path):
        """Pre-existing .pt files are replaced when overwrite=True."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        self._video_file(video_dir)

        for sub, dim in [("video_features", 768), ("audio_features", 1024),
                          ("text_features", 768)]:
            (tmp_path / sub).mkdir(parents=True, exist_ok=True)
            torch.save(torch.zeros(5, dim), tmp_path / sub / "clip_abc.pt")

        T_new = 9
        self._run(video_dir, tmp_path, T=T_new, overwrite=True)

        v = torch.load(tmp_path / "video_features" / "clip_abc.pt",
                       weights_only=True)
        assert v.shape[0] == T_new, "file should be overwritten with new T"

    def test_multiple_videos_all_processed(self, tmp_path):
        """extract_all processes every .mp4 in the directory."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        names = ["alpha.mp4", "beta.mp4", "gamma.mp4"]
        for n in names:
            self._video_file(video_dir, name=n)

        self._run(video_dir, tmp_path)

        for stem in ("alpha", "beta", "gamma"):
            assert (tmp_path / "video_features" / f"{stem}.pt").exists()
            assert (tmp_path / "audio_features" / f"{stem}.pt").exists()
            assert (tmp_path / "text_features"  / f"{stem}.pt").exists()

    @integration
    def test_real_video_full_pipeline(self, tmp_path):
        """Integration: extract_all on bit_0EHvMSiEHVoc.mp4 produces correct shapes."""
        import shutil
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not in PATH (required by Whisper)")
        video_dir = Path(_VIDEO).parent
        extract_all(str(video_dir), str(tmp_path), fps=_FPS, device="cpu")
        vid_id = Path(_VIDEO).stem
        v = torch.load(tmp_path / "video_features" / f"{vid_id}.pt",
                       weights_only=True)
        a = torch.load(tmp_path / "audio_features" / f"{vid_id}.pt",
                       weights_only=True)
        t = torch.load(tmp_path / "text_features"  / f"{vid_id}.pt",
                       weights_only=True)
        assert v.shape[1] == 768
        assert a.shape[1] == 1024
        assert t.shape[1] == 768
        assert v.shape[0] == a.shape[0] == t.shape[0], \
            "all modalities must share the same temporal dimension T"
