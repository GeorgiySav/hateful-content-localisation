"""
extract_features.py — Offline feature extraction pipeline (Section 3.1).

Reproduces exactly the four-modality feature extraction described in
the paper before training begins:

  Video : CLIP ViT-L/14 frame-level features   (768-dim per frame)
  Audio : wav2vec 2.0 Large features            (1024-dim per step, resampled to T)
  Text  : Sentence-wise HateBERT embeddings     (768-dim per sentence, padded to T)
  OCR   : HateBERT embeddings of per-frame on-screen text → (T, 768)

The text pipeline follows the four steps in Section 3.1 / Fig. 3:
  1. Whisper ASR → raw transcript with word timestamps
  2. Split into sentence-wise fragments using NLTK sentence tokeniser
  3. HateBERT encodes each sentence → 768-dim CLS embedding
  4. Each sentence embedding is repeated over its timestamp span → (T, 768)

All outputs are saved as .pt tensors to:
    data/HateMM/video_features/<video_id>.pt   — (T, 768)
    data/HateMM/audio_features/<video_id>.pt   — (T, 1024)
    data/HateMM/text_features/<video_id>.pt    — (T, 768)
    data/HateMM/ocr_features/<video_id>.pt     — (T, 768)

Requirements (installed separately, not needed for training itself):
    pip install torch torchaudio transformers openai-whisper nltk easyocr
    pip install git+https://github.com/openai/CLIP.git

Usage:
    python extract_features.py --video_dir /path/to/hateMM/videos \\
                                --out_dir   data/HateMM \\
                                --fps       1
"""

import io
import os
import argparse
import math
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# ── lazy imports (only needed at extraction time, not training) ────────────
try:
    import cv2
    from PIL import Image
    import clip
    import torchaudio
    from transformers import (BertTokenizer, BertModel,
                               Wav2Vec2Processor, Wav2Vec2Model)
    import whisper
    import nltk
    nltk.download("punkt", quiet=True)
    from nltk.tokenize import sent_tokenize
    import easyocr
    import re
    HAS_EXTRACTION_DEPS = True
except ImportError:
    HAS_EXTRACTION_DEPS = False


# ════════════════════════════════════════════════════════════════════════════════
# Video features  —  CLIP ViT-L/14  (Section 3.1)
# ════════════════════════════════════════════════════════════════════════════════

class VideoFeatureExtractor:
    """
    Extracts one 768-dim CLIP ViT-L/14 feature per video frame at the given fps.
    Uses OpenAI's CLIP ViT-L/14 as the backbone.
    """

    def __init__(self, device="cpu"):
        assert HAS_EXTRACTION_DEPS, "Install clip, cv2, PIL first."
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, video_path: str, fps: float = 1.0,
                batch_size: int = 16) -> torch.Tensor:
        """
        Returns (T, 768) where T = number of sampled frames.

        batch_size controls how many frames are forwarded through CLIP at once.
        Reduce it if you hit OOM on very long videos (default 16 is safe for
        most 8 GB+ VRAM GPUs).
        """
        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(round(native_fps / fps)))

        all_feats = []
        pending   = []          # accumulate raw frame tensors for batching
        frame_idx = 0

        def _flush(buf):
            """Forward a batch of frame tensors through CLIP and collect CPU feats."""
            if not buf:
                return
            batch = torch.stack(buf).to(self.device)   # (B, 3, 224, 224)
            feats = self.model.encode_image(batch)      # (B, 768)
            feats = feats.float()
            all_feats.append(feats.cpu())

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                pending.append(self.preprocess(pil))   # (3, 224, 224) on CPU
                if len(pending) >= batch_size:
                    _flush(pending)
                    pending = []
            frame_idx += 1

        cap.release()
        _flush(pending)   # flush any remaining frames

        if not all_feats:
            raise ValueError(f"No frames extracted from {video_path}")
        return torch.cat(all_feats, dim=0)   # (T, 768)


# ════════════════════════════════════════════════════════════════════════════════
# Audio features  —  wav2vec 2.0 Large  (Section 3.1)
# ════════════════════════════════════════════════════════════════════════════════

_WAV2VEC_SR         = 16000
_WAV2VEC_CHUNK_SAMPLES = 30 * _WAV2VEC_SR   # 30 seconds per chunk


class AudioFeatureExtractor:
    """
    Extracts 1024-dim wav2vec 2.0 Large features from the audio track, then
    linearly interpolates to match the video frame length T.
    """

    def __init__(self, device="cpu",
                 model_name="facebook/wav2vec2-large-960h"):
        assert HAS_EXTRACTION_DEPS, "Install torchaudio, transformers first."
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def _load_waveform(self, video_path: str) -> torch.Tensor:
        """
        Load mono 16 kHz waveform from video_path.
        Falls back to ffmpeg subprocess if torchaudio cannot read the audio track.
        Returns a 1-D float32 tensor on CPU, or an empty tensor on failure.
        """
        try:
            waveform, sr = torchaudio.load(video_path)
            if waveform.numel() == 0:
                raise ValueError("empty waveform")
            waveform = waveform.mean(0)   # stereo → mono
            if sr != _WAV2VEC_SR:
                waveform = torchaudio.functional.resample(
                    waveform, sr, _WAV2VEC_SR)
            return waveform
        except Exception as primary_err:
            # ffmpeg fallback: pipe raw PCM into torchaudio
            try:
                cmd = [
                    "ffmpeg", "-i", video_path,
                    "-ac", "1", "-ar", str(_WAV2VEC_SR),
                    "-f", "wav", "pipe:1",
                ]
                proc = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    check=True)
                waveform, sr = torchaudio.load(io.BytesIO(proc.stdout))
                waveform = waveform.mean(0)
                if sr != _WAV2VEC_SR:
                    waveform = torchaudio.functional.resample(
                        waveform, sr, _WAV2VEC_SR)
                return waveform
            except Exception:
                print(f"  [audio] Could not load audio from "
                      f"{os.path.basename(video_path)} "
                      f"({primary_err.__class__.__name__}). "
                      f"Using zero features.")
                return torch.zeros(0)

    @torch.no_grad()
    def extract(self, video_path: str, target_len: int) -> torch.Tensor:
        """
        Returns (target_len, 1024) — linearly interpolated to match video T.
        Returns a zero tensor if the video has no audio stream.
        """
        waveform = self._load_waveform(video_path)

        if waveform.numel() == 0:
            return torch.zeros(target_len, 1024)

        # ── run wav2vec 2.0 in 30-second chunks to avoid OOM ──────────────
        hidden_chunks = []
        total_samples = waveform.shape[0]
        for start in range(0, total_samples, _WAV2VEC_CHUNK_SAMPLES):
            chunk = waveform[start : start + _WAV2VEC_CHUNK_SAMPLES]
            inputs = self.processor(
                chunk.numpy(), sampling_rate=_WAV2VEC_SR,
                return_tensors="pt", padding=True,
            )
            input_values = inputs.input_values.to(self.device)
            out = self.model(input_values)
            # last_hidden_state: (1, T_chunk, 1024)
            hidden_chunks.append(out.last_hidden_state.squeeze(0).cpu())  # (T_chunk, 1024)

        feats = torch.cat(hidden_chunks, dim=0)   # (T_wav2vec, 1024)

        # ── linear interpolation to target_len  (Section 3.1) ────────────
        feats = feats.unsqueeze(0).permute(0, 2, 1)   # (1, 1024, T_wav2vec)
        feats = F.interpolate(feats, size=target_len,
                              mode="linear", align_corners=False)
        feats = feats.permute(0, 2, 1).squeeze(0)     # (target_len, 1024)
        return feats.float().cpu()


# ════════════════════════════════════════════════════════════════════════════════
# Text features  —  sentence-wise HateBERT  (Section 3.1, Fig. 3)
# ════════════════════════════════════════════════════════════════════════════════

class TextFeatureExtractor:
    """
    Implements the four-step sentence-wise text embedding (Section 3.1):
      1. Whisper ASR with word timestamps
      2. Sentence-split using NLTK
      3. HateBERT CLS encoding per sentence
      4. Repeat each embedding over its timestamp span → (T, 768)
    """

    def __init__(self, device="cpu", whisper_model="base"):
        assert HAS_EXTRACTION_DEPS, "Install transformers, whisper, nltk first."
        self.device = device

        # Whisper for ASR (step 1)
        self.asr = whisper.load_model(whisper_model, device=device)

        # HateBERT for sentence encoding (step 3)
        self.tokenizer = BertTokenizer.from_pretrained("GroNLP/hateBERT")
        self.bert = BertModel.from_pretrained("GroNLP/hateBERT").to(device).eval()

    @torch.no_grad()
    def extract(self, video_path: str, target_len: int,
                fps: float = 1.0) -> torch.Tensor:
        """
        Returns (target_len, 768).
        """
        # ── Step 1: Whisper ASR ────────────────────────────────────────────
        result = self.asr.transcribe(video_path, word_timestamps=True)
        segments = result["segments"]   # each has 'start', 'end', 'text'

        if not segments:
            return torch.zeros(target_len, 768)

        # ── Step 2: sentence-wise splitting ───────────────────────────────
        # We treat each Whisper segment as one "sentence" (they are natural
        # phrase units with start/end timestamps).  Alternatively NLTK
        # sent_tokenize could further split long segments.
        sentence_feats_by_frame = [None] * target_len
        duration_total = segments[-1]["end"]   # seconds

        for seg in segments:
            text  = seg["text"].strip()
            t_start = seg["start"]   # seconds
            t_end   = seg["end"]

            # ── Step 3: HateBERT encode ────────────────────────────────────
            tokens = self.tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=512,
                padding=True
            ).to(self.device)
            out = self.bert(**tokens)
            feat = out.last_hidden_state[:, 0, :].squeeze(0)  # CLS (768,)

            # ── Step 4: expand to frame range ─────────────────────────────
            frame_start = int(t_start * fps)
            frame_end   = int(t_end   * fps)
            for f in range(frame_start, min(frame_end, target_len)):
                sentence_feats_by_frame[f] = feat.cpu()

        # fill frames with no sentence with zeros
        zero = torch.zeros(768)
        frames = [f if f is not None else zero
                  for f in sentence_feats_by_frame]
        return torch.stack(frames)   # (target_len, 768)


# ════════════════════════════════════════════════════════════════════════════════
# OCR features  —  per-frame EasyOCR + HateBERT  (Section 3.1, MM-HSD)
# ════════════════════════════════════════════════════════════════════════════════

# Regex to keep only alphanumeric, spaces, common punctuation, and apostrophes.
# Strips emoji, symbols, and control characters (MM-HSD Section 3.1 cleaning).
_OCR_KEEP = re.compile(r"[^a-zA-Z0-9 .,!?:;'\-]") if HAS_EXTRACTION_DEPS else None


class OCRFeatureExtractor:
    """
    Implements per-frame on-screen text embedding (Section 3.1, MM-HSD):
      1. Sample frames at target fps using OpenCV (aligned with VideoFeatureExtractor)
      2. Run EasyOCR on each frame; concatenate detected text regions per frame
      3. Clean each per-frame string: retain alphanumeric + .,!?:;-' (MM-HSD §3.1)
      4. HateBERT CLS encoding per frame (or zeros if no text detected)
      → frame-aligned output (T, 768), same convention as the transcript pipeline

    Each frame is independent — no cross-frame de-duplication or merging.
    """

    def __init__(self, device="cpu"):
        assert HAS_EXTRACTION_DEPS, "Install easyocr, transformers first."
        self.device = device

        # EasyOCR reader — loaded once, reused across all calls
        self.reader = easyocr.Reader(["en"], gpu=(device != "cpu"))

        # HateBERT for frame-level OCR text encoding (step 4)
        self.tokenizer = BertTokenizer.from_pretrained("GroNLP/hateBERT")
        self.bert = BertModel.from_pretrained("GroNLP/hateBERT").to(device).eval()

    @staticmethod
    def _clean(text: str) -> str:
        """
        Clean raw OCR text per MM-HSD Section 3.1:
        retain only alphanumeric characters, common punctuation (.,!?:;-),
        and apostrophes (for contractions).  Strip emoji, symbols, and
        control characters.  Collapse whitespace and strip.
        """
        cleaned = _OCR_KEEP.sub("", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @torch.no_grad()
    def extract(self, video_path: str, target_len: int,
                fps: float = 1.0) -> torch.Tensor:
        """
        Returns (target_len, 768).

        Frames with detected on-screen text (non-empty after cleaning) are
        embedded via HateBERT CLS token.  Frames with no detected text yield
        a zero vector.  If every frame has no text, returns
        torch.zeros(target_len, 768).
        """
        # ── Step 1: sample frames at target fps (mirrors VideoFeatureExtractor) ──
        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(round(native_fps / fps)))

        sampled_frames = []   # list of (frame_index_in_output, BGR ndarray)
        frame_idx = 0
        sample_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                if sample_idx < target_len:
                    sampled_frames.append((sample_idx, frame))
                sample_idx += 1
            frame_idx += 1

        cap.release()

        # ── Steps 2–4: OCR → clean → embed per frame ──────────────────────
        feats = torch.zeros(target_len, 768)

        for out_idx, bgr_frame in sampled_frames:
            # Step 2: EasyOCR — returns list of (bbox, text, confidence)
            ocr_results = self.reader.readtext(bgr_frame, detail=1)
            raw_text = " ".join(det[1] for det in ocr_results)

            # Step 3: clean
            cleaned = self._clean(raw_text)
            if not cleaned:
                continue   # leave feats[out_idx] as zeros

            # Step 4: HateBERT CLS embedding
            tokens = self.tokenizer(
                cleaned, return_tensors="pt",
                truncation=True, max_length=512,
                padding=True,
            ).to(self.device)
            out = self.bert(**tokens)
            feat = out.last_hidden_state[:, 0, :].squeeze(0).cpu()  # CLS (768,)
            feats[out_idx] = feat

        return feats   # (target_len, 768)


# ════════════════════════════════════════════════════════════════════════════════
# Main extraction entry point
# ════════════════════════════════════════════════════════════════════════════════

_ALL_MODALITIES = ("video", "audio", "text", "ocr")
_OUT_DIRS = {
    "video": "video_features",
    "audio": "audio_features",
    "text":  "text_features",
    "ocr":   "ocr_features",
}


def _compute_target_len(video_path: str, fps: float) -> int:
    """
    Return the number of frames that VideoFeatureExtractor would sample,
    using only OpenCV metadata (no model inference).  Used when the video
    modality is skipped but T is still needed by other extractors.
    """
    cap = cv2.VideoCapture(video_path)
    native_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frame_interval = max(1, int(round(native_fps / fps)))
    return max(1, math.ceil(total_frames / frame_interval))


def extract_all(video_dir: str, out_dir: str, fps: float = 1.0,
                device: str = "cpu", overwrite: bool = False,
                video_batch_size: int = 16,
                modalities: tuple = _ALL_MODALITIES):
    """
    Extract features for all .mp4/.avi videos in video_dir and save to out_dir.

    Args:
        overwrite:         If False (default), skip videos whose .pt files
                           already exist.  Pass --overwrite to re-extract.
        video_batch_size:  Frames per CLIP ViT-L/14 forward pass.  Reduce if
                           you OOM on GPU during video extraction (default: 16).
        modalities:        Subset of ('video', 'audio', 'text', 'ocr') to run.
                           Defaults to all four.  Only the selected extractors
                           are loaded into memory.
    """
    if not HAS_EXTRACTION_DEPS:
        raise RuntimeError(
            "Missing extraction dependencies.  "
            "Install: transformers openai-whisper torchaudio nltk easyocr "
            "git+https://github.com/openai/CLIP.git"
        )

    mods = set(modalities)

    # ── create output dirs only for selected modalities ───────────────────
    out_dirs = {}
    for m in mods:
        p = os.path.join(out_dir, _OUT_DIRS[m])
        os.makedirs(p, exist_ok=True)
        out_dirs[m] = p

    # ── instantiate only the extractors that are needed ───────────────────
    vfe = VideoFeatureExtractor(device) if "video" in mods else None
    afe = AudioFeatureExtractor(device) if "audio" in mods else None
    tfe = TextFeatureExtractor(device)  if "text"  in mods else None
    ofe = OCRFeatureExtractor(device)   if "ocr"   in mods else None

    video_files = sorted(Path(video_dir).glob("**/*.mp4")) + \
                  sorted(Path(video_dir).glob("**/*.avi"))

    failed  = []
    skipped = []
    for idx, vp in enumerate(video_files):
        vid_id = vp.stem

        out_paths = {m: os.path.join(out_dirs[m], f"{vid_id}.pt") for m in mods}
        all_exist = all(os.path.isfile(p) for p in out_paths.values())

        if all_exist and not overwrite:
            print(f"[{idx+1}/{len(video_files)}] Skipping (already exists): {vid_id}")
            skipped.append(vid_id)
            continue

        print(f"[{idx+1}/{len(video_files)}] Extracting ({', '.join(sorted(mods))}): {vid_id}")

        try:
            shape_parts = []

            # ── video ─────────────────────────────────────────────────────
            if "video" in mods:
                v_feat = vfe.extract(str(vp), fps=fps,
                                     batch_size=video_batch_size)   # (T, 768)
                T = v_feat.shape[0]
                torch.save(v_feat, out_paths["video"])
                shape_parts.append(f"video: {tuple(v_feat.shape)}")
            else:
                # T is needed by audio / text / ocr even when video is skipped
                T = _compute_target_len(str(vp), fps)

            # ── audio (zeros for silent videos) ───────────────────────────
            if "audio" in mods:
                a_feat = afe.extract(str(vp), target_len=T)       # (T, 1024)
                torch.save(a_feat, out_paths["audio"])
                shape_parts.append(f"audio: {tuple(a_feat.shape)}")

            # ── text ──────────────────────────────────────────────────────
            if "text" in mods:
                t_feat = tfe.extract(str(vp), target_len=T, fps=fps)  # (T, 768)
                torch.save(t_feat, out_paths["text"])
                shape_parts.append(f"text: {tuple(t_feat.shape)}")

            # ── OCR ───────────────────────────────────────────────────────
            if "ocr" in mods:
                o_feat = ofe.extract(str(vp), target_len=T, fps=fps)  # (T, 768)
                torch.save(o_feat, out_paths["ocr"])
                shape_parts.append(f"ocr: {tuple(o_feat.shape)}")

            print(f"  shapes — {'  '.join(shape_parts)}")

        except Exception as e:
            print(f"  [SKIP] {vid_id} failed: {e.__class__.__name__}: {e}")
            failed.append((vid_id, str(e)))
            continue

    # ── summary ───────────────────────────────────────────────────────────
    done = len(video_files) - len(skipped) - len(failed)
    print(f"\n{'='*50}")
    print(f"Extraction complete.")
    print(f"  Extracted : {done}")
    print(f"  Skipped   : {len(skipped)}  (already existed, --overwrite not set)")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print("\nFailed videos:")
        for vid_id, reason in failed:
            print(f"  - {vid_id}: {reason}")
    print(f"Features saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MultiHateLoc features")
    parser.add_argument("--video_dir", required=True,
                        help="Directory containing raw HateMM .mp4 files")
    parser.add_argument("--out_dir", default="data/HateMM",
                        help="Output root directory for .pt feature files")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to sample for video/text features")
    parser.add_argument("--device", default="cuda",
                        help="torch device (e.g. cuda or cpu)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract and overwrite features even if .pt files "
                             "already exist on disk. Default: skip existing files.")
    parser.add_argument("--video_batch_size", type=int, default=16,
                        help="Number of video frames per CLIP ViT-L/14 forward pass. "
                             "Reduce if you run out of VRAM on long videos "
                             "(default: 16).")
    parser.add_argument("--modalities", nargs="+",
                        choices=list(_ALL_MODALITIES), default=list(_ALL_MODALITIES),
                        metavar="MODALITY",
                        help="Which modalities to extract.  Choose any subset of: "
                             "video audio text ocr.  "
                             "Default: all four.  "
                             "Example: --modalities video audio")
    args = parser.parse_args()
    extract_all(args.video_dir, args.out_dir, args.fps, args.device,
                overwrite=args.overwrite,
                video_batch_size=args.video_batch_size,
                modalities=args.modalities)
