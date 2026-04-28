"""
Extracts the following modalities:
  Video : CLIP ViT-L/14 frame-level features    (768-dim per frame)
  Audio : wav2vec 2.0 Large features            (1024-dim per step, resampled to T)
  Text  : Sentence-wise HateBERT embeddings     (768-dim per sentence, padded to T)

All outputs are saved as .pt tensors to:
    data/HateMM/video_features/<video_id>.pt   — (T, 768)
    data/HateMM/audio_features/<video_id>.pt   — (T, 1024)
    data/HateMM/text_features/<video_id>.pt    — (T, 768)

Usage:
    python extract_features.py --video_dir /path/to/hateclipseg/videos \\
                                --out_dir   data/hateclipseg \\
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
    HAS_EXTRACTION_DEPS = True
except ImportError:
    HAS_EXTRACTION_DEPS = False


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
        """
        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(round(native_fps / fps)))

        all_feats = []
        pending   = [] # accumulate raw frame tensors for batching
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
            waveform = waveform.mean(0)   # stereo to mono
            if sr != _WAV2VEC_SR:
                waveform = torchaudio.functional.resample(
                    waveform, sr, _WAV2VEC_SR)
            return waveform
        except Exception as primary_err:
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

        # run wav2vec 2.0 in 30-second chunks to avoid OOM
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

        # linear interpolation to target_len
        feats = feats.unsqueeze(0).permute(0, 2, 1)   # (1, 1024, T_wav2vec)
        feats = F.interpolate(feats, size=target_len,
                              mode="linear", align_corners=False)
        feats = feats.permute(0, 2, 1).squeeze(0)     # (target_len, 1024)
        return feats.float().cpu()


class TextFeatureExtractor:
    """
    Implements the four-step sentence-wise text embedding:
      1. Whisper ASR with sentence timestamps
      3. HateBERT CLS encoding per sentence
      4. Repeat each embedding over its timestamp span
    """

    def __init__(self, device="cpu", whisper_model="base"):
        assert HAS_EXTRACTION_DEPS, "Install transformers, whisper, nltk first."
        self.device = device

        # Whisper for ASR
        self.asr = whisper.load_model(whisper_model, device=device)

        # HateBERT for sentence encoding
        self.tokenizer = BertTokenizer.from_pretrained("GroNLP/hateBERT")
        self.bert = BertModel.from_pretrained("GroNLP/hateBERT").to(device).eval()

    @torch.no_grad()
    def extract(self, video_path: str, target_len: int,
                fps: float = 1.0) -> torch.Tensor:
        """
        Returns (target_len, 768).
        """
        # Step 1: Whisper ASR
        result = self.asr.transcribe(video_path, word_timestamps=True)
        segments = result["segments"]   # each has 'start', 'end', 'text'

        if not segments:
            return torch.zeros(target_len, 768)

        # Step 2: sentence-wise splitting
        # We treat each Whisper segment as one "sentence"
        sentence_feats_by_frame = [None] * target_len
        duration_total = segments[-1]["end"]   # seconds

        for seg in segments:
            text  = seg["text"].strip()
            t_start = seg["start"]   # seconds
            t_end   = seg["end"]

            # Step 3: HateBERT encode
            tokens = self.tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=512,
                padding=True
            ).to(self.device)
            out = self.bert(**tokens)
            feat = out.last_hidden_state[:, 0, :].squeeze(0)  # CLS (768,)

            # Step 4: expand to frame range
            frame_start = int(t_start * fps)
            frame_end   = int(t_end   * fps)
            for f in range(frame_start, min(frame_end, target_len)):
                sentence_feats_by_frame[f] = feat.cpu()

        # fill frames with no sentence with zeros
        zero = torch.zeros(768)
        frames = [f if f is not None else zero
                  for f in sentence_feats_by_frame]
        return torch.stack(frames)   # (target_len, 768)


def extract_all(video_dir: str, out_dir: str, fps: float = 1.0,
                device: str = "cpu", overwrite: bool = False,
                video_batch_size: int = 16):
    """
    Extract features for all .mp4/.avi videos in video_dir and save to out_dir.

    Args:
        overwrite:         If False (default), skip videos whose .pt files
                           already exist.  Pass --overwrite to re-extract.
        video_batch_size:  Frames per CLIP ViT-L/14 forward pass.  Reduce if
                           you OOM on GPU during video extraction (default: 16).
    """
    if not HAS_EXTRACTION_DEPS:
        raise RuntimeError(
            "Missing extraction dependencies.  "
            "Install: transformers openai-whisper torchaudio nltk "
            "git+https://github.com/openai/CLIP.git"
        )

    vid_out   = os.path.join(out_dir, "video_features");  os.makedirs(vid_out, exist_ok=True)
    aud_out   = os.path.join(out_dir, "audio_features");  os.makedirs(aud_out, exist_ok=True)
    text_out  = os.path.join(out_dir, "text_features");   os.makedirs(text_out, exist_ok=True)

    vfe = VideoFeatureExtractor(device)
    afe = AudioFeatureExtractor(device)
    tfe = TextFeatureExtractor(device)

    video_files = sorted(Path(video_dir).glob("**/*.mp4")) + \
                  sorted(Path(video_dir).glob("**/*.avi"))

    failed  = []
    skipped = []
    for idx, vp in enumerate(video_files):
        vid_id = vp.stem

        v_path = os.path.join(vid_out,  f"{vid_id}.pt")
        a_path = os.path.join(aud_out,  f"{vid_id}.pt")
        t_path = os.path.join(text_out, f"{vid_id}.pt")
        all_exist = all(os.path.isfile(p) for p in (v_path, a_path, t_path))

        if all_exist and not overwrite:
            print(f"[{idx+1}/{len(video_files)}] Skipping (already exists): {vid_id}")
            skipped.append(vid_id)
            continue

        print(f"[{idx+1}/{len(video_files)}] Extracting: {vid_id}")

        try:
            # video
            v_feat = vfe.extract(str(vp), fps=fps,
                                   batch_size=video_batch_size)   # (T, 768)
            T = v_feat.shape[0]
            torch.save(v_feat, v_path)

            # audio
            a_feat = afe.extract(str(vp), target_len=T)       # (T, 1024)
            torch.save(a_feat, a_path)

            # text
            t_feat = tfe.extract(str(vp), target_len=T, fps=fps)  # (T, 768)
            torch.save(t_feat, t_path)

            print(f"  shapes — video: {tuple(v_feat.shape)}  "
                  f"audio: {tuple(a_feat.shape)}  "
                  f"text: {tuple(t_feat.shape)}")

        except Exception as e:
            print(f"  [SKIP] {vid_id} failed: {e.__class__.__name__}: {e}")
            failed.append((vid_id, str(e)))
            continue

    # summary
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
    args = parser.parse_args()
    extract_all(args.video_dir, args.out_dir, args.fps, args.device,
                overwrite=args.overwrite,
                video_batch_size=args.video_batch_size)