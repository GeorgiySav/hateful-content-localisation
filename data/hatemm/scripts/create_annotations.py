"""
Convert HateMM_annotation.csv → annotations.json (ActivityNet format)

Usage:
    python data/hatemm/scripts/create_annotations.py \
        --csv  data/hatemm/dataset/HateMM_annotation.csv \
        --videos data/hatemm/dataset/videos \
        --output data/hatemm/dataset/annotations.json \
        [--val_frac 0.15] [--test_frac 0.15] [--seed 42]

Output format (ActivityNet-style, as expected by libs/datasets/hatemm.py):
{
  "database": {
    "<video_stem>": {
      "duration":     <float seconds>,
      "subset":       "train" | "val" | "test",
      "annotations":  [{"segment": [start, end], "label": "hate"}, ...],
      "video_label":  "hate" | "non_hate"   (kept for weak-supervision fallback)
    },
    ...
  }
}

Duration source priority:
  1. ffprobe  (requires ffmpeg on PATH)
  2. OpenCV   (cv2)
  3. Falls back to duration inferred from last annotation end-time + 1 s
     (hate videos only; non-hate with no fallback are skipped with a warning)
"""

import argparse
import ast
import json
import os
import random
import subprocess
import sys
import csv
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hhmmss_to_seconds(t: str) -> float:
    """'HH:MM:SS' or 'MM:SS' → float seconds."""
    parts = t.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def duration_ffprobe(path: str) -> float | None:
    """Return duration via ffprobe, or None if unavailable."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        val = result.stdout.strip()
        if val:
            return float(val)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def duration_opencv(path: str) -> float | None:
    """Return duration via OpenCV, or None if unavailable."""
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0 and frames > 0:
            return frames / fps
    except ImportError:
        pass
    return None


def get_duration(video_path: str, fallback: float | None = None) -> float | None:
    dur = duration_ffprobe(video_path)
    if dur is not None:
        return dur
    dur = duration_opencv(video_path)
    if dur is not None:
        return dur
    return fallback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build annotations.json from HateMM CSV")
    p.add_argument("--csv",    default="data/hatemm/dataset/HateMM_annotation.csv")
    p.add_argument("--videos", default="data/hatemm/dataset/videos")
    p.add_argument("--output", default="data/hatemm/dataset/annotations.json")
    p.add_argument("--val_frac",  type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()


def build_split(indices: list[int], val_frac: float, test_frac: float,
                rng: random.Random) -> dict[int, str]:
    """Stratified-by-order random split → {idx: 'train'|'val'|'test'}."""
    shuffled = indices[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val  = max(1, round(n * val_frac))
    n_test = max(1, round(n * test_frac))
    split = {}
    for i, idx in enumerate(shuffled):
        if i < n_val:
            split[idx] = "val"
        elif i < n_val + n_test:
            split[idx] = "test"
        else:
            split[idx] = "train"
    return split


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    csv_path    = Path(args.csv)
    videos_dir  = Path(args.videos)
    output_path = Path(args.output)

    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    # ---- Read CSV ----
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Read {len(rows)} rows from {csv_path}")

    # ---- Parse each row ----
    hate_indices    = []
    non_hate_indices = []
    parsed = []  # list of dicts

    for i, row in enumerate(rows):
        filename   = row["video_file_name"].strip()
        label      = row["label"].strip()       # "Hate" / "Non Hate"
        snippet_raw = row.get("hate_snippet", "").strip()

        stem = Path(filename).stem              # drop .mp4
        video_path = str(videos_dir / filename)

        # Parse temporal segments
        segments = []
        if snippet_raw:
            try:
                raw_list = ast.literal_eval(snippet_raw)  # list of [start, end]
                for pair in raw_list:
                    start = hhmmss_to_seconds(pair[0])
                    end   = hhmmss_to_seconds(pair[1])
                    if end > start:
                        segments.append([start, end])
            except Exception as e:
                print(f"  WARNING: could not parse snippet for {filename}: {e}")

        parsed.append({
            "stem":       stem,
            "video_path": video_path,
            "is_hate":    label.lower() == "hate",
            "segments":   segments,
        })

        if label.lower() == "hate":
            hate_indices.append(i)
        else:
            non_hate_indices.append(i)

    # ---- Stratified split (hate and non-hate independently) ----
    split_map = {}
    split_map.update(build_split(hate_indices,     args.val_frac, args.test_frac, rng))
    split_map.update(build_split(non_hate_indices, args.val_frac, args.test_frac, rng))

    # ---- Build database ----
    database = {}
    skipped = 0
    missing_video = 0

    for i, p in enumerate(parsed):
        stem = p["stem"]
        vpath = p["video_path"]

        # Duration: fallback = last segment end + 1 s (hate only)
        fallback_dur = None
        if p["segments"]:
            fallback_dur = max(seg[1] for seg in p["segments"]) + 1.0

        if not Path(vpath).exists():
            # Try to still include if we have a fallback duration
            if fallback_dur is not None:
                dur = fallback_dur
                missing_video += 1
                print(f"  WARNING: video not found, using fallback duration: {vpath}")
            else:
                skipped += 1
                print(f"  WARNING: video not found, skipping: {vpath}")
                continue
        else:
            dur = get_duration(vpath, fallback=fallback_dur)
            if dur is None:
                skipped += 1
                print(f"  WARNING: could not determine duration, skipping: {vpath}")
                continue

        annotations = [
            {"segment": seg, "label": "hate"}
            for seg in p["segments"]
        ]

        entry = {
            "duration":    round(dur, 3),
            "subset":      split_map[i],
            "annotations": annotations,
            "video_label": "hate" if p["is_hate"] else "non_hate",
        }
        database[stem] = entry

    # ---- Write output ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"database": database}, f, indent=2)

    # ---- Summary ----
    subsets = {"train": 0, "val": 0, "test": 0}
    hate_cnt = {"train": 0, "val": 0, "test": 0}
    for entry in database.values():
        subsets[entry["subset"]] += 1
        if entry["video_label"] == "hate":
            hate_cnt[entry["subset"]] += 1

    print(f"\nWrote {len(database)} entries to {output_path}")
    print(f"  Skipped: {skipped}  (missing video + no duration fallback)")
    if missing_video:
        print(f"  Missing video (fallback duration used): {missing_video}")
    print(f"\n  {'subset':<8} {'total':>6} {'hate':>6} {'non-hate':>9}")
    print(f"  {'-'*35}")
    for s in ("train", "val", "test"):
        nh = subsets[s] - hate_cnt[s]
        print(f"  {s:<8} {subsets[s]:>6} {hate_cnt[s]:>6} {nh:>9}")


if __name__ == "__main__":
    main()
