"""
Convert MultiHateClip TSV annotation files → annotations.json (ActivityNet format)

The MultiHateClip dataset already provides train/valid/test splits as separate TSV
files.  This script merges them and writes a single annotations.json that is
directly consumable by MultiHateClipDataset (and HateMMDataset).

Usage:
    python data/multihateclip/scripts/create_annotations.py \
        --tsv_dir   data/multihateclip/dataset/annotation \
        --videos    data/multihateclip/dataset/videos \
        --output    data/multihateclip/dataset/annotations.json \
        [--include_offensive]   # treat Offensive as hate (default: only Hateful)

Output format (ActivityNet-style, as expected by libs/datasets/multihateclip.py):
{
  "database": {
    "<video_id>": {
      "duration":     <float seconds>,
      "subset":       "train" | "val" | "test",
      "annotations":  [{"segment": [start, end], "label": "hate"}, ...],
      "video_label":  "hate" | "non_hate"
    },
    ...
  }
}

Label mapping:
  - "Hateful"  → video_label="hate",     annotations = temporal segments from TSV
  - "Offensive"→ video_label="hate"      (only when --include_offensive is set)
                  OR video_label="non_hate" and no annotations (default)
  - "Normal"   → video_label="non_hate", annotations = []

Duration source priority:
  1. ffprobe  (requires ffmpeg on PATH)
  2. OpenCV   (cv2)
  3. Falls back to last segment end-time + 1 s  (for hate videos)
  4. Videos with no duration and no segments are skipped with a warning.
"""

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Duration helpers  (identical to data/hatemm/scripts/create_annotations.py)
# ---------------------------------------------------------------------------

def duration_ffprobe(path: str) -> float | None:
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
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        fps    = cap.get(cv2.CAP_PROP_FPS)
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
# TSV parsing
# ---------------------------------------------------------------------------

def parse_components(component_cell: str) -> list[str]:
    """
    Parse the 'Component' column of the MultiHateClip TSV.

    The cell contains a Python-literal list of strings, e.g.
    "['Transcript', 'Audio', 'Metadata']" or "[]".
    Returns a list of lowercase component names.
    """
    component_cell = component_cell.strip()
    if not component_cell or component_cell == "[]":
        return []
    try:
        raw = ast.literal_eval(component_cell)
        return [c.strip().lower() for c in raw]
    except Exception as e:
        print(f"  WARNING: could not parse Component cell '{component_cell}': {e}")
        return []


def parse_segments(duration_cell: str) -> list[list[float]]:
    """
    Parse the 'Duration' column of the MultiHateClip TSV.

    The cell contains a Python-literal list of (start, end) tuples in seconds,
    e.g.  "[(30, 39)]"  or  "[]".  Returns a list of [start, end] float pairs.
    """
    duration_cell = duration_cell.strip()
    if not duration_cell or duration_cell == "[]":
        return []
    try:
        raw = ast.literal_eval(duration_cell)  # list of tuples
        segs = []
        for pair in raw:
            start, end = float(pair[0]), float(pair[1])
            if end > start:
                segs.append([start, end])
        return segs
    except Exception as e:
        print(f"  WARNING: could not parse Duration cell '{duration_cell}': {e}")
        return []


def read_tsv(path: Path) -> list[dict]:
    """Return rows as list of dicts (keyed by header names)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        header = None
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if header is None:
                header = cols
                continue
            rows.append(dict(zip(header, cols)))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build annotations.json from MultiHateClip TSV files"
    )
    p.add_argument(
        "--tsv_dir",
        default="data/multihateclip/dataset/annotation",
        help="Directory containing train.tsv, valid.tsv, test.tsv",
    )
    p.add_argument(
        "--videos",
        default="data/multihateclip/dataset/videos",
        help="Directory containing downloaded .mp4 files",
    )
    p.add_argument(
        "--output",
        default="data/multihateclip/dataset/annotations.json",
        help="Output path for annotations.json",
    )
    p.add_argument(
        "--include_offensive",
        action="store_true",
        default=False,
        help=(
            "Treat 'Offensive' videos as hate (include their temporal segments). "
            "By default only 'Hateful' videos contribute hate annotations."
        ),
    )
    return p.parse_args()


# TSV file name → ActivityNet subset name
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test":  "test",
}


def main():
    args = parse_args()

    tsv_dir    = Path(args.tsv_dir)
    videos_dir = Path(args.videos)
    output     = Path(args.output)

    database: dict = {}
    skipped = 0

    for tsv_name, subset in SPLIT_MAP.items():
        tsv_path = tsv_dir / f"{tsv_name}.tsv"
        if not tsv_path.exists():
            print(f"WARNING: TSV not found, skipping split '{tsv_name}': {tsv_path}")
            continue

        rows = read_tsv(tsv_path)
        print(f"Read {len(rows)} rows from {tsv_path}")

        for row in rows:
            vid_id     = row.get("Video_ID", "").strip()
            majority   = row.get("Majority_Voting", "Normal").strip()
            dur_cell   = row.get("Duration", "[]").strip()
            comp_cell  = row.get("Component", "[]").strip()

            if not vid_id:
                continue

            # Determine whether this video contributes hate annotations
            is_hateful   = majority.lower() == "hateful"
            is_offensive = majority.lower() == "offensive"
            treat_as_hate = is_hateful or (args.include_offensive and is_offensive)

            # Skip hate/offensive examples where hatefulness is signalled
            # only through metadata (title/description), since metadata is not
            # available as a video/audio/text feature at inference time.
            if treat_as_hate:
                components = parse_components(comp_cell)
                if components and all(c == "metadata" for c in components):
                    skipped += 1
                    continue

            segments = parse_segments(dur_cell) if treat_as_hate else []

            # ── Duration ─────────────────────────────────────────────────────
            video_path = videos_dir / f"{vid_id}.mp4"
            fallback_dur = (
                max(seg[1] for seg in segments) + 1.0 if segments else None
            )

            if video_path.exists():
                dur = get_duration(str(video_path), fallback=fallback_dur)
            else:
                dur = fallback_dur
                if dur is None:
                    # Normal video without a downloaded file — skip or include
                    # with duration=0.  We skip to avoid degenerate entries.
                    skipped += 1
                    print(f"  WARNING: video not found and no duration fallback, "
                          f"skipping: {vid_id}")
                    continue
                else:
                    print(f"  WARNING: video not found, using fallback duration "
                          f"({dur:.1f}s): {vid_id}")

            annotations = [
                {"segment": seg, "label": "hate"}
                for seg in segments
            ]

            database[vid_id] = {
                "duration":    round(float(dur), 3),
                "subset":      subset,
                "annotations": annotations,
                "video_label": "hate" if treat_as_hate else "non_hate",
            }

    # ── Write output ────────────────────────────────────────────────────────
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"database": database}, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────────
    subsets   = {"train": 0, "val": 0, "test": 0}
    hate_cnt  = {"train": 0, "val": 0, "test": 0}
    seg_cnt   = 0
    for entry in database.values():
        s = entry["subset"]
        subsets[s] += 1
        if entry["video_label"] == "hate":
            hate_cnt[s] += 1
        seg_cnt += len(entry["annotations"])

    print(f"\nWrote {len(database)} entries to {output}  (skipped: {skipped})")
    print(f"Total temporal segments: {seg_cnt}")
    print(f"include_offensive={args.include_offensive}")
    print(f"\n  {'subset':<8} {'total':>6} {'hate':>6} {'non-hate':>9}")
    print(f"  {'-'*35}")
    for s in ("train", "val", "test"):
        nh = subsets[s] - hate_cnt[s]
        print(f"  {s:<8} {subsets[s]:>6} {hate_cnt[s]:>6} {nh:>9}")


if __name__ == "__main__":
    main()
