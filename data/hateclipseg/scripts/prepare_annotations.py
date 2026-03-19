"""
prepare_annotations.py — Convert HateCliPSeg CSV to ActionFormer JSON format.

Outputs:
    data/hateclipseg/annotations/hateclipseg.json

The CSV has one row per video with:
  - Video Id: e.g. "bit_0EHvMSiEHVoc"
  - Segment-Level Label: list of 6-element binary vectors, one per segment
      index 0 = benign; indices 1-5 = hate categories
  - Segment Timestamp: list of [start, end] pairs (seconds)

A segment is "hate" if any of label indices 1-5 is 1.
All videos are included: those with hate segments as positives, the rest as
true negatives (empty annotations list). Only videos whose feature file is
present on disk are included (negatives without features are skipped).
Train/val split is 80/20 within each group separately to keep the ratio
balanced, then merged. Split is deterministic via random.seed(42).
Duration is inferred from the last segment's end time (or feature file if present).

Usage (run from repo root):
    python data/hateclipseg/scripts/prepare_annotations.py
"""

import ast
import csv
import json
import os
import random

import torch


CSV_PATH    = "data/hateclipseg/dataset/segment_level_annotation.csv"
FEAT_DIR    = "data/hateclipseg/video_features"
OUT_PATH    = "data/hateclipseg/dataset/hateclipseg.json"
TRAIN_RATIO = 0.8
SEED        = 42


def is_hate_segment(label_vec):
    """Returns True if any hate category (indices 1-5) is active."""
    return any(label_vec[1:])


def main():
    positives = []
    negatives = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video_id   = row["Video Id"].strip()
            labels     = ast.literal_eval(row["Segment-Level Label"])
            timestamps = ast.literal_eval(row["Segment Timestamp"])

            assert len(labels) == len(timestamps), (
                f"{video_id}: label/timestamp length mismatch"
            )

            # collect hate segments
            hate_segs = [
                [float(ts[0]), float(ts[1])]
                for lbl, ts in zip(labels, timestamps)
                if is_hate_segment(lbl)
            ]

            # duration: from feature file if available, else last segment end
            feat_path = os.path.join(FEAT_DIR, f"{video_id}.pt")
            if os.path.isfile(feat_path):
                feat = torch.load(feat_path, map_location="cpu")
                duration = float(feat.shape[0])
            else:
                if not hate_segs:
                    # no features and no hate segments — can't determine duration
                    continue
                duration = float(timestamps[-1][1])

            item = {"id": video_id, "duration": duration, "segments": hate_segs}
            if hate_segs:
                positives.append(item)
            else:
                negatives.append(item)

    print(f"Found {len(positives)} videos with hate segments, "
          f"{len(negatives)} true-negative videos.")

    # stratified split: maintain 80/20 within each group so the val set
    # always contains both positives and negatives
    random.seed(SEED)
    random.shuffle(positives)
    random.shuffle(negatives)

    def split_group(group):
        n = int(len(group) * TRAIN_RATIO)
        return group[:n], group[n:]

    pos_train, pos_val = split_group(positives)
    neg_train, neg_val = split_group(negatives)

    rows = (
        [(item, "train") for item in pos_train] +
        [(item, "val")   for item in pos_val]   +
        [(item, "train") for item in neg_train] +
        [(item, "val")   for item in neg_val]
    )

    database = {}
    for item, subset in rows:
        annotations = [
            {
                "segment":  seg,
                "label":    "hate",
                "label_id": 0,
            }
            for seg in item["segments"]
        ]
        database[item["id"]] = {
            "subset":      subset,
            "fps":         1.0,
            "duration":    item["duration"],
            "annotations": annotations,
        }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"database": database}, f, indent=2)

    n_train_out = sum(1 for v in database.values() if v["subset"] == "train")
    n_val_out   = sum(1 for v in database.values() if v["subset"] == "val")
    print(f"Written {len(database)} entries to {OUT_PATH}")
    print(f"  Train : {n_train_out}  (pos={len(pos_train)}, neg={len(neg_train)})")
    print(f"  Val   : {n_val_out}  (pos={len(pos_val)}, neg={len(neg_val)})")


if __name__ == "__main__":
    main()
