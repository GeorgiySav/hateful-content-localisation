"""
Evaluation entry point for HateMM temporal hateful content localization.

Usage:
    python eval.py --config configs/default.yaml --checkpoint /path/to/model_best.pth.tar
                   [--subset val] [--output_file results.pkl]

The script:
  1. Loads config and model checkpoint.
  2. Runs inference on the specified subset.
  3. Computes temporal mAP at tIoU thresholds [0.3, 0.5, 0.7].
  4. Optionally saves raw predictions to a pickle file.
"""
import os
import sys
import argparse
import yaml
import json
import pickle

import torch
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from libs.modeling.meta_arch import HatefulContentLocalizer
from libs.datasets.hatemm     import build_dataloader
from libs.utils.eval_utils    import ANETdetection


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HateMM temporal localizer")
    parser.add_argument('--config',      required=True, help="Path to YAML config")
    parser.add_argument('--checkpoint',  required=True, help="Path to .pth.tar checkpoint")
    parser.add_argument('--subset',      default='val', choices=['val', 'test'])
    parser.add_argument('--output_file', default=None,  help="Save raw predictions to pickle")
    parser.add_argument('--tiou',        nargs='+',     type=float,
                        default=[0.3, 0.5, 0.7],
                        help="tIoU thresholds for mAP")
    return parser.parse_args()


def load_config(path):
    config_dir = os.path.dirname(os.path.abspath(path))
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    ds = cfg['dataset']
    for key in ('video_feat_dir', 'audio_feat_dir', 'text_feat_dir', 'annotation_file'):
        if key in ds and not os.path.isabs(ds[key]):
            ds[key] = os.path.normpath(os.path.join(config_dir, ds[key]))
    return cfg


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[eval] Using device: {device}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = HatefulContentLocalizer(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_key = 'state_dict' if 'state_dict' in ckpt else None
    if state_key:
        model.load_state_dict(ckpt[state_key])
    else:
        model.load_state_dict(ckpt)
    print(f"[eval] Loaded checkpoint: {args.checkpoint}")
    model.eval()

    # ── Data loader ────────────────────────────────────────────────────────────
    val_loader = build_dataloader(cfg, subset=args.subset, is_training=False)

    # ── Inference ──────────────────────────────────────────────────────────────
    results = {
        'video-id': [],
        't-start' : [],
        't-end'   : [],
        'label'   : [],
        'score'   : [],
    }

    print(f"[eval] Running inference on '{args.subset}' set ...")
    for batch_idx, batch in enumerate(val_loader):
        with torch.no_grad():
            output = model(batch)
        for res in output:
            segs   = res['segments']
            scores = res['scores']
            labels = res['labels']
            if segs.shape[0] > 0:
                results['video-id'].extend([res['video_id']] * segs.shape[0])
                results['t-start'].append(segs[:, 0].numpy())
                results['t-end'].append(segs[:, 1].numpy())
                results['label'].append(labels.numpy())
                results['score'].append(scores.numpy())
        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")

    if results['t-start']:
        results['t-start'] = np.concatenate(results['t-start'])
        results['t-end']   = np.concatenate(results['t-end'])
        results['label']   = np.concatenate(results['label'])
        results['score']   = np.concatenate(results['score'])
    else:
        for k in ('t-start', 't-end', 'label', 'score'):
            results[k] = np.array([])

    # ── Optional: save raw predictions ────────────────────────────────────────
    if args.output_file is not None:
        with open(args.output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"[eval] Saved predictions to {args.output_file}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    evaluator = ANETdetection(
        ground_truth_file=cfg['dataset']['annotation_file'],
        subset=args.subset,
        tiou_thresholds=args.tiou,
        num_classes=cfg['dataset'].get('num_classes', 1),
        label_map={0: 'hate'},
        verbose=True,
    )

    ap_table, mAP, _ = evaluator.evaluate(results, verbose=True)

    print("\n[eval] Results:")
    for tiou_idx, tiou in enumerate(args.tiou):
        print(f"  AP@{tiou:.1f} = {ap_table[0, tiou_idx]:.4f}")
    print(f"  mAP = {mAP:.4f}")


if __name__ == '__main__':
    main()
