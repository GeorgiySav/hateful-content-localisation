"""
Temporal action localization evaluation: mAP at multiple tIoU thresholds.

Follows the ActivityNet evaluation protocol (ANETdetection):
  - For each tIoU threshold, compute per-class Average Precision (AP).
  - mAP = mean AP across all tIoU thresholds.
"""
import numpy as np
from typing import List, Dict


def compute_temporal_iou(seg_a, seg_b):
    """
    Compute temporal IoU between two segments.

    Args:
        seg_a: (start, end)
        seg_b: (start, end)

    Returns:
        float IoU in [0, 1]
    """
    inter_start = max(seg_a[0], seg_b[0])
    inter_end   = min(seg_a[1], seg_b[1])
    inter = max(0.0, inter_end - inter_start)
    union = (seg_a[1] - seg_a[0]) + (seg_b[1] - seg_b[0]) - inter
    if union <= 0:
        return 0.0
    return inter / union


def interpolated_prec_rec(prec, rec):
    """
    Interpolated AP: integrate precision over recall curve.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec  = np.hstack([[0], rec,  [1]])
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    idxs = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[idxs] - mrec[idxs - 1]) * mprec[idxs])
    return float(ap)


def compute_average_precision(ground_truth, prediction, tiou_threshold=0.5):
    """
    Compute Average Precision for a single class and IoU threshold.

    Args:
        ground_truth: dict {video_id: [(start, end), ...]}
        prediction  : list of {'video_id': str, 't-start': float,
                                't-end': float, 'score': float}
                      sorted by descending score.
        tiou_threshold: float

    Returns:
        float AP
    """
    npos = sum(len(v) for v in ground_truth.values())
    if npos == 0:
        return 0.0

    tp = np.zeros(len(prediction))
    fp = np.zeros(len(prediction))

    # Track which GT segments have been matched
    gt_matched = {vid: [False] * len(segs)
                  for vid, segs in ground_truth.items()}

    for idx, pred in enumerate(prediction):
        vid   = pred['video_id']
        seg_p = (pred['t-start'], pred['t-end'])

        if vid not in ground_truth or len(ground_truth[vid]) == 0:
            fp[idx] = 1
            continue

        gt_segs = ground_truth[vid]
        iou_max = 0.0
        jmax    = -1
        for j, gt_seg in enumerate(gt_segs):
            iou = compute_temporal_iou(seg_p, gt_seg)
            if iou > iou_max:
                iou_max = iou
                jmax    = j

        if iou_max >= tiou_threshold:
            if not gt_matched[vid][jmax]:
                tp[idx] = 1
                gt_matched[vid][jmax] = True
            else:
                fp[idx] = 1    # duplicate detection
        else:
            fp[idx] = 1

    tp_cumsum = np.cumsum(tp).astype(float)
    fp_cumsum = np.cumsum(fp).astype(float)
    rec  = tp_cumsum / npos
    prec = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-8)

    return interpolated_prec_rec(prec, rec)


class ANETdetection:
    """
    ActivityNet Detection evaluation wrapper.

    Args:
        ground_truth_file: Path to annotations.json.
        subset           : "val".
        tiou_thresholds  : List of tIoU thresholds to evaluate at.
        num_classes      : Number of action classes (1 for binary hate/no-hate).
        label_map        : Dict {class_index: label_name}.
    """

    def __init__(
        self,
        ground_truth_file,
        subset="val",
        tiou_thresholds=None,
        num_classes=1,
        label_map=None,
        verbose=False,
        video_ids=None,
    ):
        import json
        if tiou_thresholds is None:
            tiou_thresholds = [0.3, 0.5, 0.7]
        self.tiou_thresholds = tiou_thresholds
        self.num_classes     = num_classes
        self.label_map       = label_map or {0: 'hate'}
        self.verbose         = verbose

        with open(ground_truth_file, 'r') as f:
            db = json.load(f)['database']

        # explicit set of IDs to evaluate on
        video_id_filter = set(video_ids) if video_ids is not None else None

        # Build ground truth dict: {video_id: [(start, end), ...]}
        self.ground_truth = {}
        for vid_id, meta in db.items():
            if video_id_filter is not None:
                if vid_id not in video_id_filter:
                    continue
            elif meta.get('subset', '') != subset:
                continue
            segs = [(float(a['segment'][0]), float(a['segment'][1]))
                    for a in meta.get('annotations', [])
                    if a.get('label', 'hate').lower() in ('hate', 'hateful')]
            self.ground_truth[vid_id] = segs

        if verbose:
            n_videos  = len(self.ground_truth)
            n_segs    = sum(len(v) for v in self.ground_truth.values())
            print(f"[ANETdetection] {n_videos} videos, {n_segs} GT segments ({subset})")

    def evaluate(self, results, verbose=None):
        """
        Evaluate predictions against ground truth.

        Args:
            results: dict with keys:
                'video-id': list of str
                't-start' : np.ndarray of float
                't-end'   : np.ndarray of float
                'label'   : np.ndarray of int
                'score'   : np.ndarray of float
            verbose: override self.verbose if not None.

        Returns:
            ap_table : np.ndarray (n_classes, n_tiou)
            mAP      : float
            tiou_thresholds: list
        """
        if verbose is None:
            verbose = self.verbose

        n_classes = self.num_classes
        n_tiou    = len(self.tiou_thresholds)
        ap_table  = np.zeros((n_classes, n_tiou), dtype=np.float32)

        for cls_idx in range(n_classes):
            cls_name = self.label_map.get(cls_idx, str(cls_idx))

            # Filter predictions for this class and sort by descending score
            if len(results['video-id']) == 0:
                ap_table[cls_idx] = 0.0
                continue

            cls_mask = (np.asarray(results['label']) == cls_idx)
            if cls_mask.sum() == 0:
                ap_table[cls_idx] = 0.0
                continue

            video_ids = np.asarray(results['video-id'])[cls_mask]
            t_starts  = np.asarray(results['t-start'])[cls_mask]
            t_ends    = np.asarray(results['t-end'])[cls_mask]
            scores    = np.asarray(results['score'])[cls_mask]

            # Sort by descending score
            order = np.argsort(-scores)
            preds = [
                {
                    'video_id': video_ids[i],
                    't-start' : float(t_starts[i]),
                    't-end'   : float(t_ends[i]),
                    'score'   : float(scores[i]),
                }
                for i in order
            ]

            for tiou_idx, tiou in enumerate(self.tiou_thresholds):
                ap = compute_average_precision(
                    self.ground_truth, preds, tiou_threshold=tiou)
                ap_table[cls_idx, tiou_idx] = ap

            if verbose:
                ap_str = '  '.join(
                    f'@{t:.1f}={ap_table[cls_idx, i]:.3f}'
                    for i, t in enumerate(self.tiou_thresholds))
                print(f"  [{cls_name}]  {ap_str}")

        mAP = float(ap_table.mean())
        if verbose:
            print(f"  mAP = {mAP:.4f}")

        return ap_table, mAP, self.tiou_thresholds
