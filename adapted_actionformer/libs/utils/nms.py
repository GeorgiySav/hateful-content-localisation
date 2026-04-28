"""
Soft-NMS for 1D temporal segments.

The original ActionFormer used a compiled C++ extension (nms_1d_cpu) which
isn't convenient.
"""
import torch


def seg_iou_1d(segs_a, segs_b):
    """
    Compute pairwise 1D IoU between two sets of segments.

    Args:
        segs_a: (M, 2)
        segs_b: (N, 2)

    Returns:
        iou: (M, N)
    """
    # Intersection
    left  = torch.maximum(segs_a[:, None, 0], segs_b[None, :, 0])  # (M, N)
    right = torch.minimum(segs_a[:, None, 1], segs_b[None, :, 1])  # (M, N)
    inter = (right - left).clamp(min=0)                              # (M, N)

    # Union
    len_a = segs_a[:, 1] - segs_a[:, 0]  # (M,)
    len_b = segs_b[:, 1] - segs_b[:, 0]  # (N,)
    union = len_a[:, None] + len_b[None, :] - inter                  # (M, N)

    iou = inter / union.clamp(min=1e-8)
    return iou


def soft_nms(segs, scores, sigma=0.4, min_score=1e-3, max_num=200):
    """
    Gaussian Soft-NMS for 1D temporal segments.

    For each iteration:
      1. Pick the highest-scoring segment.
      2. Decay scores of remaining segments by exp(-IoU² / sigma).
      3. Remove segments whose score falls below min_score.
      4. Repeat until no segments remain or max_num is reached.

    Args:
        segs     : (N, 2) float tensor of segment boundaries.
        scores   : (N,)   float tensor of confidence scores.
        sigma    : Gaussian decay factor (default 0.4 per config).
        min_score: Score threshold below which segments are suppressed.
        max_num  : Maximum number of kept segments.

    Returns:
        kept_segs  : (K, 2)
        kept_scores: (K,)
    """
    if segs.shape[0] == 0:
        return torch.zeros((0, 2), dtype=segs.dtype), \
               torch.zeros((0,),   dtype=scores.dtype)

    segs   = segs.clone().float()
    scores = scores.clone().float()

    kept_segs   = []
    kept_scores = []

    while segs.shape[0] > 0 and len(kept_segs) < max_num:
        # Pick best segment
        best_idx = scores.argmax()
        best_seg   = segs[best_idx]
        best_score = scores[best_idx]

        if best_score < min_score:
            break

        kept_segs.append(best_seg)
        kept_scores.append(best_score)

        # Remove selected segment from pool
        rest_mask = torch.ones(segs.shape[0], dtype=torch.bool)
        rest_mask[best_idx] = False
        segs   = segs[rest_mask]
        scores = scores[rest_mask]

        if segs.shape[0] == 0:
            break

        # Compute IoU of selected segment with all remaining
        iou = seg_iou_1d(best_seg.unsqueeze(0), segs).squeeze(0)  # (M,)

        # Gaussian decay
        decay = torch.exp(-(iou ** 2) / sigma)
        scores = scores * decay

        # Remove segments whose score has dropped below threshold
        valid  = scores >= min_score
        segs   = segs[valid]
        scores = scores[valid]

    if len(kept_segs) == 0:
        return torch.zeros((0, 2), dtype=torch.float32), \
               torch.zeros((0,),   dtype=torch.float32)

    return torch.stack(kept_segs), torch.stack(kept_scores)


def seg_voting(nms_segs, all_segs, all_scores, iou_threshold, score_offset=1.5):
    """
    Boundary refinement via weighted combination with neighbouring segments.

    Args:
        nms_segs    : (K, 2) NMS-selected segments.
        all_segs    : (N, 2) all candidate segments before NMS.
        all_scores  : (N,)   scores of all candidates.
        iou_threshold: IoU threshold for considering a segment a neighbour.
        score_offset: Offset added to all_scores before weighting (> 0).

    Returns:
        refined_segs: (K, 2)
    """
    offset_scores = all_scores + score_offset
    iou = seg_iou_1d(nms_segs, all_segs)  # (K, N)

    seg_weights = (iou >= iou_threshold).to(all_scores.dtype) * \
                  offset_scores[None, :] * iou          # (K, N)
    seg_weights = seg_weights / seg_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    refined_segs = seg_weights @ all_segs               # (K, 2)
    return refined_segs
