"""
Full model: CrossModalFusion → Projection+Encoder → Decoder heads → Loss / Decode.

Architecture differences from vanilla ActionFormer:
  1. Input: three separate modality arrays (text, audio, video) from .npz files,
     fused via CrossModalFusion (CMA), instead of a single pre-fused feature.
  2. CrossModalFusion module inserted before the backbone projection layers.
  3. Binary hate/no-hate classification (1 sigmoid) via configurable num_classes.
  4. Feature stride = 1 s (1 FPS) instead of ~0.13 s; regression ranges calibrated.
  5. Weak-supervision fallback handled in the dataset (no change in model logic).

Architecture differences from vanilla MM-HSD:
  1. Per-timestep temporal localization + boundary regression, not video-level label.
  2. Three modalities only, no OCR.
  3. ActionFormer's transformer encoder replaces MM-HSD's LSTM/FC encoders.
  4. Output: temporal segments with confidence, not a binary label.

Forward pass (training):
  (text, audio, video) → CrossModalFusion → (B, T, fused_dim)
  → permute to (B, fused_dim, T)
  → ConvTransformerBackbone (projection + transformer pyramid)
  → Identity FPN (layer norm per level)
  → ClsHead + RegHead
  → focal loss + DIoU loss

Forward pass (inference):
  Same backbone → decode candidate segments → Soft-NMS
  → list of (start_time, end_time, confidence) tuples
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .cross_modal_fusion import CrossModalFusion
from .backbone import ConvTransformerBackbone
from .heads import ClsHead, RegHead
from .blocks import MaskedConv1D, LayerNorm


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions (ported from ActionFormer's losses.py)
# ──────────────────────────────────────────────────────────────────────────────

def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="none"):
    """
    Sigmoid Focal Loss for binary classification.
    Critical for handling the massive foreground/background imbalance —
    most timesteps are non-hateful.
    """
    inputs  = inputs.float()
    targets = targets.float()
    p       = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t  = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def ctr_diou_loss_1d(input_offsets, target_offsets, reduction="none", eps=1e-8):
    """
    1D DIoU loss for segment boundary regression.
    Only applied to positive (foreground) timesteps.
    """
    input_offsets  = input_offsets.float()
    target_offsets = target_offsets.float()
    assert (input_offsets  >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0],  input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)
    intsctk = rkis + lkis
    unionk   = (lp + rp) + (lg + rg) - intsctk
    iouk     = intsctk / unionk.clamp(min=eps)

    lc   = torch.max(lp, lg)
    rc   = torch.max(rp, rg)
    len_c = lc + rc
    rho  = 0.5 * (rp - lp - rg + lg)
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Point generator (ported from ActionFormer's loc_generators.py)
# ──────────────────────────────────────────────────────────────────────────────

class PointGenerator(nn.Module):
    """
    Pre-computes temporal grid points for each FPN level.
    Each point stores (t, reg_range_min, reg_range_max, stride).
    """

    def __init__(self, max_seq_len, fpn_strides, regression_range):
        super().__init__()
        assert len(fpn_strides) == len(regression_range)
        self.max_seq_len     = max_seq_len
        self.fpn_levels      = len(fpn_strides)
        self.fpn_strides     = fpn_strides
        self.regression_range = regression_range

        # Buffer the point lists (non-persistent so they are not saved in checkpoints)
        points_list = []
        for l, stride in enumerate(fpn_strides):
            reg_range  = torch.as_tensor(regression_range[l], dtype=torch.float)
            fpn_stride = torch.as_tensor(stride, dtype=torch.float)
            points     = torch.arange(0, max_seq_len, stride)[:, None]
            reg_range  = reg_range[None].repeat(points.shape[0], 1)
            fpn_stride = fpn_stride[None].repeat(points.shape[0], 1)
            points_list.append(torch.cat((points, reg_range, fpn_stride), dim=1))

        for i, pts in enumerate(points_list):
            self.register_buffer(str(i), pts, persistent=False)

    def forward(self, feats):
        pts_list  = []
        feat_lens = [f.shape[-1] for f in feats]
        for l, feat_len in enumerate(feat_lens):
            buffer_pts = getattr(self, str(l))
            assert feat_len <= buffer_pts.shape[0], \
                f"Feature length {feat_len} exceeds point generator buffer at level {l}"
            pts_list.append(buffer_pts[:feat_len, :])
        return pts_list


# ──────────────────────────────────────────────────────────────────────────────
# Identity FPN neck (layer norm only, no lateral/top-down fusion)
# ──────────────────────────────────────────────────────────────────────────────

class FPNIdentity(nn.Module):
    """Pass-through neck with per-level LayerNorm."""

    def __init__(self, n_levels, n_embd, with_ln=True):
        super().__init__()
        self.norms = nn.ModuleList([
            LayerNorm(n_embd) if with_ln else nn.Identity()
            for _ in range(n_levels)
        ])

    def forward(self, feats, masks):
        out_feats, out_masks = tuple(), tuple()
        for i, (f, m) in enumerate(zip(feats, masks)):
            out_feats += (self.norms[i](f),)
            out_masks += (m,)
        return out_feats, out_masks


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────

class HatefulContentLocalizer(nn.Module):
    """
    Temporal hateful content localization model.

    Input  : (text, audio, video) feature tensors + padding mask
    Output : (training)  loss dict with cls_loss, reg_loss, final_loss
             (inference) list of dicts with keys: video_id, segments, scores, labels
    """

    def __init__(self, cfg):
        super().__init__()
        # ── Config shortcuts ─────────────────────────────────────────────────
        ds_cfg  = cfg['dataset']
        fus_cfg = cfg['fusion']
        bb_cfg  = cfg['backbone']
        hd_cfg  = cfg['heads']
        loss_cfg = cfg['loss']
        infer_cfg = cfg['inference']
        train_cfg = cfg.get('training', {})

        text_dim  = ds_cfg['input_dims']['text']   # 768
        audio_dim = ds_cfg['input_dims']['audio']  # 1024
        video_dim = ds_cfg['input_dims']['video']  # 768
        num_classes = ds_cfg.get('num_classes', 1)

        d_cma     = fus_cfg['d_cma']       # 128
        fused_dim = d_cma                  # CMA output only (not re-concatenated)

        d_model    = bb_cfg['d_model']       # 512
        n_layers   = bb_cfg['n_layers']      # 6
        n_head     = bb_cfg['n_heads']       # 4
        n_proj     = bb_cfg['n_proj_layers'] # 2
        win_size   = bb_cfg['window_size']   # 19
        ds_start   = bb_cfg['downsample_start']  # 1 (0-indexed, first branch index)
        scale_factor = bb_cfg['downsample_ratio'] # 2

        n_stem   = ds_start          # number of blocks WITHOUT downsampling
        n_branch = n_layers - n_stem  # blocks WITH downsampling
        # Default: n_stem=1, n_branch=2 (total 3 transformer blocks)

        self.n_levels    = 1 + n_branch
        self.num_classes = num_classes
        self.scale_factor = scale_factor

        # FPN strides: 1, 2, 4, 8, 16, 32
        self.fpn_strides = [scale_factor ** i for i in range(self.n_levels)]

        # Regression ranges per level (in absolute time units = seconds at 1 FPS)
        reg_ranges_cfg = cfg.get('regression_ranges', None)
        if reg_ranges_cfg is None:
            self.reg_range = self._default_reg_range(self.n_levels)
        else:
            self.reg_range = reg_ranges_cfg
        assert len(self.reg_range) == self.n_levels

        # ── Stage 1: Cross-modal fusion ──────────────────────────────────────
        self.fusion = CrossModalFusion(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            d_cma=d_cma,
            num_heads=fus_cfg['num_heads'],
            dropout=fus_cfg['dropout'],
        )

        # ── Stage 2: Multiscale transformer encoder ──────────────────────────
        mha_win_size = [win_size] * (1 + n_branch)  # same window for all levels
        self.backbone = ConvTransformerBackbone(
            n_in=fused_dim,
            n_embd=d_model,
            n_head=n_head,
            n_embd_ks=3,
            max_len=ds_cfg.get('max_seq_len', 2304),
            arch=(n_proj, n_stem, n_branch),
            mha_win_size=mha_win_size,
            scale_factor=scale_factor,
            with_ln=True,
            attn_pdrop=0.0,
            proj_pdrop=train_cfg.get('dropout', 0.0),
            path_pdrop=train_cfg.get('droppath', 0.0),
            use_abs_pe=False,  # no positional encoding per ActionFormer recommendation
            use_rel_pe=False,
        )

        # ── Identity FPN neck (layer norm only) ──────────────────────────────
        self.neck = FPNIdentity(self.n_levels, d_model, with_ln=True)

        # ── Stage 3: Decoder heads ────────────────────────────────────────────
        self.cls_head = ClsHead(
            input_dim=d_model,
            feat_dim=d_model,
            num_classes=num_classes,
            prior_prob=train_cfg.get('cls_prior_prob', 0.01),
            n_layers=hd_cfg['n_layers'],
            kernel_size=hd_cfg['kernel_size'],
            with_ln=hd_cfg['use_layer_norm'],
        )
        self.reg_head = RegHead(
            input_dim=d_model,
            feat_dim=d_model,
            fpn_levels=self.n_levels,
            n_layers=hd_cfg['n_layers'],
            kernel_size=hd_cfg['kernel_size'],
            with_ln=hd_cfg['use_layer_norm'],
        )

        # ── Point generator ───────────────────────────────────────────────────
        max_buf = ds_cfg.get('max_seq_len', 2304) * 4
        self.point_generator = PointGenerator(
            max_seq_len=max_buf,
            fpn_strides=self.fpn_strides,
            regression_range=self.reg_range,
        )

        # ── Loss config ───────────────────────────────────────────────────────
        self.focal_alpha  = loss_cfg.get('focal_alpha', 0.25)
        self.focal_gamma  = loss_cfg.get('focal_gamma', 2.0)
        self.lambda_reg   = loss_cfg.get('lambda_reg', 1.0)
        self.center_sampling = loss_cfg.get('center_sampling', True)
        self.center_sampling_radius = loss_cfg.get('center_sampling_radius', 1.5)
        self.label_smoothing = train_cfg.get('label_smoothing', 0.0)

        # EMA of #foreground for stable loss normalisation (from ActionFormer)
        self.loss_normalizer = 100
        self.loss_normalizer_momentum = 0.9

        # ── Inference config ──────────────────────────────────────────────────
        self.score_threshold = infer_cfg.get('score_threshold', 0.001)
        self.nms_sigma       = infer_cfg.get('nms_sigma', 0.4)
        self.nms_threshold   = infer_cfg.get('nms_threshold', 0.1)
        self.max_detections  = infer_cfg.get('max_detections', 200)
        self.feature_fps     = ds_cfg.get('feature_fps', 1.0)

        # Maximum sequence length needed for padding divisibility check
        self.max_seq_len = ds_cfg.get('max_seq_len', 2304)
        # Compute max_div_factor (same logic as ActionFormer)
        max_div = 1
        for s, w in zip(self.fpn_strides, mha_win_size):
            stride = s * (w // 2) * 2 if w > 1 else s
            if max_div < stride:
                max_div = stride
        self.max_div_factor = max_div

    @staticmethod
    def _default_reg_range(n_levels):
        """Default regression ranges in seconds (at 1 FPS feature stride)."""
        ranges = []
        r = 4
        for i in range(n_levels):
            if i == 0:
                ranges.append([0, r])
            elif i == n_levels - 1:
                ranges.append([r, float('inf')])
            else:
                ranges.append([r, r * 2])
                r *= 2
        return ranges

    @property
    def device(self):
        return next(self.parameters()).device

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, batch):
        """
        Args (training):
            batch: dict with keys:
                'text_feat'  : (B, T_max, 768)
                'audio_feat' : (B, T_max, 1024)
                'video_feat' : (B, T_max, 768)
                'mask'       : (B, T_max)  float, 1 for valid, 0 for padded
                'segments'   : list[Tensor(N_i, 2)]  — GT segments in seconds
                'labels'     : list[Tensor(N_i,)]    — GT class indices

        Returns (training):
            dict with cls_loss, reg_loss, final_loss

        Returns (inference):
            list of dicts with video_id, segments (T×2), scores (T,), labels (T,)
        """
        text  = batch['text_feat'].to(self.device)    # (B, T, 768)
        audio = batch['audio_feat'].to(self.device)   # (B, T, 1024)
        video = batch['video_feat'].to(self.device)   # (B, T, 768)
        mask_float = batch['mask'].to(self.device)    # (B, T)

        B, T, _ = text.shape

        # ── Stage 1: Cross-modal fusion ──────────────────────────────────────
        fused = self.fusion(text, audio, video)  # (B, T, fused_dim)

        # Convert to (B, fused_dim, T) for Conv1D backbone
        x = fused.permute(0, 2, 1)
        # Mask: (B, 1, T) bool
        mask_bcT = mask_float.unsqueeze(1).bool()  # (B, 1, T)

        # ── Stage 2: Multiscale transformer encoder ──────────────────────────
        feats, masks = self.backbone(x, mask_bcT)

        # ── Identity FPN neck ─────────────────────────────────────────────────
        fpn_feats, fpn_masks = self.neck(feats, masks)

        # ── FPN grid points ───────────────────────────────────────────────────
        points = self.point_generator(fpn_feats)

        # ── Stage 3: Classification + regression heads ────────────────────────
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)  # tuple of (B, C, T_i)
        out_offsets    = self.reg_head(fpn_feats, fpn_masks)  # tuple of (B, 2, T_i)

        # Permute to (B, T_i, C) / (B, T_i, 2)
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        out_offsets    = [x.permute(0, 2, 1) for x in out_offsets]
        # Squeeze mask to (B, T_i)
        fpn_masks_2d   = [m.squeeze(1) for m in fpn_masks]

        if self.training:
            gt_segments = [s.to(self.device) for s in batch['segments']]
            gt_labels   = [l.to(self.device) for l in batch['labels']]
            gt_cls, gt_offsets = self.label_points(points, gt_segments, gt_labels)
            return self.losses(fpn_masks_2d, out_cls_logits, out_offsets, gt_cls, gt_offsets)
        else:
            return self.inference(batch, points, fpn_masks_2d, out_cls_logits, out_offsets)

    # ─────────────────────────────────────────────────────────────────────────
    # Label assignment
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        concat_points = torch.cat(points, dim=0)  # (FT, 4)
        gt_cls, gt_off = [], []
        for segs, labels in zip(gt_segments, gt_labels):
            cls_t, reg_t = self._label_single_video(concat_points, segs, labels)
            gt_cls.append(cls_t)
            gt_off.append(reg_t)
        return gt_cls, gt_off

    @torch.no_grad()
    def _label_single_video(self, concat_points, gt_segment, gt_label):
        """
        Assign GT class and regression targets to each FPN grid point.

        concat_points : (FT, 4)  — (t, reg_min, reg_max, stride)
        gt_segment    : (N, 2)   — start/end in seconds
        gt_label      : (N,)     — class indices
        """
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        lens   = gt_segment[:, 1] - gt_segment[:, 0]        # (N,)
        lens   = lens[None, :].repeat(num_pts, 1)             # (FT, N)

        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left  = concat_points[:, 0, None] - gt_segs[:, :, 0]  # (FT, N)
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]  # (FT, N)
        reg_targets = torch.stack((left, right), dim=-1)        # (FT, N, 2)

        if self.center_sampling:
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            t_mins = center_pts - concat_points[:, 3, None] * self.center_sampling_radius
            t_maxs = center_pts + concat_points[:, 3, None] * self.center_sampling_radius
            cb_dist_left  = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
            center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        max_regress_distance = reg_targets.max(-1)[0]  # (FT, N)
        inside_regress_range = torch.logical_and(
            max_regress_distance >= concat_points[:, 1, None],
            max_regress_distance <= concat_points[:, 2, None],
        )

        lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))
        lens.masked_fill_(inside_regress_range == 0, float('inf'))
        min_len, min_len_inds = lens.min(dim=1)

        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)),
            (lens < float('inf'))
        ).to(reg_targets.dtype)

        gt_label_one_hot = F.one_hot(gt_label, self.num_classes).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot   # (FT, C)
        cls_targets.clamp_(min=0.0, max=1.0)

        reg_targets = reg_targets[range(num_pts), min_len_inds]  # (FT, 2)
        reg_targets /= concat_points[:, 3, None]   # normalise by stride
        return cls_targets, reg_targets

    # ─────────────────────────────────────────────────────────────────────────
    # Loss computation
    # ─────────────────────────────────────────────────────────────────────────

    def losses(self, fpn_masks, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets):
        """
        L = (focal_cls + λ_reg * L_diou_reg) / T+

        fpn_masks      : list[B, T_i]
        out_cls_logits : list[B, T_i, C]
        out_offsets    : list[B, T_i, 2]
        gt_cls_labels  : list[FT, C]  (length B)
        gt_offsets     : list[FT, 2]  (length B)
        """
        valid_mask = torch.cat(fpn_masks, dim=1)      # (B, FT)
        gt_cls     = torch.stack(gt_cls_labels)        # (B, FT, C)
        pos_mask   = torch.logical_and(gt_cls.sum(-1) > 0, valid_mask)  # (B, FT)

        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]  # (#Pos, 2)
        gt_off       = torch.stack(gt_offsets)[pos_mask]         # (#Pos, 2)

        num_pos = pos_mask.sum().item()
        self.loss_normalizer = (
            self.loss_normalizer_momentum * self.loss_normalizer
            + (1 - self.loss_normalizer_momentum) * max(num_pos, 1)
        )

        gt_target = gt_cls[valid_mask]
        if self.label_smoothing > 0:
            gt_target = gt_target * (1 - self.label_smoothing)
            gt_target = gt_target + self.label_smoothing / (self.num_classes + 1)

        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction='sum',
        ) / self.loss_normalizer

        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            reg_loss = ctr_diou_loss_1d(pred_offsets, gt_off, reduction='sum')
            reg_loss = reg_loss / self.loss_normalizer

        final_loss = cls_loss + self.lambda_reg * reg_loss
        return {
            'cls_loss'   : cls_loss,
            'reg_loss'   : reg_loss,
            'final_loss' : final_loss,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def inference(self, batch, points, fpn_masks, out_cls_logits, out_offsets):
        """Decode predictions, apply Soft-NMS, return segment list."""
        from ..utils.nms import soft_nms

        results = []
        video_ids = batch.get('video_id', [f'video_{i}' for i in range(len(out_cls_logits[0]))])
        durations  = batch.get('duration', [None] * len(out_cls_logits[0]))
        if not isinstance(video_ids, (list, tuple)):
            video_ids = [video_ids]
        if not isinstance(durations, (list, tuple)):
            durations = [durations]

        B = out_cls_logits[0].shape[0]
        for b in range(B):
            segs_all, scores_all = [], []
            for cls_i, off_i, pts_i, mask_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks
            ):
                # Sigmoid score for this sample
                # cls_i: (B, T_i, C=1), mask_i: (B, T_i)
                score = cls_i[b].sigmoid() * mask_i[b].unsqueeze(-1)  # (T_i, 1)
                score = score.squeeze(-1)                               # (T_i,)

                keep = score > self.score_threshold
                if keep.sum() == 0:
                    continue

                score_k = score[keep]
                pts_k   = pts_i[keep]
                off_k   = off_i[b][keep]

                # Decode: start = t - d_start*stride, end = t + d_end*stride
                seg_l = pts_k[:, 0] - off_k[:, 0] * pts_k[:, 3]
                seg_r = pts_k[:, 0] + off_k[:, 1] * pts_k[:, 3]
                segs_k = torch.stack((seg_l, seg_r), -1)

                segs_all.append(segs_k)
                scores_all.append(score_k)

            if len(segs_all) == 0:
                results.append({
                    'video_id': video_ids[b],
                    'segments': torch.zeros((0, 2)),
                    'scores'  : torch.zeros((0,)),
                    'labels'  : torch.zeros((0,), dtype=torch.long),
                })
                continue

            segs_all   = torch.cat(segs_all, dim=0).cpu()
            scores_all = torch.cat(scores_all, dim=0).cpu()

            # Soft-NMS
            segs_nms, scores_nms = soft_nms(
                segs_all, scores_all,
                sigma=self.nms_sigma,
                min_score=self.score_threshold,
                max_num=self.max_detections,
            )

            # Convert from feature grid coords to seconds
            # grid coord * (1/fps) ≈ seconds at 1 FPS
            segs_sec = segs_nms / self.feature_fps

            # Clamp to [0, duration]
            dur = durations[b]
            if dur is not None:
                segs_sec = segs_sec.clamp(min=0.0, max=float(dur))

            n = segs_sec.shape[0]
            results.append({
                'video_id': video_ids[b],
                'segments': segs_sec,
                'scores'  : scores_nms,
                'labels'  : torch.zeros(n, dtype=torch.long),  # binary: all class 0
            })

        return results
