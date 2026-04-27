"""
Training utilities: optimizer, LR scheduler, EMA, training/validation loops.

Ported from ActionFormer's train_utils.py with these adaptations:
  - Adapted for HatefulContentLocalizer (dict-based batch, not video_list)
  - LR scheduler: Adam with linear warmup + cosine annealing (from config)
  - EMA: optional exponential moving average of model weights
  - make_optimizer: separates decay/no_decay parameter groups
"""
import os
import time
import random
import math
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from ..modeling.blocks import MaskedConv1D, LayerNorm, Scale, AffineDropPath


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state, is_best, file_folder, file_name='checkpoint.pth.tar'):
    os.makedirs(file_folder, exist_ok=True)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        best_state = {k: v for k, v in state.items()
                      if k not in ('optimizer', 'scheduler')}
        torch.save(best_state, os.path.join(file_folder, 'model_best.pth.tar'))


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer
# ──────────────────────────────────────────────────────────────────────────────

def make_optimizer(model, cfg):
    """
    Build AdamW optimizer with separate weight-decay groups.

    Parameters whose names end with 'bias', belong to LayerNorm / GroupNorm,
    or are Scale / AffineDropPath scalars or relative-PE parameters do NOT
    receive weight decay (following ActionFormer's pattern).
    """
    decay   = set()
    no_decay = set()
    whitelist = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D, torch.nn.MultiheadAttention)
    blacklist = (LayerNorm, torch.nn.LayerNorm, torch.nn.GroupNorm)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist):
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                no_decay.add(fpn)

    param_dict   = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Params in both decay/no_decay: {inter_params}"
    assert len(param_dict.keys() - union_params) == 0, \
        f"Unclassified params: {param_dict.keys() - union_params}"

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)],
         "weight_decay": cfg.get('weight_decay', 1e-4)},
        {"params": [param_dict[pn] for pn in sorted(no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optim_groups, lr=cfg.get('learning_rate', 1e-4))
    return optimizer


# ──────────────────────────────────────────────────────────────────────────────
# LR Scheduler: linear warmup + cosine annealing
# ──────────────────────────────────────────────────────────────────────────────

class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup then cosine annealing, stepped per iteration."""

    def __init__(self, optimizer, warmup_steps, max_steps,
                 warmup_start_lr=0.0, eta_min=1e-8, last_epoch=-1):
        self.warmup_steps    = warmup_steps
        self.max_steps       = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min         = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_steps:
            return [
                group['lr'] + (base_lr - self.warmup_start_lr) / (self.warmup_steps - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_steps:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_steps) % (
                2 * (self.max_steps - self.warmup_steps)) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_steps - self.warmup_steps))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) /
                          (self.max_steps - self.warmup_steps))) /
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps - 1) /
                          (self.max_steps - self.warmup_steps))) *
            (group['lr'] - self.eta_min) + self.eta_min
            for group in self.optimizer.param_groups
        ]


def make_scheduler(optimizer, cfg, num_iters_per_epoch, last_epoch=-1):
    """
    Build an LR scheduler from the training config.
    The scheduler is designed to be stepped once per *iteration*.
    """
    epochs        = cfg.get('epochs', 50)
    warmup_epochs = cfg.get('warmup_epochs', 5)
    warmup_steps  = warmup_epochs  * num_iters_per_epoch
    max_steps     = (epochs + warmup_epochs) * num_iters_per_epoch

    sched_type = cfg.get('lr_scheduler', 'cosine')
    if sched_type == 'cosine':
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_steps, max_steps, last_epoch=last_epoch)
    elif sched_type == 'multistep':
        steps = [num_iters_per_epoch * s for s in cfg.get('schedule_steps', [])]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, steps, gamma=cfg.get('schedule_gamma', 0.1),
            last_epoch=last_epoch)
    else:
        raise ValueError(f"Unknown lr_scheduler: {sched_type}")
    return scheduler


# ──────────────────────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────────────────────

class ModelEma(torch.nn.Module):
    """Exponential Moving Average of model weights."""

    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay  = decay
        self.device = device
        if device is not None:
            self.module.to(device=device)

    @torch.no_grad()
    def update(self, model):
        for ema_v, m_v in zip(self.module.state_dict().values(),
                               model.state_dict().values()):
            if self.device is not None:
                m_v = m_v.to(device=self.device)
            ema_v.copy_(self.decay * ema_v + (1.0 - self.decay) * m_v)

    @torch.no_grad()
    def set(self, model):
        for ema_v, m_v in zip(self.module.state_dict().values(),
                               model.state_dict().values()):
            if self.device is not None:
                m_v = m_v.to(device=self.device)
            ema_v.copy_(m_v)


# ──────────────────────────────────────────────────────────────────────────────
# AverageMeter
# ──────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    model_ema=None,
    clip_grad_norm=1.0,
    print_freq=20,
):
    """Train for one epoch."""
    batch_time    = AverageMeter()
    losses_tracker = {}
    num_iters = len(train_loader)
    model.train()

    print(f"\n[Train] Epoch {curr_epoch} started")
    start = time.time()

    for iter_idx, batch in enumerate(train_loader, 0):
        optimizer.zero_grad(set_to_none=True)
        losses = model(batch)
        losses['final_loss'].backward()

        if clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        if (iter_idx != 0) and (iter_idx % print_freq == 0):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            for key, value in losses.items():
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                losses_tracker[key].update(value.item())

            lr = scheduler.get_last_lr()[0]

            loss_str = '  '.join(
                f'{k} {v.val:.4f}' for k, v in losses_tracker.items())
            print(f"  [{curr_epoch}][{iter_idx:05d}/{num_iters:05d}]"
                  f"  t={batch_time.val:.2f}s  {loss_str}  lr={lr:.2e}")

    lr = scheduler.get_last_lr()[0]
    print(f"[Train] Epoch {curr_epoch} done  lr={lr:.8f}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Validation loop
# ──────────────────────────────────────────────────────────────────────────────

def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    evaluator=None,
    output_file=None,
    print_freq=1000,
):
    """
    Run inference on val set and compute mAP.

    Either evaluator (ANETdetection) or output_file must be provided.
    """
    assert (evaluator is not None) or (output_file is not None), \
        "Provide evaluator or output_file"

    batch_time = AverageMeter()
    model.eval()

    results = {
        'video-id': [],
        't-start' : [],
        't-end'   : [],
        'label'   : [],
        'score'   : [],
    }

    start = time.time()
    for iter_idx, batch in enumerate(val_loader, 0):
        # Only evaluate on hateful videos (those with ground truth segments)
        if not any(s.shape[0] > 0 for s in batch['segments']):
            continue

        with torch.no_grad():
            output = model(batch)

        for res in output:
            segs   = res['segments']
            scores = res['scores']
            labels = res['labels']
            if segs.shape[0] > 0:
                results['video-id'].extend([res['video_id']] * segs.shape[0])
                results['t-start'].append(segs[:, 0])
                results['t-end'].append(segs[:, 1])
                results['label'].append(labels)
                results['score'].append(scores)

        if (iter_idx != 0) and (iter_idx % print_freq == 0):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()
            print(f"  Val [{iter_idx:05d}/{len(val_loader):05d}]"
                  f"  t={batch_time.val:.2f}s")

    # Concatenate
    if results['t-start']:
        results['t-start'] = torch.cat(results['t-start']).numpy()
        results['t-end']   = torch.cat(results['t-end']).numpy()
        results['label']   = torch.cat(results['label']).numpy()
        results['score']   = torch.cat(results['score']).numpy()
    else:
        import numpy as np
        results['t-start'] = np.array([])
        results['t-end']   = np.array([])
        results['label']   = np.array([])
        results['score']   = np.array([])

    mAP = 0.0
    mAP_per_tiou = []
    if evaluator is not None:
        ap_table, mAP, tiou_thresholds = evaluator.evaluate(results, verbose=True)
        mAP_per_tiou = ap_table.mean(axis=0).tolist()
    elif output_file is not None:
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)

    return mAP, mAP_per_tiou
