"""
Training entry point for HateMM temporal hateful content localization.

Usage:
    python train.py --config configs/default.yaml [--output_dir runs/exp1]
                    [--seed 42] [--resume /path/to/checkpoint.pth.tar]

The script:
  1. Loads config from YAML.
  2. Builds model, optimizer, LR scheduler.
  3. Optionally loads a checkpoint to resume training.
  4. Trains for cfg.training.epochs epochs, saving checkpoints.
  5. After each epoch, runs validation and reports mAP.
  6. Optionally writes TensorBoard logs.
"""
import os
import sys
import argparse
import yaml

import torch
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# Make sure the scripts/ directory is on the path when called from any cwd
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from libs.modeling.meta_arch  import HatefulContentLocalizer
from libs.datasets.hatemm      import build_dataloader
from libs.utils.train_utils    import (
    fix_random_seed, make_optimizer, make_scheduler,
    ModelEma, save_checkpoint, train_one_epoch, valid_one_epoch
)
from libs.utils.eval_utils     import ANETdetection


def parse_args():
    parser = argparse.ArgumentParser(description="Train HateMM temporal localizer")
    parser.add_argument('--config',     required=True,  help="Path to YAML config")
    parser.add_argument('--output_dir', default='runs', help="Output directory")
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--resume',     default=None,   help="Checkpoint to resume from")
    parser.add_argument('--no_tb',      action='store_true', help="Disable TensorBoard")
    return parser.parse_args()


def load_config(path):
    # Resolve paths relative to the config file location
    config_dir = os.path.dirname(os.path.abspath(path))
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths in dataset config
    ds = cfg['dataset']
    for key in ('video_feat_dir', 'audio_feat_dir', 'text_feat_dir', 'annotation_file'):
        if key in ds and not os.path.isabs(ds[key]):
            ds[key] = os.path.normpath(os.path.join(config_dir, ds[key]))
    return cfg


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    fix_random_seed(args.seed)

    # ── Output directory ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    tb_writer = None
    if HAS_TENSORBOARD and not args.no_tb:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tb'))

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_loader      = build_dataloader(cfg, subset='train', is_training=True)
    train_eval_loader = build_dataloader(cfg, subset='train', is_training=False)
    val_loader        = build_dataloader(cfg, subset='val',   is_training=False)

    # ── Evaluators ────────────────────────────────────────────────────────────
    tiou_thresholds = [0.3, 0.5, 0.7]
    evaluator_kwargs = dict(
        ground_truth_file=cfg['dataset']['annotation_file'],
        tiou_thresholds=tiou_thresholds,
        num_classes=cfg['dataset'].get('num_classes', 1),
        label_map={0: 'hate'},
    )
    train_evaluator = ANETdetection(**evaluator_kwargs, subset='train', verbose=False)
    evaluator       = ANETdetection(**evaluator_kwargs, subset='val',   verbose=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] Using device: {device}")

    model = HatefulContentLocalizer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {n_params:,}")

    # ── Optimizer / scheduler ─────────────────────────────────────────────────
    train_cfg = cfg['training']
    optimizer = make_optimizer(model, train_cfg)
    scheduler = make_scheduler(optimizer, train_cfg, len(train_loader))

    # ── Optional EMA ─────────────────────────────────────────────────────────
    model_ema = None
    if train_cfg.get('use_ema', True):
        model_ema = ModelEma(model, decay=train_cfg.get('ema_decay', 0.999))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_mAP    = 0.0
    if args.resume is not None and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_mAP    = ckpt.get('best_mAP', 0.0)
        print(f"[train] Resumed from epoch {start_epoch}, best mAP={best_mAP:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    clip_grad = train_cfg.get('clip_grad_norm', 1.0)
    epochs    = train_cfg.get('epochs', 50)

    for epoch in range(start_epoch, epochs):
        train_one_epoch(
            train_loader, model, optimizer, scheduler,
            curr_epoch=epoch,
            model_ema=model_ema,
            clip_grad_norm=clip_grad,
            tb_writer=tb_writer,
        )

        eval_model = model_ema.module if model_ema is not None else model
 
        if epoch % 5 == 0:
            # Train-set mAP (no grad, deterministic cropping)
            train_mAP = valid_one_epoch(
                train_eval_loader, eval_model,
                curr_epoch=epoch,
                evaluator=train_evaluator,
                tb_writer=None,         # log separately below
            )
        else:
            train_mAP = 0.0  # skip expensive train mAP every epoch, log 0.0 as placeholder

        # Validation mAP
        mAP = valid_one_epoch(
            val_loader, eval_model,
            curr_epoch=epoch,
            evaluator=evaluator,
            tb_writer=tb_writer,
        )

        if tb_writer is not None:
            tb_writer.add_scalar('train/mAP', train_mAP, epoch)

        if train_mAP > 0.0:
            print(f"[epoch {epoch}]  train mAP={train_mAP:.4f}  "
              f"val mAP={mAP:.4f}  best={max(mAP, best_mAP):.4f}")
        else:
            print(f"[epoch {epoch}]  val mAP={mAP:.4f}  best={max(mAP, best_mAP):.4f}")

        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)

        save_checkpoint(
            {
                'epoch'      : epoch,
                'state_dict' : model.state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'scheduler'  : scheduler.state_dict(),
                'best_mAP'   : best_mAP,
                'mAP'        : mAP,
            },
            is_best=is_best,
            file_folder=args.output_dir,
        )

    if tb_writer is not None:
        tb_writer.close()
    print(f"\n[train] Done. Best val mAP = {best_mAP:.4f}")


if __name__ == '__main__':
    main()
