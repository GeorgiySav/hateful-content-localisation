"""
Sequential experiment runner for TemporalMaxer ablations on HateMM.

Usage:
    # Run all experiments (skip any already done):
    python run_experiments.py

    # Force re-run a specific experiment (by name):
    python run_experiments.py --force low_dropout

    # Dry-run: print what would run without training:
    python run_experiments.py --dry_run

    # Use a specific Python executable:
    python run_experiments.py --python C:/Users/Georgiy/anaconda3/envs/hcl/python.exe

After all experiments finish (or are skipped) a comparison table is printed and
results are saved to runs/experiment_results.json.

Skip logic:
    If <output_dir>/model_best.pth.tar already exists for an experiment, training
    is skipped and best_mAP is read directly from the checkpoint.  Use --force
    <name> to delete that file and re-run a single experiment.

Resume logic:
    If training is interrupted mid-run, <output_dir>/checkpoint.pth.tar holds the
    last saved state.  On the next invocation the runner resumes automatically by
    passing --resume <output_dir>/checkpoint.pth.tar to train.py (unless a
    model_best already exists, in which case it is skipped).
"""
import os
import sys
import json
import argparse
import subprocess

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Experiment registry ────────────────────────────────────────────────────────
# (name, config_path_relative_to_script_dir, output_dir, description)
EXPERIMENTS = [
    (
        "baseline",
        "configs/temporalmaxer.yaml",
        "runs/temporalmaxer",
        "Baseline: MaxPool + identity neck + shallow heads",
    ),
    (
        "low_dropout",
        "configs/exp_low_dropout.yaml",
        "runs/exp_low_dropout",
        "Reduced regularization (dropout 0.5->0.3, droppath 0.3->0.1)",
    ),
    (
        "fpn_neck",
        "configs/exp_fpn_neck.yaml",
        "runs/exp_fpn_neck",
        "FPN neck: lateral convs + top-down feature fusion",
    ),
    (
        "rich_proj",
        "configs/exp_rich_proj.yaml",
        "runs/exp_rich_proj",
        "Rich projection: d_cma 64->128, n_proj 1->2, head layers 1->2",
    ),
    (
        "4level",
        "configs/exp_4level.yaml",
        "runs/exp_4level",
        "4-level pyramid: n_layers 3->4, extended regression ranges",
    ),
    (
        "combined",
        "configs/exp_combined.yaml",
        "runs/exp_combined",
        "Combined: rich proj + FPN neck + lower dropout",
    ),
    # ── Round 2: informed by round-1 results ──────────────────────────────────
    # Finding: FPN hurts; low_dropout==baseline; rich_proj slightly below baseline
    (
        "lr_high",
        "configs/exp_lr_high.yaml",
        "runs/exp_lr_high",
        "Higher LR (1e-4->3e-4): test if optimizer is the bottleneck",
    ),
    (
        "d_cma_only",
        "configs/exp_d_cma_only.yaml",
        "runs/exp_d_cma_only",
        "Wider CMA only (d_cma 64->128): isolate fusion capacity",
    ),
    (
        "long_baseline",
        "configs/exp_long_baseline.yaml",
        "runs/exp_long_baseline",
        "Long training (10->50 epochs): baseline arch with more training time",
    ),
    # ── Round 3: combine training-duration insight with architecture wins ──────
    # Finding: 50 epochs >> 10 epochs; LR 3e-4 and d_cma=128 also help
    (
        "long_lr_high",
        "configs/exp_long_lr_high.yaml",
        "runs/exp_long_lr_high",
        "Long + high LR: 50 ep, LR 3e-4",
    ),
    (
        "long_d_cma",
        "configs/exp_long_d_cma.yaml",
        "runs/exp_long_d_cma",
        "Long + wide CMA: 50 ep, d_cma=128",
    ),
    (
        "long_best",
        "configs/exp_long_best.yaml",
        "runs/exp_long_best",
        "Long + high LR + wide CMA: best combination from rounds 1-2",
    ),
    # ── Round 4: attack overfitting (train/val gap ~10x at epoch 35) ──────────
    # Finding: long_baseline (207K, LR 1e-4, 50ep) is best at 0.0874
    # All capacity increases hurt. Overfitting is the primary bottleneck.
    (
        "anti_overfit",
        "configs/exp_anti_overfit.yaml",
        "runs/exp_anti_overfit",
        "Stronger regularization: dropout 0.7, droppath 0.5, wd 2e-3, smooth 0.2",
    ),
    (
        "strong_aug",
        "configs/exp_strong_aug.yaml",
        "runs/exp_strong_aug",
        "Stronger augmentation: 5x noise, 2x mask spans + length, 2x jitter",
    ),
    # -- Round 5: backbone/neck cross-combination (all on strong_aug training) --
    # Baseline for this round: strong_aug (MaxPool + identity + strong_aug) = 0.1143
    (
        "transformer_strong",
        "configs/exp_transformer_strong.yaml",
        "runs/exp_transformer_strong",
        "Transformer backbone + identity neck + strong_aug training",
    ),
    (
        "transformer_fpn_strong",
        "configs/exp_transformer_fpn_strong.yaml",
        "runs/exp_transformer_fpn_strong",
        "Transformer backbone + FPN neck + strong_aug training",
    ),
    (
        "sgp_strong",
        "configs/exp_sgp_strong.yaml",
        "runs/exp_sgp_strong",
        "SGP backbone + identity neck + strong_aug training",
    ),
    (
        "sgp_fpn_strong",
        "configs/exp_sgp_fpn_strong.yaml",
        "runs/exp_sgp_fpn_strong",
        "SGP backbone + FPN neck + strong_aug training",
    ),
    (
        "tridet_strong",
        "configs/exp_tridet_strong.yaml",
        "runs/exp_tridet_strong",
        "Full TriDet (SGP + trident head) + strong_aug training",
    ),
    (
        "maxpool_combined",
        "configs/exp_maxpool_combined.yaml",
        "runs/exp_maxpool_combined",
        "MaxPool + strong_aug + heavier regularization (dropout 0.6, wd 1e-3)",
    ),
    # -- Round 6: scale up winning TriDet architecture -------------------------
    # Baseline: tridet_strong = 0.1402
    (
        "tridet_fpn",
        "configs/exp_tridet_fpn.yaml",
        "runs/exp_tridet_fpn",
        "TriDet + FPN neck (FPN helped SGP standard head +0.01)",
    ),
    (
        "tridet_deep",
        "configs/exp_tridet_deep.yaml",
        "runs/exp_tridet_deep",
        "TriDet 4-level pyramid (n_layers 3->4)",
    ),
    (
        "tridet_bins32",
        "configs/exp_tridet_bins32.yaml",
        "runs/exp_tridet_bins32",
        "TriDet finer boundary distribution (num_bins 16->32)",
    ),
    (
        "tridet_wide",
        "configs/exp_tridet_wide.yaml",
        "runs/exp_tridet_wide",
        "TriDet wider (d_model 192, d_cma 128, sgp_mlp 768) ~2.1M params",
    ),
    (
        "tridet_wide_aug",
        "configs/exp_tridet_wide_aug.yaml",
        "runs/exp_tridet_wide_aug",
        "TriDet wider + heavier regularization (dropout 0.6, wd 1e-3)",
    ),
    # ── Round 7: structural refinements of the 4-level TriDet winner ──────────
    (
        "tridet_combined",
        "configs/exp_tridet_combined.yaml",
        "runs/exp_tridet_combined",
        "TriDet 4-level + num_bins=32 (combine both round-6 winners), patience=10",
    ),
    (
        "tridet_deep_k2",
        "configs/exp_tridet_deep_k2.yaml",
        "runs/exp_tridet_deep_k2",
        "TriDet 4-level + sgp_k=2.0 (wider per-level receptive field), patience=10",
    ),
    (
        "tridet_stem2",
        "configs/exp_tridet_stem2.yaml",
        "runs/exp_tridet_stem2",
        "TriDet 4-level with deeper stem (downsample_start=2, n_layers=5), patience=10",
    ),
    (
        "tridet_5level",
        "configs/exp_tridet_5level.yaml",
        "runs/exp_tridet_5level",
        "TriDet 5-level pyramid (n_layers 4->5), patience=10",
    ),
    # ── Round 8: paper-faithful settings from original repos ──────────────────
    (
        "paper_actionformer",
        "configs/exp_paper_actionformer.yaml",
        "runs/exp_paper_actionformer",
        "ActionFormer THUMOS-optimal: transformer, 6 levels, wd=0.05, radius=1.5, head 3-layer+LN",
    ),
    (
        "paper_temporalmaxer",
        "configs/exp_paper_temporalmaxer.yaml",
        "runs/exp_paper_temporalmaxer",
        "TemporalMaxer THUMOS-optimal: maxpool, 6 levels, no center_sampling, wd=0.05, 60 epochs",
    ),
    (
        "paper_tridet",
        "configs/exp_paper_tridet.yaml",
        "runs/exp_paper_tridet",
        "TriDet THUMOS-optimal: SGP k=5, iou_power=0.2, 6 levels, mlp=768, wd=0.025, 40 epochs",
    ),
    (
        "paper_tridet_full",
        "configs/exp_paper_tridet_full.yaml",
        "runs/exp_paper_tridet_full",
        "TriDet fully faithful: adds focal_alpha=0.25, gamma=2.0, nms_sigma=0.5, max_det=2000",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def best_ckpt_path(output_dir):
    return os.path.join(_SCRIPT_DIR, output_dir, "model_best.pth.tar")


def resume_ckpt_path(output_dir):
    return os.path.join(_SCRIPT_DIR, output_dir, "checkpoint.pth.tar")


def read_best_map(output_dir):
    path = best_ckpt_path(output_dir)
    try:
        ckpt = torch.load(path, map_location="cpu")
        return float(ckpt.get("best_mAP", ckpt.get("mAP", 0.0)))
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"  [warn] Could not read checkpoint {path}: {e}")
        return None


def run_training(name, config_path, output_dir, python_exe, seed=42):
    """Call train.py as a subprocess; returns True on success."""
    abs_config = os.path.join(_SCRIPT_DIR, config_path)
    abs_output = os.path.join(_SCRIPT_DIR, output_dir)
    resume_ckpt = resume_ckpt_path(output_dir)

    cmd = [
        python_exe,
        os.path.join(_SCRIPT_DIR, "train.py"),
        "--config", abs_config,
        "--output_dir", abs_output,
        "--seed", str(seed),
    ]
    if os.path.isfile(resume_ckpt):
        cmd += ["--resume", resume_ckpt]
        print(f"  -> Resuming from {resume_ckpt}")

    print(f"\n{'='*70}")
    print(f"  RUNNING: {name}")
    print(f"  Config : {config_path}")
    print(f"  Output : {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    try:
        subprocess.run(cmd, check=True, cwd=_SCRIPT_DIR)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  [ERROR] Training failed for '{name}' (exit code {e.returncode})")
        return False
    except KeyboardInterrupt:
        print(f"\n  [INTERRUPTED] Training for '{name}' was interrupted.")
        raise  # re-raise so the outer loop can catch it and print summary


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run TemporalMaxer ablation experiments")
    parser.add_argument(
        "--force", default=None, metavar="NAME",
        help="Force re-run a specific experiment by name (deletes its model_best.pth.tar)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would run without actually training",
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python executable to use for training subprocesses",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed passed to train.py",
    )
    parser.add_argument(
        "--only", default=None, metavar="NAME",
        help="Run only a single experiment by name",
    )
    return parser.parse_args()


def print_table(results):
    """Print a comparison table sorted by best_mAP descending."""
    baseline_map = next(
        (r["best_mAP"] for r in results if r["name"] == "baseline" and r["best_mAP"] is not None),
        None,
    )

    header = f"{'Experiment':<18} {'Best mAP':>10}  {'vs baseline':>12}  Description"
    print("\n" + "-" * 80)
    print(header)
    print("-" * 80)

    sorted_results = sorted(
        results,
        key=lambda r: (r["best_mAP"] is not None, r["best_mAP"] or 0.0),
        reverse=True,
    )
    for r in sorted_results:
        mAP_str = f"{r['best_mAP']:.4f}" if r["best_mAP"] is not None else "  N/A  "
        if baseline_map is not None and r["best_mAP"] is not None:
            delta = r["best_mAP"] - baseline_map
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "      ?"
        status = ""
        if r["skipped"]:
            status = " [skipped]"
        elif r["failed"]:
            status = " [FAILED]"
        print(f"  {r['name']:<16} {mAP_str:>10}  {delta_str:>12}  {r['description']}{status}")
    print("-" * 80 + "\n")


def main():
    args = parse_args()

    # Filter to a single experiment if requested
    experiments = EXPERIMENTS
    if args.only is not None:
        experiments = [e for e in EXPERIMENTS if e[0] == args.only]
        if not experiments:
            print(f"[error] No experiment named '{args.only}'. Available: "
                  + ", ".join(e[0] for e in EXPERIMENTS))
            sys.exit(1)

    # Force re-run: delete the best checkpoint for that experiment
    if args.force is not None:
        forced = [e for e in experiments if e[0] == args.force]
        if not forced:
            print(f"[error] No experiment named '{args.force}'.")
            sys.exit(1)
        path = best_ckpt_path(forced[0][2])
        if os.path.isfile(path):
            os.remove(path)
            print(f"[force] Deleted {path}")
        else:
            print(f"[force] Nothing to delete at {path}")

    results = []
    interrupted = False

    for name, config_path, output_dir, description in experiments:
        result = dict(
            name=name,
            config=config_path,
            output_dir=output_dir,
            description=description,
            best_mAP=None,
            skipped=False,
            failed=False,
        )

        # Check if already done
        existing_mAP = read_best_map(output_dir)
        if existing_mAP is not None and args.force != name:
            print(f"\n[skip] '{name}' already has model_best.pth.tar  (best_mAP={existing_mAP:.4f})")
            result["best_mAP"] = existing_mAP
            result["skipped"] = True
            results.append(result)
            continue

        if args.dry_run:
            print(f"\n[dry_run] Would train: {name} -> {config_path}")
            results.append(result)
            continue

        try:
            success = run_training(name, config_path, output_dir, args.python, seed=args.seed)
        except KeyboardInterrupt:
            interrupted = True
            # Read whatever was saved so far
            result["best_mAP"] = read_best_map(output_dir)
            result["failed"] = True
            results.append(result)
            break

        if success:
            result["best_mAP"] = read_best_map(output_dir)
            if result["best_mAP"] is None:
                print(f"  [warn] Training succeeded but no model_best.pth.tar found for '{name}'")
                result["failed"] = True
        else:
            result["failed"] = True
            result["best_mAP"] = read_best_map(output_dir)  # partial best, if any

        results.append(result)

    # ── Save results to JSON ──────────────────────────────────────────────────
    results_file = os.path.join(_SCRIPT_DIR, "runs", "experiment_results.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[results] Saved to {results_file}")

    # ── Print comparison table ────────────────────────────────────────────────
    print_table(results)

    if interrupted:
        print("[interrupted] Run again to continue remaining experiments.\n")
        sys.exit(130)


if __name__ == "__main__":
    main()
