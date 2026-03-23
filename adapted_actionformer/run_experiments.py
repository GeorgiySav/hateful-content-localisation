"""
Sequential experiment runner for architecture experiments on HateClipSeg.

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
    # ── TriFuse preprocessor experiments ──────────────────────────────────────
    (
        "trifuse_actionformer",
        "configs/experiments/trifuse_actionformer.yaml",
        "runs/exp/trifuse_actionformer",
        "TriFuse + ActionFormer (windowed self-attention backbone)",
    ),
    (
        "trifuse_temporalmaxer",
        "configs/experiments/trifuse_temporalmaxer.yaml",
        "runs/exp/trifuse_temporalmaxer",
        "TriFuse + TemporalMaxer (parameter-free MaxPool backbone)",
    ),
    (
        "trifuse_trident_temporalmaxer",
        "configs/experiments/trifuse_trident_temporalmaxer.yaml",
        "runs/exp/trifuse_trident_temporalmaxer",
        "TriFuse + TemporalMaxer + trident head (boundary distribution)",
    ),
    (
        "trifuse_tridet",
        "configs/experiments/trifuse_tridet.yaml",
        "runs/exp/trifuse_tridet",
        "TriFuse + TriDet (SGP backbone + trident distribution head)",
    ),
    # ── Concat preprocessor baselines ─────────────────────────────────────────
    (
        "concat_actionformer",
        "configs/experiments/concat_actionformer.yaml",
        "runs/exp/concat_actionformer",
        "Concat + ActionFormer (baseline for trifuse_actionformer)",
    ),
    (
        "concat_temporalmaxer",
        "configs/experiments/concat_temporalmaxer.yaml",
        "runs/exp/concat_temporalmaxer",
        "Concat + TemporalMaxer (baseline for trifuse_temporalmaxer)",
    ),
    (
        "concat_tridet",
        "configs/experiments/concat_tridet.yaml",
        "runs/exp/concat_tridet",
        "Concat + TriDet (baseline for trifuse_tridet)",
    ),
    # ── Unimodal experiments ─────────────────────────────────────────────────
    (
        "unimodal_video_actionformer",
        "configs/experiments/unimodal_video_actionformer.yaml",
        "runs/exp/unimodal_video_actionformer",
        "Video-only + ActionFormer (CLIP ViT-L/14 unimodal ablation)",
    ),
    (
        "unimodal_audio_actionformer",
        "configs/experiments/unimodal_audio_actionformer.yaml",
        "runs/exp/unimodal_audio_actionformer",
        "Audio-only + ActionFormer (Wav2Vec2 Large unimodal ablation)",
    ),
    (
        "unimodal_text_actionformer",
        "configs/experiments/unimodal_text_actionformer.yaml",
        "runs/exp/unimodal_text_actionformer",
        "Text-only + ActionFormer (HateBERT CLS unimodal ablation)",
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
    parser = argparse.ArgumentParser(description="Run TriFuse preprocessor experiments on HateClipSeg")
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
        (r["best_mAP"] for r in results if r["name"] == "trifuse_actionformer" and r["best_mAP"] is not None),
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
