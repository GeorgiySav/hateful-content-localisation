"""
Sequential experiment runner for architecture experiments on HateClipSeg.

Usage:
    python run_experiments.py

    # Run every experiment with multiple seeds to get variance:
    python run_experiments.py --seeds 42,123,2024

After all experiments finish (or are skipped) a comparison table is printed and
results are saved to runs/experiment_results.json.

Output layout (per seed):
    runs/exp/<name>/seed_<N>/model_best.pth.tar
    runs/exp/<name>/seed_<N>/checkpoint.pth.tar   (last in-progress state)

"""
import os
import sys
import json
import argparse
import statistics
import subprocess

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Experiments (name, config_path_relative_to_script_dir, output_dir, description)
EXPERIMENTS = [
    # TriFuse preprocessor experiments
    (
        "trifuse_actionformer",
        "configs/experiments/trifuse_actionformer.yaml",
        "runs/exp/trifuse_actionformer",
        "TriFuse + ActionFormer",
    ),
    (
        "trifuse_temporalmaxer",
        "configs/experiments/trifuse_temporalmaxer.yaml",
        "runs/exp/trifuse_temporalmaxer",
        "TriFuse + TemporalMaxer",
    ),
    (
        "trifuse_trident_temporalmaxer",
        "configs/experiments/trifuse_trident_temporalmaxer.yaml",
        "runs/exp/trifuse_trident_temporalmaxer",
        "TriFuse + TemporalMaxer + trident head",
    ),
    (
        "trifuse_tridet",
        "configs/experiments/trifuse_tridet.yaml",
        "runs/exp/trifuse_tridet",
        "TriFuse + TriDet",
    ),
    # Concat preprocessor baselines
    (
        "concat_actionformer",
        "configs/experiments/concat_actionformer.yaml",
        "runs/exp/concat_actionformer",
        "Concat + ActionFormer",
    ),
    (
        "concat_temporalmaxer",
        "configs/experiments/concat_temporalmaxer.yaml",
        "runs/exp/concat_temporalmaxer",
        "Concat + TemporalMaxer",
    ),
    (
        "concat_tridet",
        "configs/experiments/concat_tridet.yaml",
        "runs/exp/concat_tridet",
        "Concat + TriDet",
    ),
    # Trident head ablations
    (
        "trifuse_trident_actionformer",
        "configs/experiments/trifuse_trident_actionformer.yaml",
        "runs/exp/trifuse_trident_actionformer",
        "TriFuse + ActionFormer + trident head",
    ),
    (
        "trifuse_sgp_standard",
        "configs/experiments/trifuse_sgp_standard.yaml",
        "runs/exp/trifuse_sgp_standard",
        "TriFuse + SGP + standard head",
    ),
    (
        "concat_trident_actionformer",
        "configs/experiments/concat_trident_actionformer.yaml",
        "runs/exp/concat_trident_actionformer",
        "Concat + ActionFormer + trident head",
    ),
    (
        "concat_trident_temporalmaxer",
        "configs/experiments/concat_trident_temporalmaxer.yaml",
        "runs/exp/concat_trident_temporalmaxer",
        "Concat + TemporalMaxer + trident head",
    ),
    (
        "concat_sgp_standard",
        "configs/experiments/concat_sgp_standard.yaml",
        "runs/exp/concat_sgp_standard",
        "Concat + SGP + standard head",
    ),
    # Bimodal ablations
    (
        "bimodal_va_actionformer",
        "configs/experiments/bimodal_va_actionformer.yaml",
        "runs/exp/bimodal_va_actionformer",
        "(Video, Audio) + ActionFormer",
    ),
    (
        "bimodal_at_actionformer",
        "configs/experiments/bimodal_at_actionformer.yaml",
        "runs/exp/bimodal_at_actionformer",
        "(Audio, Text) + ActionFormer",
    ),
    (
        "bimodal_vt_actionformer",
        "configs/experiments/bimodal_vt_actionformer.yaml",
        "runs/exp/bimodal_vt_actionformer",
        "(Video, Text) + ActionFormer",
    ),
    # FPS ablations
    (
        "fps_ablation_2fps",
        "configs/experiments/fps_ablation_2fps.yaml",
        "runs/exp/fps_ablation_2fps",
        "TriFuse + SGP + Trident @ 2fps",
    ),
    (
        "fps_ablation_4fps",
        "configs/experiments/fps_ablation_4fps.yaml",
        "runs/exp/fps_ablation_4fps",
        "TriFuse + SGP + Trident @ 4fps",
    ),
    # Unimodal experiments
    (
        "unimodal_video_actionformer",
        "configs/experiments/unimodal_video_actionformer.yaml",
        "runs/exp/unimodal_video_actionformer",
        "Video + ActionFormer",
    ),
    (
        "unimodal_audio_actionformer",
        "configs/experiments/unimodal_audio_actionformer.yaml",
        "runs/exp/unimodal_audio_actionformer",
        "Audio + ActionFormer",
    ),
    (
        "unimodal_text_actionformer",
        "configs/experiments/unimodal_text_actionformer.yaml",
        "runs/exp/unimodal_text_actionformer",
        "Text + ActionFormer",
    ),
]


def _seed_dir(output_dir, seed):
    return os.path.join(_SCRIPT_DIR, output_dir, f"seed_{seed}")


def best_ckpt_path(output_dir, seed):
    return os.path.join(_seed_dir(output_dir, seed), "model_best.pth.tar")


def resume_ckpt_path(output_dir, seed):
    return os.path.join(_seed_dir(output_dir, seed), "checkpoint.pth.tar")


def read_best_map(output_dir, seed):
    """Return (best_mAP, mAP_per_tiou, tiou_thresholds) from model_best.pth.tar.
    Returns (None, [], []) if the checkpoint does not exist or cannot be read.
    """
    path = best_ckpt_path(output_dir, seed)
    try:
        ckpt = torch.load(path, map_location="cpu")
        mAP = float(ckpt.get("best_mAP", ckpt.get("mAP", 0.0)))
        per_tiou   = ckpt.get("best_mAP_per_tiou", [])
        thresholds = ckpt.get("tiou_thresholds", [])
        return mAP, per_tiou, thresholds
    except FileNotFoundError:
        return None, [], []
    except Exception as e:
        print(f"  [warn] Could not read checkpoint {path}: {e}")
        return None, [], []


def run_training(name, config_path, output_dir, seed, python_exe):
    """Call train.py as a subprocess for one experiment."""
    abs_config = os.path.join(_SCRIPT_DIR, config_path)
    abs_output = _seed_dir(output_dir, seed)
    resume_ckpt = resume_ckpt_path(output_dir, seed)

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
    print(f"  RUNNING: {name}  (seed={seed})")
    print(f"  Config : {config_path}")
    print(f"  Output : {os.path.relpath(abs_output, _SCRIPT_DIR)}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    try:
        subprocess.run(cmd, check=True, cwd=_SCRIPT_DIR)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  [ERROR] Training failed for '{name}' seed={seed} (exit code {e.returncode})")
        return False
    except KeyboardInterrupt:
        print(f"\n  [INTERRUPTED] Training for '{name}' seed={seed} was interrupted.")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run TAL experiments on HateClipSeg")
    parser.add_argument(
        "--seeds", default="42",
        help="Comma-separated list of random seeds; each experiment is run once per seed (default: '42')",
    ) 
    return parser.parse_args()


def parse_seeds(seeds_str):
    """Parse a comma-separated string of integers, e.g. '42,123,2024' -> [42, 123, 2024]."""
    seeds = []
    for tok in seeds_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            seeds.append(int(tok))
        except ValueError:
            raise SystemExit(f"[error] --seeds must be comma-separated integers (got '{tok}')")
    if not seeds:
        raise SystemExit("[error] --seeds must contain at least one integer")
    return seeds


def _aggregate_group(group):
    """Compute (mean_mAP, std_mAP, n_ok, n_total, per_tiou_mean, thresholds) for a list of seed records."""
    valid = [r for r in group if r["best_mAP"] is not None]
    n_total = len(group)
    n_ok = len(valid)
    if not valid:
        return None, None, n_ok, n_total, [], []
    mAPs = [r["best_mAP"] for r in valid]
    mean_mAP = statistics.mean(mAPs)
    std_mAP = statistics.stdev(mAPs) if len(mAPs) >= 2 else None

    thresholds = valid[0].get("tiou_thresholds") or []
    per_tiou_mean = []
    if thresholds:
        for i in range(len(thresholds)):
            vals = [
                r["best_mAP_per_tiou"][i]
                for r in valid
                if r.get("tiou_thresholds") == thresholds
                and len(r.get("best_mAP_per_tiou", [])) > i
            ]
            per_tiou_mean.append(statistics.mean(vals) if vals else None)
    return mean_mAP, std_mAP, n_ok, n_total, per_tiou_mean, thresholds


def print_table(results):
    """Print comparison table: aggregate row per experiment, then per-seed rows."""
    # Group by experiment name, preserving first-seen order
    groups = {}  # name -> list[result]
    order = []
    for r in results:
        if r["name"] not in groups:
            groups[r["name"]] = []
            order.append(r["name"])
        groups[r["name"]].append(r)

    # Union of tIoU thresholds across all results, for column headers
    all_thresholds = []
    for r in results:
        for t in r.get("tiou_thresholds", []):
            if t not in all_thresholds:
                all_thresholds.append(t)
    all_thresholds = sorted(all_thresholds)

    tiou_header = "  ".join(f"@{t:.1f}" for t in all_thresholds)
    name_w = 28
    mAP_w = 16  # wide enough for "0.1234+/-0.1234"
    sep_width = name_w + (len(tiou_header) + 2 if all_thresholds else 0) + mAP_w + 50

    header = (
        f"  {'Experiment':<{name_w}}"
        + (f"  {tiou_header}" if all_thresholds else "")
        + f"  {'mAP':>{mAP_w}}  Description"
    )
    print("\n" + "-" * sep_width)
    print(header)
    print("-" * sep_width)

    # Sort by aggregate mean mAP descending
    def sort_key(name):
        m, *_ = _aggregate_group(groups[name])
        return (m is not None, m or 0.0)
    sorted_names = sorted(order, key=sort_key, reverse=True)

    for name in sorted_names:
        rs = groups[name]
        mean_mAP, std_mAP, n_ok, n_total, per_tiou_mean, thresholds = _aggregate_group(rs)
        description = rs[0]["description"]

        # Aggregate row
        if mean_mAP is None:
            mAP_str = "N/A"
        else:
            mAP_str = (
                f"{mean_mAP:.4f}+/-{std_mAP:.4f}" if std_mAP is not None
                else f"{mean_mAP:.4f}"
            )

        if all_thresholds:
            per_tiou_map = dict(zip(thresholds, per_tiou_mean))
            tiou_str = "  ".join(
                f"{per_tiou_map[t]:.4f}" if t in per_tiou_map and per_tiou_map[t] is not None else "  N/A"
                for t in all_thresholds
            )
            row = f"  {name:<{name_w}}  {tiou_str}  {mAP_str:>{mAP_w}}  {description} [n={n_ok}/{n_total}]"
        else:
            row = f"  {name:<{name_w}}  {mAP_str:>{mAP_w}}  {description} [n={n_ok}/{n_total}]"
        print(row)

        # Per-seed rows
        if len(rs) > 1:
            for r in rs:
                tag = f"seed={r['seed']}"
                if r["skipped"]:
                    tag += " [skipped]"
                elif r["failed"]:
                    tag += " [FAILED]"
                seed_label = f"      {tag}"
                seed_mAP_str = f"{r['best_mAP']:.4f}" if r["best_mAP"] is not None else "N/A"
                if all_thresholds:
                    per_tiou_map = dict(zip(r.get("tiou_thresholds", []), r.get("best_mAP_per_tiou", [])))
                    tiou_str = "  ".join(
                        f"{per_tiou_map[t]:.4f}" if t in per_tiou_map else "  N/A"
                        for t in all_thresholds
                    )
                    print(f"  {seed_label:<{name_w}}  {tiou_str}  {seed_mAP_str:>{mAP_w}}")
                else:
                    print(f"  {seed_label:<{name_w}}  {seed_mAP_str:>{mAP_w}}")
    print("-" * sep_width + "\n")


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    experiments = EXPERIMENTS
    results = []
    interrupted = False

    for name, config_path, output_dir, description in experiments:
        if interrupted:
            break
        for seed in seeds:
            result = dict(
                name=name,
                seed=seed,
                config=config_path,
                output_dir=output_dir,
                description=description,
                best_mAP=None,
                best_mAP_per_tiou=[],
                tiou_thresholds=[],
                skipped=False,
                failed=False,
            )

            existing_mAP, existing_per_tiou, existing_thresholds = read_best_map(output_dir, seed)
            if existing_mAP is not None:
                print(f"\n[skip] '{name}' seed={seed} already has model_best.pth.tar  (best_mAP={existing_mAP:.4f})")
                result["best_mAP"]          = existing_mAP
                result["best_mAP_per_tiou"] = existing_per_tiou
                result["tiou_thresholds"]   = existing_thresholds
                result["skipped"] = True
                results.append(result)
                continue

            try:
                success = run_training(name, config_path, output_dir, seed, sys.executable)
            except KeyboardInterrupt:
                interrupted = True
                mAP, per_tiou, thresholds = read_best_map(output_dir, seed)
                result["best_mAP"]          = mAP
                result["best_mAP_per_tiou"] = per_tiou
                result["tiou_thresholds"]   = thresholds
                result["failed"] = True
                results.append(result)
                break

            mAP, per_tiou, thresholds = read_best_map(output_dir, seed)
            result["best_mAP"]          = mAP
            result["best_mAP_per_tiou"] = per_tiou
            result["tiou_thresholds"]   = thresholds
            if success:
                if result["best_mAP"] is None:
                    print(f"  [warn] Training succeeded but no model_best.pth.tar found for '{name}' seed={seed}")
                    result["failed"] = True
            else:
                result["failed"] = True

            results.append(result)

    # Save results to JSON
    results_file = os.path.join(_SCRIPT_DIR, "runs", "experiment_results.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[results] Saved to {results_file}")

    print_table(results)

    if interrupted:
        print("[interrupted] Run again to continue remaining experiments.\n")
        sys.exit(130)


if __name__ == "__main__":
    main()
