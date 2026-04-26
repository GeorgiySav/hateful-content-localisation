"""
Sequential experiment runner for architecture experiments on HateClipSeg.

Usage:
    # Run all experiments with the default seed (42), skipping any already done:
    python run_experiments.py

    # Run every experiment with multiple seeds for variance reporting:
    python run_experiments.py --seeds 42,123,2024

    # Force re-run a specific experiment (deletes all its seed checkpoints):
    python run_experiments.py --force trifuse_actionformer

    # Dry-run: print what would run without training:
    python run_experiments.py --dry_run

    # Use a specific Python executable:
    python run_experiments.py --python C:/Users/Georgiy/anaconda3/envs/hcl/python.exe

After all experiments finish (or are skipped) a comparison table is printed and
results are saved to runs/experiment_results.json.

Output layout (per seed):
    runs/exp/<name>/seed_<N>/model_best.pth.tar
    runs/exp/<name>/seed_<N>/checkpoint.pth.tar   (last in-progress state)

Skip / resume logic is per (experiment, seed):
    * If <output_dir>/seed_<N>/model_best.pth.tar exists for a seed, that seed
      is skipped and best_mAP is read from the checkpoint.
    * If <output_dir>/seed_<N>/checkpoint.pth.tar exists (mid-run), that seed
      resumes via --resume <checkpoint>.
    * --force NAME deletes both files for every seed of experiment NAME.

Note on what varies between seeds:
    --seed only perturbs model init, dropout, data-order, and augmentation RNG.
    The train/val/test split is governed by the separate `split_seed` field in
    the dataset config and is held constant across all seed runs (so variance
    reflects model stochasticity on a fixed split, not split variance).
"""
import os
import sys
import json
import argparse
import statistics
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
    # ── Trident head ablations ────────────────────────────────────────────────
    (
        "trifuse_trident_actionformer",
        "configs/experiments/trifuse_trident_actionformer.yaml",
        "runs/exp/trifuse_trident_actionformer",
        "TriFuse + ActionFormer + trident head (vs standard in trifuse_actionformer)",
    ),
    (
        "trifuse_sgp_standard",
        "configs/experiments/trifuse_sgp_standard.yaml",
        "runs/exp/trifuse_sgp_standard",
        "TriFuse + SGP + standard head (vs trident in trifuse_tridet)",
    ),
    (
        "concat_trident_actionformer",
        "configs/experiments/concat_trident_actionformer.yaml",
        "runs/exp/concat_trident_actionformer",
        "Concat + ActionFormer + trident head (vs standard in concat_actionformer)",
    ),
    (
        "concat_trident_temporalmaxer",
        "configs/experiments/concat_trident_temporalmaxer.yaml",
        "runs/exp/concat_trident_temporalmaxer",
        "Concat + TemporalMaxer + trident head (vs standard in concat_temporalmaxer)",
    ),
    (
        "concat_sgp_standard",
        "configs/experiments/concat_sgp_standard.yaml",
        "runs/exp/concat_sgp_standard",
        "Concat + SGP + standard head (vs trident in concat_tridet)",
    ),
    # ── Bimodal ablations ────────────────────────────────────────────────────
    (
        "bimodal_va_actionformer",
        "configs/experiments/bimodal_va_actionformer.yaml",
        "runs/exp/bimodal_va_actionformer",
        "Video+Audio + ActionFormer (bimodal ablation, no text)",
    ),
    (
        "bimodal_at_actionformer",
        "configs/experiments/bimodal_at_actionformer.yaml",
        "runs/exp/bimodal_at_actionformer",
        "Audio+Text + ActionFormer (bimodal ablation, no video)",
    ),
    (
        "bimodal_vt_actionformer",
        "configs/experiments/bimodal_vt_actionformer.yaml",
        "runs/exp/bimodal_vt_actionformer",
        "Video+Text + ActionFormer (bimodal ablation, no audio)",
    ),
    # ── FPS ablations ────────────────────────────────────────────────────────
    (
        "fps_ablation_2fps",
        "configs/experiments/fps_ablation_2fps.yaml",
        "runs/exp/fps_ablation_2fps",
        "TriFuse + SGP + Trident @ 2fps (vs trifuse_tridet @ 1fps)",
    ),
    (
        "fps_ablation_4fps",
        "configs/experiments/fps_ablation_4fps.yaml",
        "runs/exp/fps_ablation_4fps",
        "TriFuse + SGP + Trident @ 4fps (vs trifuse_tridet @ 1fps)",
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


def run_evaluation(name, config_path, output_dir, seed, python_exe):
    """Call eval.py to compute per-threshold mAP and patch it into model_best.pth.tar."""
    abs_config = os.path.join(_SCRIPT_DIR, config_path)
    abs_ckpt   = best_ckpt_path(output_dir, seed)
    cmd = [
        python_exe,
        os.path.join(_SCRIPT_DIR, "eval.py"),
        "--config",     abs_config,
        "--checkpoint", abs_ckpt,
        "--patch_checkpoint",
    ]
    print(f"\n[eval] '{name}' seed={seed} missing per-threshold data — re-evaluating checkpoint ...")
    print(f"  Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=_SCRIPT_DIR)
    except subprocess.CalledProcessError as e:
        print(f"  [warn] Evaluation failed for '{name}' seed={seed} (exit code {e.returncode})")


def run_training(name, config_path, output_dir, seed, python_exe):
    """Call train.py as a subprocess for one (experiment, seed) pair; returns True on success."""
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
        raise  # re-raise so the outer loop can catch it and print summary


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run TriFuse preprocessor experiments on HateClipSeg")
    parser.add_argument(
        "--force", default=None, metavar="NAME",
        help="Force re-run a specific experiment by name (deletes its seed_*/model_best.pth.tar files)",
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
        "--seeds", default="42",
        help="Comma-separated list of random seeds; each experiment is run once per seed (default: '42')",
    )
    parser.add_argument(
        "--only", default=None, metavar="NAME",
        help="Run only a single experiment by name",
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

    # Baseline mean for the "vs base" delta column
    baseline_mean = None
    if "trifuse_actionformer" in groups:
        baseline_mean, *_ = _aggregate_group(groups["trifuse_actionformer"])

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
    delta_w = 9
    sep_width = name_w + (len(tiou_header) + 2 if all_thresholds else 0) + mAP_w + delta_w + 50

    header = (
        f"  {'Experiment':<{name_w}}"
        + (f"  {tiou_header}" if all_thresholds else "")
        + f"  {'mAP':>{mAP_w}}  {'vs base':>{delta_w}}  Description"
    )
    print("\n" + "-" * sep_width)
    print(header)
    print("-" * sep_width)

    # Sort by aggregate mean mAP descending (None last)
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
            delta_str = "?"
        else:
            mAP_str = (
                f"{mean_mAP:.4f}+/-{std_mAP:.4f}" if std_mAP is not None
                else f"{mean_mAP:.4f}"
            )
            delta_str = (
                f"{mean_mAP - baseline_mean:+.4f}"
                if baseline_mean is not None else "?"
            )

        if all_thresholds:
            per_tiou_map = dict(zip(thresholds, per_tiou_mean))
            tiou_str = "  ".join(
                f"{per_tiou_map[t]:.4f}" if t in per_tiou_map and per_tiou_map[t] is not None else "  N/A"
                for t in all_thresholds
            )
            row = f"  {name:<{name_w}}  {tiou_str}  {mAP_str:>{mAP_w}}  {delta_str:>{delta_w}}  {description} [n={n_ok}/{n_total}]"
        else:
            row = f"  {name:<{name_w}}  {mAP_str:>{mAP_w}}  {delta_str:>{delta_w}}  {description} [n={n_ok}/{n_total}]"
        print(row)

        # Per-seed rows (only if more than one seed, otherwise the aggregate IS the seed)
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
                    print(f"  {seed_label:<{name_w}}  {tiou_str}  {seed_mAP_str:>{mAP_w}}  {'':>{delta_w}}")
                else:
                    print(f"  {seed_label:<{name_w}}  {seed_mAP_str:>{mAP_w}}  {'':>{delta_w}}")
    print("-" * sep_width + "\n")


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    # Filter to a single experiment if requested
    experiments = EXPERIMENTS
    if args.only is not None:
        experiments = [e for e in EXPERIMENTS if e[0] == args.only]
        if not experiments:
            print(f"[error] No experiment named '{args.only}'. Available: "
                  + ", ".join(e[0] for e in EXPERIMENTS))
            sys.exit(1)

    # Force re-run: delete every seed's checkpoints for that experiment so no
    # stale weights are resumed (important when the architecture changes).
    if args.force is not None:
        forced = [e for e in experiments if e[0] == args.force]
        if not forced:
            print(f"[error] No experiment named '{args.force}'.")
            sys.exit(1)
        for seed in seeds:
            for ckpt_path in (best_ckpt_path(forced[0][2], seed),
                              resume_ckpt_path(forced[0][2], seed)):
                if os.path.isfile(ckpt_path):
                    os.remove(ckpt_path)
                    print(f"[force] Deleted {ckpt_path}")
                else:
                    print(f"[force] Nothing to delete at {ckpt_path}")

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
            if existing_mAP is not None and args.force != name:
                if not existing_per_tiou and not args.dry_run:
                    run_evaluation(name, config_path, output_dir, seed, args.python)
                    existing_mAP, existing_per_tiou, existing_thresholds = read_best_map(output_dir, seed)
                print(f"\n[skip] '{name}' seed={seed} already has model_best.pth.tar  (best_mAP={existing_mAP:.4f})")
                result["best_mAP"]          = existing_mAP
                result["best_mAP_per_tiou"] = existing_per_tiou
                result["tiou_thresholds"]   = existing_thresholds
                result["skipped"] = True
                results.append(result)
                continue

            if args.dry_run:
                rel_out = os.path.relpath(_seed_dir(output_dir, seed), _SCRIPT_DIR)
                print(f"\n[dry_run] Would train: {name} seed={seed} -> {config_path}  (out: {rel_out})")
                results.append(result)
                continue

            try:
                success = run_training(name, config_path, output_dir, seed, args.python)
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
