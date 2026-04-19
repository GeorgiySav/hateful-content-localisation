"""
Smoke-test script: verify that every architecture config can perform a
forward pass and backward pass with dummy data.

Usage (from adapted_actionformer/ directory):
    python test_models.py

For each config the script:
  1. Instantiates HatefulContentLocalizer from the YAML.
  2. Runs a forward pass in training mode with dummy batch data.
  3. Calls loss.backward() and checks gradients flow.
  4. Runs a forward pass in eval mode (inference).
  5. Reports OK or the exception traceback.
"""
import os
import sys
import yaml
import traceback

import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from libs.modeling.meta_arch import HatefulContentLocalizer


# ── Config loader (mirrors train.py) ─────────────────────────────────────────

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── Dummy batch builder ───────────────────────────────────────────────────────

def make_dummy_batch(B=2, T=64, device='cpu'):
    """
    Create a minimal dummy batch matching the HateMM dataloader output format.
    T must be divisible by the model's max_div_factor (check after building model).
    """
    text  = torch.randn(B, T, 768,  device=device)
    audio = torch.randn(B, T, 1024, device=device)
    video = torch.randn(B, T, 768,  device=device)

    # All timesteps valid
    mask = torch.ones(B, T, device=device)

    # Two ground-truth hateful segments per sample (in seconds = grid coords at 1 FPS)
    segments = [
        torch.tensor([[5.0, 20.0], [30.0, 45.0]], device=device)
        for _ in range(B)
    ]
    labels = [
        torch.zeros(2, dtype=torch.long, device=device)
        for _ in range(B)
    ]

    return {
        'text_feat' : text,
        'audio_feat': audio,
        'video_feat': video,
        'mask'      : mask,
        'segments'  : segments,
        'labels'    : labels,
        'video_id'  : [f'dummy_{b}' for b in range(B)],
        'duration'  : [float(T)] * B,
    }


# ── Test runner ───────────────────────────────────────────────────────────────

def test_config(config_path, device='cpu'):
    cfg = load_config(config_path)

    # Ensure max_seq_len >= T used in dummy batch
    cfg['dataset']['max_seq_len'] = 64

    model = HatefulContentLocalizer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Training forward + backward ──────────────────────────────────────────
    model.train()
    batch = make_dummy_batch(B=2, T=64, device=device)
    losses = model(batch)
    assert 'final_loss' in losses, "Missing 'final_loss' key in loss dict"
    losses['final_loss'].backward()

    # Check at least one gradient is non-None
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters() if p.requires_grad
    )
    assert has_grad, "No gradients found after backward()"

    # ── Inference forward ────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        results = model(batch)
    assert isinstance(results, list), "Inference should return a list"
    assert len(results) == 2,         "Should have one result per batch item"
    for r in results:
        assert 'segments' in r and 'scores' in r, "Missing keys in result"

    return n_params


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    configs_dir = os.path.join(_script_dir, 'configs', 'experiments')
    configs = [
        ('concat_actionformer.yaml',          'Concat + ActionFormer (transformer + standard)'),
        ('concat_temporalmaxer.yaml',         'Concat + TemporalMaxer (MaxPool + standard)'),
        ('concat_tridet.yaml',                'Concat + TriDet (SGP + trident)'),
        ('trifuse_actionformer.yaml',         'TriFuse + ActionFormer (transformer + standard)'),
        ('trifuse_temporalmaxer.yaml',        'TriFuse + TemporalMaxer (MaxPool + standard)'),
        ('trifuse_tridet.yaml',               'TriFuse + TriDet (SGP + trident)'),
        ('trifuse_trident_actionformer.yaml', 'TriFuse + ActionFormer + trident head'),
        ('unimodal_video_actionformer.yaml',  'Unimodal video + ActionFormer'),
        ('unimodal_audio_actionformer.yaml',  'Unimodal audio + ActionFormer'),
        ('unimodal_text_actionformer.yaml',   'Unimodal text + ActionFormer'),
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running smoke tests on device: {device}\n")
    print(f"{'Config':<30} {'Description':<45} {'Params':>10}  Status")
    print('-' * 100)

    all_ok = True
    for fname, desc in configs:
        path = os.path.join(configs_dir, fname)
        if not os.path.exists(path):
            print(f"  {fname:<28} {desc:<45}  {'SKIP (not found)':>10}")
            continue
        try:
            n_params = test_config(path, device=device)
            print(f"  {fname:<28} {desc:<45} {n_params:>10,}  OK")
        except Exception:
            all_ok = False
            print(f"  {fname:<28} {desc:<45} {'':>10}  FAIL")
            traceback.print_exc()
        print()

    if all_ok:
        print("\nAll tests passed.")
    else:
        print("\nSome tests FAILED — see tracebacks above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
