# Claude Code Prompt: Integrate TemporalMaxer & TriDet into Hateful Content Localisation Repo

## Objective

Update the `hateful-content-localisation` repository to support configurable backbone and neck architectures from TemporalMaxer and TriDet, in addition to the existing model. The goal is to allow switching between different temporal action localisation (TAL) architectures via YAML config files, enabling experimentation with different approaches for hateful content temporal localisation.

---

## Context & Background

This repo trains temporal action localisation models for hateful content localisation in video. We want to integrate components from two additional TAL architectures so we can experiment with them. All three architectures (ActionFormer-style, TemporalMaxer, TriDet) share the same high-level TAL pipeline:

```
Input Features → Projection → Backbone (temporal context modeling) → Neck (feature pyramid) → Head (classification + regression)
```

The key differences between the architectures are:

### 1. ActionFormer (current/baseline)
- **Backbone**: Multi-head self-attention transformer blocks with local windowed attention
- **Neck**: Standard Feature Pyramid Network (FPN) with strided convolution downsampling
- **Head**: Standard classification head + offset regression head

### 2. TemporalMaxer (https://github.com/TuanTNG/TemporalMaxer)
- **Backbone**: Parameter-free MaxPool1D blocks. Replaces self-attention with local-region max pooling (`nn.MaxPool1d(kernel_size=k, stride=2, padding=w)`). This is the simplest possible temporal context modeling — no learnable parameters in the backbone at all. Key details:
  - Uses `nn.MaxPool1d` with configurable kernel size, stride=2, appropriate padding
  - Operates on each pyramid level independently
  - Results in 2.8x fewer GMACs and 3x faster inference vs ActionFormer
  - The backbone block is essentially just: `MaxPool1d → LayerNorm → FeedForward`
- **Neck**: Same FPN structure as ActionFormer
- **Head**: Same classification + regression head as ActionFormer

### 3. TriDet (https://github.com/dingfengshi/TriDet, CVPR 2023)
- **Backbone**: Scalable-Granularity Perception (SGP) layer. Replaces self-attention with a dual-branch convolutional structure:
  - **Instant-level branch**: Depthwise separable 1D convolution (pointwise conv → depthwise conv → pointwise conv) for per-instant feature enhancement
  - **Window-level branch**: Depthwise convolution with larger kernel for multi-scale temporal aggregation across different granularities
  - Both branches use Group Normalization (GN) instead of Layer Normalization
  - The SGP layer resolves the "rank loss problem" where self-attention causes features at different time steps to become too similar
  - Macro-architecture: same pre-norm residual structure as transformer (LayerNorm → SGP → residual → LayerNorm → FFN → residual)
- **Neck**: Same FPN structure but uses max-pooling with stride 2 for downsampling between pyramid levels
- **Head**: Novel **Trident-head** for boundary modeling:
  - Instead of directly regressing boundary offsets, it estimates a **relative probability distribution** around each boundary
  - Three branches: Start Boundary head, End Boundary head, Center Offset head
  - Each boundary branch predicts a probability distribution over neighboring temporal bins
  - The boundary offset is computed as the expected value of the distribution
  - This improves localization of ambiguous action boundaries

---

## Step-by-Step Instructions

### Phase 1: Understand the Existing Codebase

1. **Clone and examine the existing repo structure:**
   ```
   Read through the full directory tree of the hateful-content-localisation repo.
   ```

2. **Identify the current model architecture files.** Look for:
   - The model definition (likely in a `models/` or `libs/modeling/` directory)
   - The config/YAML files that define model hyperparameters
   - The backbone implementation (likely transformer/self-attention blocks)
   - The neck/FPN implementation
   - The detection head implementation
   - The model builder/factory that constructs the full model from config
   - The training script and how it loads configs

3. **Map the existing architecture to the backbone → neck → head pattern.** Document which files contain:
   - Feature projection layer
   - Backbone blocks (temporal context modeling layers)
   - Neck (feature pyramid network)
   - Detection head (classification + regression)
   - Config parsing and model construction

### Phase 2: Clone and Study Reference Implementations

4. **Clone TemporalMaxer and TriDet repos** into a temporary reference directory:
   ```bash
   git clone https://github.com/TuanTNG/TemporalMaxer.git /tmp/ref/TemporalMaxer
   git clone https://github.com/dingfengshi/TriDet.git /tmp/ref/TriDet
   ```

5. **Study the TemporalMaxer backbone implementation.** Key files to examine:
   - `libs/modeling/blocks.py` — contains the MaxPool backbone block
   - `libs/modeling/backbones.py` — contains the full backbone module
   - `libs/modeling/meta_archs.py` — contains the full model architecture and how backbone/neck/head are composed
   - `configs/temporalmaxer_thumos_i3d.yaml` — example config showing backbone parameters
   - Pay special attention to:
     - How the MaxPool1D block replaces self-attention
     - The kernel size, stride, and padding configuration
     - How the FPN levels are constructed
     - Any differences in the projection layer

6. **Study the TriDet SGP layer and Trident-head implementation.** Key files:
   - `libs/modeling/blocks.py` — contains the SGP layer implementation
   - `libs/modeling/backbones.py` — SGP-based backbone
   - `libs/modeling/meta_archs.py` — full model with Trident-head
   - `configs/` — YAML configs for different datasets
   - Pay special attention to:
     - The SGP layer's dual-branch structure (instant-level + window-level)
     - How Group Normalization is used
     - The depthwise separable convolution configuration
     - The Trident-head's three branches and how they compute boundary distributions
     - The relative probability distribution mechanism
     - Any additional loss functions required by the Trident-head

### Phase 3: Design the Modular Architecture

7. **Create a backbone registry/factory.** Design a system where backbones are registered and selected by name from config. The config should support at minimum:

   ```yaml
   model:
     backbone:
       type: "transformer"  # or "temporalmaxer" or "sgp"
       # Common parameters
       n_embd: 512
       n_layers: 6          # number of backbone layers (applied at each pyramid level)
       # Transformer-specific
       n_head: 8
       window_size: 19
       # TemporalMaxer-specific
       pool_kernel_size: 3
       # SGP-specific
       sgp_mlp_groups: 4    # groups for depthwise conv
       sgp_kernel_size: 3   # instant-level conv kernel
       sgp_window_sizes: [5, 9, 17]  # window-level multi-scale kernels
     neck:
       type: "fpn"           # standard FPN (shared across all)
       n_levels: 6           # number of pyramid levels
       downsample_type: "conv"  # or "maxpool" (TriDet uses maxpool)
       scale_factor: 2
     head:
       type: "standard"      # or "trident"
       # Standard head params
       n_head_layers: 3
       # Trident-head params
       num_bins: 16          # number of bins for boundary distribution
       trident_head_layers: 3
   ```

8. **Plan the file structure.** Create new files as needed. A recommended structure:

   ```
   libs/modeling/
   ├── backbones/
   │   ├── __init__.py          # registry + factory function
   │   ├── transformer.py       # existing transformer backbone
   │   ├── temporal_maxer.py    # MaxPool backbone from TemporalMaxer
   │   └── sgp.py               # SGP backbone from TriDet
   ├── necks/
   │   ├── __init__.py          # registry + factory
   │   └── fpn.py               # FPN (unified, with configurable downsampling)
   ├── heads/
   │   ├── __init__.py          # registry + factory
   │   ├── standard_head.py     # existing cls + reg head
   │   └── trident_head.py      # TriDet's Trident-head
   └── meta_archs.py            # updated to use factories
   ```

   Adapt the structure above to match the existing codebase conventions. If the current repo uses a flat structure (everything in one file), refactor minimally — extract only what's needed.

### Phase 4: Implement the TemporalMaxer Backbone

9. **Implement the MaxPool backbone block.** The core of TemporalMaxer is remarkably simple:

   ```python
   # Pseudocode — adapt to match the existing codebase's conventions
   class MaxPoolBlock(nn.Module):
       """Single MaxPool temporal context modeling block."""
       def __init__(self, n_embd, kernel_size=3, padding=1):
           self.pool = nn.MaxPool1d(kernel_size, stride=1, padding=padding)
           self.norm = nn.LayerNorm(n_embd)
           self.ffn = FFN(n_embd)  # feed-forward network (same as transformer FFN)
       
       def forward(self, x):
           # x shape: (B, C, T)
           residual = x
           x = self.pool(x)
           x = x.transpose(1, 2)  # (B, T, C) for LayerNorm
           x = self.norm(x)
           x = x.transpose(1, 2)  # back to (B, C, T)
           x = residual + x
           # FFN with residual
           residual = x
           x = self.ffn(x)
           x = residual + x
           return x
   ```

   **Important**: Refer to the actual TemporalMaxer source code for the exact implementation. The above is a simplified guide. The actual implementation may:
   - Use different normalization placement (pre-norm vs post-norm)
   - Have a different FFN structure
   - Handle masking for variable-length sequences
   - Use different pooling configurations at different pyramid levels

10. **Wire it into the backbone factory** so that setting `backbone.type: "temporalmaxer"` in config instantiates the MaxPool backbone.

### Phase 5: Implement the TriDet SGP Backbone

11. **Implement the SGP layer.** The SGP layer has two branches:

    ```python
    # Pseudocode — refer to TriDet source for exact implementation
    class SGPBlock(nn.Module):
        """Scalable-Granularity Perception layer from TriDet."""
        def __init__(self, n_embd, kernel_size=3, window_sizes=[5, 9], n_groups=4):
            # Instant-level branch: depthwise separable conv
            self.instant_branch = nn.Sequential(
                nn.Conv1d(n_embd, n_embd, 1),          # pointwise
                nn.Conv1d(n_embd, n_embd, kernel_size,  # depthwise
                          padding=kernel_size//2, groups=n_embd),
                nn.GroupNorm(n_groups, n_embd),
                nn.Conv1d(n_embd, n_embd, 1),           # pointwise
            )
            
            # Window-level branch: multi-scale depthwise convs
            self.window_branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(n_embd, n_embd, ws, padding=ws//2, groups=n_embd),
                    nn.GroupNorm(n_groups, n_embd),
                )
                for ws in window_sizes
            ])
            
            self.norm = nn.GroupNorm(n_groups, n_embd)
            self.ffn = FFN(n_embd)
        
        def forward(self, x):
            # x shape: (B, C, T)
            residual = x
            instant_out = self.instant_branch(x)
            window_out = sum(branch(x) for branch in self.window_branches)
            x = self.norm(instant_out + window_out)
            x = residual + x
            # FFN with residual
            residual = x
            x = self.ffn(x)
            x = residual + x
            return x
    ```

    **Important**: The pseudocode above is simplified. Refer to the actual TriDet source for:
    - Exact normalization placement and type
    - Activation functions used
    - How the two branches are combined (addition, concatenation, gating, etc.)
    - Any dropout or regularization
    - Masking for variable-length sequences
    - The scaling factor between branches

12. **Wire the SGP backbone into the factory** for `backbone.type: "sgp"`.

### Phase 6: Implement the Trident-Head

13. **Implement the Trident-head from TriDet.** This is the most complex new component:

    ```python
    # Pseudocode — refer to TriDet source for exact implementation
    class TridentHead(nn.Module):
        """Trident-head for relative boundary modeling from TriDet."""
        def __init__(self, n_embd, num_classes, num_bins=16, n_layers=3):
            # Classification branch (same as standard)
            self.cls_head = build_cls_head(n_embd, num_classes, n_layers)
            
            # Three boundary branches
            self.start_head = build_boundary_branch(n_embd, num_bins, n_layers)
            self.end_head = build_boundary_branch(n_embd, num_bins, n_layers)
            self.center_head = build_center_branch(n_embd, n_layers)
        
        def forward(self, fpn_features):
            outputs = []
            for feat in fpn_features:
                cls_out = self.cls_head(feat)
                start_dist = self.start_head(feat)  # (B, num_bins, T)
                end_dist = self.end_head(feat)       # (B, num_bins, T)
                center_off = self.center_head(feat)  # (B, 1, T)
                
                # Convert distributions to offsets via expected value
                start_offset = self.dist_to_offset(start_dist)
                end_offset = self.dist_to_offset(end_dist)
                
                outputs.append((cls_out, start_offset + center_off, end_offset + center_off))
            return outputs
        
        def dist_to_offset(self, dist):
            """Convert probability distribution over bins to expected offset."""
            dist = F.softmax(dist, dim=1)
            bin_centers = ...  # precomputed bin center positions
            return (dist * bin_centers).sum(dim=1, keepdim=True)
    ```

    **Critical**: The Trident-head requires changes to the loss function as well. Study the TriDet loss implementation carefully:
    - It may use a distribution-based loss (e.g., cross-entropy on the boundary distributions)
    - The bin configuration and offset computation must match the loss exactly
    - There may be additional regularization losses

14. **Wire into the head factory** for `head.type: "trident"`.

15. **Update the loss computation** to support both standard regression loss and the Trident distribution loss. Make this configurable.

### Phase 7: Update the Neck / FPN

16. **Make the FPN downsampling configurable.** The current FPN likely uses strided convolution for downsampling. TriDet uses max-pooling instead. Add a config option:

    ```yaml
    neck:
      downsample_type: "conv"    # strided convolution (default, ActionFormer/TemporalMaxer)
      # or
      downsample_type: "maxpool"  # max-pooling with stride 2 (TriDet)
    ```

### Phase 8: Create Config Files

17. **Create example YAML configs** for each architecture combination. At minimum:

    ```
    configs/
    ├── actionformer.yaml          # existing baseline
    ├── temporalmaxer.yaml         # MaxPool backbone + standard head
    ├── tridet.yaml                # SGP backbone + Trident head
    ├── sgp_standard_head.yaml     # SGP backbone + standard head (ablation)
    └── maxpool_trident_head.yaml  # MaxPool backbone + Trident head (ablation)
    ```

    Each config should be a complete, runnable configuration file. Copy the existing config as a base and only change the model architecture sections.

### Phase 9: Update Model Builder

18. **Update the model construction code** (likely `meta_archs.py` or equivalent) to:
    - Read `backbone.type`, `neck.downsample_type`, and `head.type` from config
    - Use the factory functions to instantiate the correct components
    - Ensure all components are compatible (same feature dimensions flow through)
    - Handle any architecture-specific initialization (e.g., Trident-head weight init)

19. **Ensure the training loop is compatible.** The main changes needed:
    - If using Trident-head, the loss function must compute the distribution loss
    - The output format from the head may differ — ensure post-processing (NMS, etc.) handles both output formats
    - Evaluation code should work with both head types

### Phase 10: Testing & Validation

20. **Verify each configuration runs without errors:**
    - Instantiate each model config and do a forward pass with dummy data
    - Check that gradients flow correctly through all components
    - Verify that the model can be saved and loaded correctly
    - Run a short training loop (1-2 epochs) with each config to verify end-to-end

21. **Write a simple test script** that:
    ```python
    # test_models.py
    for config in ["actionformer.yaml", "temporalmaxer.yaml", "tridet.yaml"]:
        model = build_model(load_config(config))
        dummy_input = torch.randn(2, C, T)  # batch of 2
        output = model(dummy_input)
        loss = compute_loss(output, dummy_targets)
        loss.backward()
        print(f"{config}: forward OK, backward OK, output shape: {output.shape}")
    ```

---

## Implementation Guidelines

### Code Style
- **Match the existing codebase style exactly.** Don't introduce new patterns or conventions.
- If the codebase uses `snake_case`, use `snake_case`. If it registers modules in a dict, use the same dict pattern.
- Keep imports consistent with the existing code.

### Config Backward Compatibility
- **The existing config files must continue to work unchanged.** Use sensible defaults so that if `backbone.type` is not specified, it defaults to the current backbone.
- Add the new config keys with defaults that reproduce the existing behaviour.

### Minimal Invasive Changes
- **Don't refactor the entire codebase.** Only modify what's necessary to support the new architectures.
- If the existing code has backbone/neck/head in a single file, it's fine to keep them there and add new classes to the same file — unless it becomes unmanageable.
- Prefer adding new files over modifying existing ones where possible.

### From Reference Repos
- When porting code from TemporalMaxer or TriDet, **adapt it to the existing codebase's conventions**, don't copy-paste blindly.
- The reference repos are all based on ActionFormer's codebase, so the patterns should be similar, but there may be differences in variable naming, tensor shapes (B,C,T vs B,T,C), mask handling, etc.
- Preserve the mathematical correctness of the implementations — especially the Trident-head's distribution computation and loss.

### Documentation
- Add docstrings to all new classes explaining which paper they come from.
- Add comments in the config files explaining each new parameter.
- Update the repo README to document the new configurable architectures.

---

## Reference Repos

- **TemporalMaxer**: https://github.com/TuanTNG/TemporalMaxer (arXiv:2303.09055)
- **TriDet**: https://github.com/dingfengshi/TriDet (CVPR 2023, arXiv:2303.07347)
- Both are built on the ActionFormer codebase: https://github.com/happyharrycn/actionformer_release

---

## Summary of Key Files to Create/Modify

| Action | File | Description |
|--------|------|-------------|
| Study | Existing model files | Understand current architecture |
| Study | `/tmp/ref/TemporalMaxer/libs/modeling/` | MaxPool backbone implementation |
| Study | `/tmp/ref/TriDet/libs/modeling/` | SGP layer + Trident-head implementation |
| Create | Backbone factory/registry | Dispatch backbone by config type |
| Create | `temporal_maxer` backbone | MaxPool1D-based temporal context modeling |
| Create | `sgp` backbone | Scalable-Granularity Perception layer |
| Create | `trident_head` | Relative boundary distribution head |
| Modify | Neck/FPN | Add configurable downsampling (conv vs maxpool) |
| Modify | Model builder | Use factories for backbone/neck/head |
| Modify | Loss computation | Support Trident distribution loss |
| Modify | Post-processing | Handle both standard and Trident output formats |
| Create | Config YAMLs | One per architecture variant |
| Create | Test script | Verify all configs run correctly |
| Modify | README | Document new architectures and configs |