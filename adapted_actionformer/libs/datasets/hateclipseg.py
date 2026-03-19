"""
HateClipSeg Dataset for temporal hateful content localization.

Shares all logic with HateMMDataset — only the class name differs.
Annotations must first be prepared with:
    python data/hateclipseg/scripts/prepare_annotations.py

See libs/datasets/hatemm.py for the full implementation.
"""
from .hatemm import HateMMDataset, collate_fn, _build_dataloader  # noqa: F401


class HateClipSegDataset(HateMMDataset):
    _name = "HateClipSegDataset"


def build_dataloader(cfg, subset, is_training=False):
    return _build_dataloader(HateClipSegDataset, cfg, subset, is_training)
