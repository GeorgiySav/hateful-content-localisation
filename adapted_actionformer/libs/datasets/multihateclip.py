"""
MultiHateClip Dataset for temporal hateful content localization.

Shares all logic with HateMMDataset — only the class name differs.
Annotations must first be prepared with:
    python data/multihateclip/scripts/create_annotations.py ...

See libs/datasets/hatemm.py for the full implementation.
"""
from .hatemm import HateMMDataset, collate_fn, _build_dataloader  # noqa: F401


class MultiHateClipDataset(HateMMDataset):
    _name = "MultiHateClipDataset"


def build_dataloader(cfg, subset, is_training=False):
    return _build_dataloader(MultiHateClipDataset, cfg, subset, is_training)
