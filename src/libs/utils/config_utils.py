import os
import yaml


def load_config(path):
    config_dir = os.path.dirname(os.path.abspath(path))
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    ds = cfg['dataset']
    for key in ('video_feat_dir', 'audio_feat_dir', 'text_feat_dir', 'annotation_file'):
        if key in ds and not os.path.isabs(ds[key]):
            ds[key] = os.path.normpath(os.path.join(config_dir, ds[key]))
    return cfg
