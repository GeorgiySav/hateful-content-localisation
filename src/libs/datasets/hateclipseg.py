"""
HateClipSeg Dataset for temporal hateful content localization.

Loads pre-extracted per-video .pt feature files and annotations.

Annotation format (annotations.json):
    {
        "database": {
            "<video_id>": {
                "duration": <float>,
                "subset": "train" | "val",
                "annotations": [
                    {"segment": [<start>, <end>], "label": "hate"},
                    ...
                ],
                "video_label": "hate" | "non_hate"
            }
        }
    }

Annotations must first be prepared with:
    python data/hateclipseg/scripts/prepare_annotations.py
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader


class HateClipSegDataset(Dataset):
    """
    PyTorch Dataset for HateClipSeg temporal localization.

    Args:
        video_feat_dir : Path to directory containing per-video video .pt files.
        audio_feat_dir : Path to directory containing per-video audio .pt files.
        text_feat_dir  : Path to directory containing per-video text .pt files.
        annotation_file: Path to annotations.json.
        max_seq_len    : Sequences are padded (or randomly cropped) to this length.
        subset         : "train", "val".
        feature_fps    : Features per second.
    """

    _name = "HateClipSegDataset"

    def __init__(
        self,
        video_feat_dir,
        audio_feat_dir,
        text_feat_dir,
        annotation_file,
        max_seq_len,
        subset="train",
        feature_fps=1.0,
        is_training=True,
        aug_cfg=None,
        use_json_split=True,
        val_ratio=0.2,
        split_seed=42,
    ):
        super().__init__()
        self.video_feat_dir = video_feat_dir
        self.audio_feat_dir = audio_feat_dir
        self.text_feat_dir  = text_feat_dir
        self.max_seq_len    = max_seq_len
        self.feature_fps    = feature_fps
        self.is_training    = is_training
        self.subset         = subset
        self.aug_cfg        = aug_cfg if aug_cfg is not None else {}

        with open(annotation_file, 'r') as f:
            db = json.load(f)['database']

        if not use_json_split:
            split_map = self._make_stratified_split(db, val_ratio, split_seed)
            n_train = sum(1 for v in split_map.values() if v == 'train')
            n_val   = sum(1 for v in split_map.values() if v == 'val')
            print(f"[{self._name}] Stratified internal split: "
                  f"{n_train} train / {n_val} val "
                  f"(val_ratio={val_ratio}, seed={split_seed})")
        else:
            split_map = None

        self.video_ids    = []
        self.annotations  = {}
        self.video_labels = []

        skipped = 0
        for vid_id, meta in db.items():
            vid_subset = (split_map[vid_id] if split_map is not None
                          else meta.get('subset', 'train'))
            if vid_subset != subset:
                continue

            vp = os.path.join(video_feat_dir, f"{vid_id}.pt")
            ap = os.path.join(audio_feat_dir, f"{vid_id}.pt")
            tp = os.path.join(text_feat_dir,  f"{vid_id}.pt")
            if not (os.path.exists(vp) and os.path.exists(ap) and os.path.exists(tp)):
                skipped += 1
                continue

            duration    = float(meta['duration'])
            raw_anns    = meta.get('annotations', [])
            video_label = meta.get('video_label', None)

            if len(raw_anns) > 0:
                segs, labels = [], []
                for ann in raw_anns:
                    seg   = ann['segment']
                    label = ann.get('label', 'hate')
                    if label.lower() in ('hate', 'hateful'):
                        s, e = float(seg[0]), float(seg[1])
                        if s == e:
                            continue  # skip zero-length annotations
                        if s > e:
                            s, e = e, s  # fix inverted timestamps
                        segs.append([s, e])
                        labels.append(0)  # hate = class 0
                self.annotations[vid_id] = {
                    'duration': duration,
                    'segments': segs,
                    'labels'  : labels,
                }
            else:
                is_hate = (video_label is not None and
                           video_label.lower() in ('hate', 'hateful'))
                if is_hate:
                    skipped += 1
                    continue
                else:
                    self.annotations[vid_id] = {
                        'duration': duration,
                        'segments': [],
                        'labels'  : [],
                    }

            self.video_ids.append(vid_id)
            self.video_labels.append(1 if self.annotations[vid_id]['segments'] else 0)

        self.split_video_ids = set(self.video_ids)

        print(f"[{self._name}] Loaded {len(self.video_ids)} videos for subset='{subset}'"
              + (f" ({skipped} skipped)" if skipped else ""))

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        ann      = self.annotations[video_id]

        video_feat = torch.load(
            os.path.join(self.video_feat_dir, f"{video_id}.pt"), map_location='cpu'
        ).float()   # (T, 768)
        audio_feat = torch.load(
            os.path.join(self.audio_feat_dir, f"{video_id}.pt"), map_location='cpu'
        ).float()   # (T, 1024)
        text_feat  = torch.load(
            os.path.join(self.text_feat_dir,  f"{video_id}.pt"), map_location='cpu'
        ).float()   # (T, 768)

        T        = video_feat.shape[0]
        duration = ann['duration']

        audio_feat = self._match_length(audio_feat, T, 1024)
        text_feat  = self._match_length(text_feat,  T, 768)

        segs = ann['segments']
        lbls = ann['labels']
        if len(segs) > 0:
            segments = torch.tensor(segs, dtype=torch.float32)
            labels   = torch.tensor(lbls, dtype=torch.long)
        else:
            segments = torch.zeros((0, 2), dtype=torch.float32)
            labels   = torch.zeros((0,),   dtype=torch.long)

        if T < self.max_seq_len:
            video_feat, audio_feat, text_feat, mask = self._pad(
                video_feat, audio_feat, text_feat, T)
        elif T > self.max_seq_len:
            if self.is_training:
                video_feat, audio_feat, text_feat, mask, segments, labels = self._random_crop(
                    video_feat, audio_feat, text_feat, segments, labels)
            else:
                video_feat = video_feat[:self.max_seq_len]
                audio_feat = audio_feat[:self.max_seq_len]
                text_feat  = text_feat[:self.max_seq_len]
                mask       = torch.ones(self.max_seq_len, dtype=torch.float32)
        else:
            mask = torch.ones(T, dtype=torch.float32)

        if self.is_training and self.aug_cfg.get('enabled', False):
            video_feat, audio_feat, text_feat, mask, segments, labels = self._augment(
                video_feat, audio_feat, text_feat, segments, labels, mask)

        return {
            'video_id'  : video_id,
            'video_feat': video_feat,
            'audio_feat': audio_feat,
            'text_feat' : text_feat,
            'mask'      : mask,
            'segments'  : segments,
            'labels'    : labels,
            'duration'  : duration,
        }

    # Helpers
    def _augment(self, video_feat, audio_feat, text_feat, segments, labels, mask):
        cfg = self.aug_cfg

        shift_max = cfg.get('temporal_shift_max_frames', 0)
        if shift_max > 0:
            shift = torch.randint(-shift_max, shift_max + 1, (1,)).item()
            if shift != 0:
                video_feat, audio_feat, text_feat, mask, segments, labels = self._temporal_shift(
                    video_feat, audio_feat, text_feat, mask, segments, labels, shift)

        noise_std = cfg.get('feature_noise_std', 0.0)
        if noise_std > 0.0:
            # mask prevents noise from corrupting zero-padded frames
            noise_mask = mask.unsqueeze(1)
            video_feat = video_feat + torch.randn_like(video_feat) * noise_std * noise_mask
            audio_feat = audio_feat + torch.randn_like(audio_feat) * noise_std * noise_mask
            text_feat  = text_feat  + torch.randn_like(text_feat)  * noise_std * noise_mask

        return video_feat, audio_feat, text_feat, mask, segments, labels

    def _temporal_shift(self, video_feat, audio_feat, text_feat, mask, segments, labels, shift):
        """
        Shift content by `shift` frames along the temporal axis, preserving total length.
        Positive `shift` pads zero frames at the start and crops from the end.
        Negative `shift` crops from the start and pads zero frames at the end.
        Segments are shifted by `shift / feature_fps`.
        Any segment that ends up outside [0, max_seq_len/feature_fps] is dropped.
        """
        if shift > 0:
            video_feat = torch.cat([torch.zeros(shift, video_feat.shape[1]), video_feat[:-shift]], dim=0)
            audio_feat = torch.cat([torch.zeros(shift, audio_feat.shape[1]), audio_feat[:-shift]], dim=0)
            text_feat  = torch.cat([torch.zeros(shift, text_feat.shape[1]),  text_feat[:-shift]],  dim=0)
            mask       = torch.cat([torch.zeros(shift),                      mask[:-shift]],       dim=0)
        else:
            s = -shift
            video_feat = torch.cat([video_feat[s:], torch.zeros(s, video_feat.shape[1])], dim=0)
            audio_feat = torch.cat([audio_feat[s:], torch.zeros(s, audio_feat.shape[1])], dim=0)
            text_feat  = torch.cat([text_feat[s:],  torch.zeros(s, text_feat.shape[1])],  dim=0)
            mask       = torch.cat([mask[s:],       torch.zeros(s)],                      dim=0)

        if segments.shape[0] > 0:
            crop_dur = self.max_seq_len / self.feature_fps
            segments = segments + shift / self.feature_fps
            valid    = (segments[:, 1] > 0) & (segments[:, 0] < crop_dur)
            segments = segments[valid].clamp(min=0.0, max=crop_dur)
            labels   = labels[valid]

        return video_feat, audio_feat, text_feat, mask, segments, labels

    @staticmethod
    def _match_length(feat, target_len, expected_dim):
        T_feat = feat.shape[0]
        if T_feat == target_len:
            return feat
        elif T_feat < target_len:
            pad = torch.zeros(target_len - T_feat, expected_dim, dtype=feat.dtype)
            return torch.cat([feat, pad], dim=0)
        else:
            return feat[:target_len]

    def _pad(self, video_feat, audio_feat, text_feat, T):
        pad_len = self.max_seq_len - T
        video_feat = torch.cat([video_feat, torch.zeros(pad_len, 768)],  dim=0)
        audio_feat = torch.cat([audio_feat, torch.zeros(pad_len, 1024)], dim=0)
        text_feat  = torch.cat([text_feat,  torch.zeros(pad_len, 768)],  dim=0)
        mask = torch.cat([torch.ones(T), torch.zeros(pad_len)], dim=0)
        return video_feat, audio_feat, text_feat, mask

    @staticmethod
    def _make_stratified_split(db, val_ratio=0.2, seed=42):
        import random
        hate_ids, non_hate_ids = [], []
        for vid_id, meta in db.items():
            has_hate = any(
                a.get('label', '').lower() in ('hate', 'hateful')
                for a in meta.get('annotations', [])
            )
            (hate_ids if has_hate else non_hate_ids).append(vid_id)

        hate_ids.sort()
        non_hate_ids.sort()

        rng = random.Random(seed)
        rng.shuffle(hate_ids)
        rng.shuffle(non_hate_ids)

        n_hate_val     = max(1, round(len(hate_ids)     * val_ratio))
        n_non_hate_val = max(1, round(len(non_hate_ids) * val_ratio))

        mapping = {}
        for i, vid in enumerate(hate_ids):
            mapping[vid] = 'val' if i < n_hate_val else 'train'
        for i, vid in enumerate(non_hate_ids):
            mapping[vid] = 'val' if i < n_non_hate_val else 'train'
        return mapping

    def _random_crop(self, video_feat, audio_feat, text_feat, segments, labels):
        T     = video_feat.shape[0]
        start = torch.randint(0, T - self.max_seq_len + 1, (1,)).item()
        end   = start + self.max_seq_len

        video_feat = video_feat[start:end]
        audio_feat = audio_feat[start:end]
        text_feat  = text_feat[start:end]
        mask       = torch.ones(self.max_seq_len, dtype=torch.float32)

        if segments.shape[0] > 0:
            s_shifted = segments - start / self.feature_fps
            crop_dur  = self.max_seq_len / self.feature_fps
            valid     = (s_shifted[:, 1] > 0) & (s_shifted[:, 0] < crop_dur)
            s_shifted = s_shifted[valid].clamp(min=0.0, max=crop_dur)
            labels    = labels[valid]
        else:
            s_shifted = segments

        return video_feat, audio_feat, text_feat, mask, s_shifted, labels


def collate_fn(batch):
    return {
        'video_id'  : [b['video_id']   for b in batch],
        'video_feat': torch.stack([b['video_feat'] for b in batch]),
        'audio_feat': torch.stack([b['audio_feat'] for b in batch]),
        'text_feat' : torch.stack([b['text_feat']  for b in batch]),
        'mask'      : torch.stack([b['mask']       for b in batch]),
        'segments'  : [b['segments']  for b in batch],
        'labels'    : [b['labels']    for b in batch],
        'duration'  : [b['duration']  for b in batch],
    }


def build_dataloader(cfg, subset, is_training=False):
    ds_cfg  = cfg['dataset']
    dataset = HateClipSegDataset(
        video_feat_dir =ds_cfg['video_feat_dir'],
        audio_feat_dir =ds_cfg['audio_feat_dir'],
        text_feat_dir  =ds_cfg['text_feat_dir'],
        annotation_file=ds_cfg['annotation_file'],
        max_seq_len    =ds_cfg['max_seq_len'],
        subset         =subset,
        feature_fps    =ds_cfg.get('feature_fps', 1.0),
        is_training    =is_training,
        aug_cfg        =cfg.get('augmentation', {}) if is_training else {},
        use_json_split =ds_cfg.get('use_json_split', True),
        val_ratio      =ds_cfg.get('val_ratio', 0.2),
        split_seed     =ds_cfg.get('split_seed', 42),
    )
    train_cfg  = cfg.get('training', {})
    batch_size = train_cfg.get('batch_size', 2) if is_training else 1

    shuffle = is_training
    if is_training:
        labels     = dataset.video_labels
        n_hate     = sum(labels)
        n_non_hate = len(labels) - n_hate
        print(f"[{dataset._name}] Uniform shuffle: {n_hate} hate / {n_non_hate} non-hate")

    return DataLoader(
        dataset,
        batch_size =batch_size,
        shuffle    =shuffle,
        num_workers=4,
        pin_memory =True,
        drop_last  =is_training,
        collate_fn =collate_fn,
    )
