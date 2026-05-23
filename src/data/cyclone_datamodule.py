"""Lightning datamodule for raw cyclone IR frames.

The canonical (and only) datamodule used by the pixel-supervision
ablation. Reads the HuggingFace Arrow dataset built by
``dataset/06_build_hf.py``, flattens cyclone *trajectories* into
individual *frames* via :class:`CycloneFrameDataset`, applies optional
discrete 90° / flip augmentation, and yields per-frame batches of
``{frames, pressure, wind, center, cyclone_id}``.

Normalization statistics are loaded from
``$DATASET_PATH/normalization_stats.json`` (produced by
``dataset/07_normalize_stats.py``). A small set of fallback constants
is used if the file is missing — useful for smoke tests but not
recommended for headline runs.
"""

import numpy as np
import torch
import json
import bisect
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule
from datasets import load_from_disk
from typing import Optional, Dict, List
from pathlib import Path

class CycloneFrameDataset(Dataset):
    """
    Flattens a dataset of Cyclone Tracks into a dataset of Individual Frames.
    Uses indexing math to avoid duplicating data in memory.
    """
    def __init__(
        self,
        hf_dataset,
        stats: Dict[str, float],
        normalize: bool = True,
        use_augmentation: bool = False,
    ):
        self.dataset = hf_dataset
        self.normalize = normalize
        
        # Stats
        self.f_min = stats['frame_min']
        self.f_max = stats['frame_max']
        self.p_mean = stats['pressure_mean']
        self.p_std = stats['pressure_std']
        self.w_mean = stats['wind_mean']
        self.w_std = stats['wind_std']

        # --- INDEX MAPPING (The Magic Part) ---
        # 1. Pre-calculate the number of frames in every track
        #    Note: accessing len() of a list column in HF is generally fast/metadata-based
        print("Indexing frames... (this happens once)")
        self.track_lengths = [len(x) for x in hf_dataset['frames']]
        
        # 2. Build cumulative sum (breakpoints)
        #    If track_lengths is [10, 20], cumulative is [10, 30]
        self.cumulative_sizes = np.cumsum(self.track_lengths)
        self.total_frames = self.cumulative_sizes[-1]
        print(f"Total individual frames indexed: {self.total_frames}")

        # Augmentation
        self.transform = None
        if use_augmentation:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        # The DataLoader sees the total sum of all frames
        return self.total_frames

    def __getitem__(self, idx):
        # 1. MAP GLOBAL IDX TO TRACK IDX
        # Use bisect (binary search) to find which track this frame belongs to
        # O(log N) complexity - extremely fast
        track_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        
        # 2. CALCULATE LOCAL FRAME IDX
        if track_idx == 0:
            local_frame_idx = idx
        else:
            local_frame_idx = idx - self.cumulative_sizes[track_idx - 1]

        # 3. LAZY LOAD SPECIFIC TRACK
        # We assume the OS file cache helps here if accessing sequential frames
        sample = self.dataset[track_idx]
        
        # 4. EXTRACT SPECIFIC FRAME
        frame = np.array(sample['frames'][local_frame_idx], dtype=np.float32)
        pressure = float(sample['pressure'][local_frame_idx])
        wind = float(sample['wind'][local_frame_idx])
        center = np.array(sample['center'][local_frame_idx], dtype=np.float32)

        # 5. AUGMENTATION
        if self.transform:
            augmented = self.transform(image=frame, keypoints=[center])
            frame = augmented['image']
            if augmented['keypoints']:
                center = np.array(augmented['keypoints'][0], dtype=np.float32)

        # 6. NORMALIZATION
        if self.normalize:
            frame = np.clip(frame, self.f_min, self.f_max)
            frame = (frame - self.f_min) / (self.f_max - self.f_min + 1e-8)
            pressure = (pressure - self.p_mean) / (self.p_std + 1e-8)
            wind = (wind - self.w_mean) / (self.w_std + 1e-8)

        # 7. TO TENSOR (Add Channel Dim: 1, H, W)
        return {
            'frames': torch.from_numpy(frame).float().unsqueeze(0),
            'pressure': torch.tensor(pressure, dtype=torch.float),
            'wind': torch.tensor(wind, dtype=torch.float),
            'center': torch.from_numpy(center).float(),
            'cyclone_id': sample['cyclone_id'],
            'original_track_idx': track_idx 
        }

class CycloneDataModule(LightningDataModule):
    PRESSURE_CATEGORIES = {
        "cat5": (0, 980), "cat4": (920, 945), "cat3": (945, 965),
        "cat2": (965, 980), "cat1": (980, 1000), "ts": (1000, 1020),
        "all": (0, 980),
    }

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pressure_category: str = "all",
        augment: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_path = Path(dataset_path)
        self.p_min, self.p_max = self.PRESSURE_CATEGORIES.get(pressure_category, (0, 2000))

    def setup(self, stage: str = None):
        full_dict = load_from_disk(str(self.dataset_path))
        self._load_stats()

        if stage in (None, 'fit'):
            # Note: We pass 'augment=True' only to train
            self.train_ds = self._prepare_split(full_dict['train'], "train", augment=self.hparams.augment)
            self.val_ds = self._prepare_split(full_dict['validation'], "val", augment=False)

        if stage in (None, 'test'):
            self.test_ds = self._prepare_split(full_dict['test'], "test", augment=False)

    def _prepare_split(self, dataset, split_name, augment=False):
        if not dataset: return None
        
        # RAM-Efficient Filtering
        # Filters TRACKS first. If a track has valid pressure, we keep the track.
        # (Later, the Dataset class will break that track into frames)
        if self.p_min > 0 or self.p_max < 2000:
            print(f"Filtering {split_name} tracks by pressure...")
            dataset = dataset.filter(
                lambda batch: [
                    np.any((np.array(p) >= self.p_min) & (np.array(p) < self.p_max)) 
                    for p in batch['pressure']
                ],
                batched=True,
                batch_size=1000, 
                num_proc=4 
            )

        return CycloneFrameDataset(
            hf_dataset=dataset,
            stats=self.stats,
            use_augmentation=augment
        )

    def _load_stats(self):
        stats_path = self.dataset_path / "normalization_stats.json"
        if stats_path.exists():
            with open(stats_path) as f: self.stats = json.load(f)
        else:
            self.stats = {
                "frame_min": 0.0, "frame_max": 255.0,
                "pressure_mean": 980.0, "pressure_std": 20.0,
                "wind_mean": 50.0, "wind_std": 30.0
            }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, # This now shuffles GLOBAL frames, mixing tracks completely
            num_workers=self.hparams.num_workers,
            pin_memory=True, 
            persistent_workers=True,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        # Standard collation
        keys = batch[0].keys()
        collated = {}
        for k in keys:
            items = [d[k] for d in batch]
            if isinstance(items[0], torch.Tensor):
                collated[k] = torch.stack(items)
            else:
                collated[k] = items
        return collated