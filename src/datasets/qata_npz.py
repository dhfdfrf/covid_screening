import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset


class QaTaNPZDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        split: str,
        augment: bool = False,
        hflip_prob: float = 0.5,
        intensity_prob: float = 0.8,
        noise_prob: float = 0.25,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.paths = [Path(p) for p in self.df["npz_path"].tolist()]
        self.augment = bool(augment)
        self.hflip_prob = float(hflip_prob)
        self.intensity_prob = float(intensity_prob)
        self.noise_prob = float(noise_prob)

    def __len__(self):
        return len(self.paths)

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Chest X-rays keep a strong vertical prior; only horizontal flip is safe here.
        if np.random.random() < self.hflip_prob:
            img = img[..., ::-1].copy()
            mask = mask[..., ::-1].copy()

        if np.random.random() < self.intensity_prob:
            contrast = np.random.uniform(0.85, 1.15)
            brightness = np.random.uniform(-0.06, 0.06)
            gamma = np.random.uniform(0.85, 1.20)
            mean = float(img.mean())
            img = (img - mean) * contrast + mean + brightness
            img = np.clip(img, 0.0, 1.0)
            img = np.power(img, gamma).astype(np.float32)

        if np.random.random() < self.noise_prob:
            noise_std = np.random.uniform(0.005, 0.02)
            img = np.clip(img + np.random.normal(0.0, noise_std, img.shape), 0.0, 1.0).astype(np.float32)

        return img.astype(np.float32), mask.astype(np.float32)

    def __getitem__(self, idx: int):
        npz = np.load(self.paths[idx])
        img = npz["img"].astype(np.float32)   # (1,H,W)
        mask = npz["mask"].astype(np.float32) # (1,H,W)

        if self.augment:
            img, mask = self._augment(img, mask)

        # torch tensor
        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask),
        }
