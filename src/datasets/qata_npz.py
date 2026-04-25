from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class QaTaNPZDataset(Dataset):
    def __init__(self, manifest_csv: str, split: str):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.paths = [Path(p) for p in self.df["npz_path"].tolist()]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        npz = np.load(self.paths[idx])
        img = npz["img"].astype(np.float32)   # (1,H,W)
        mask = npz["mask"].astype(np.float32) # (1,H,W)

        # torch tensor
        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask),
        }
