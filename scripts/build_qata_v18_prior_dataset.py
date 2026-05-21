from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.qata_npz import QaTaNPZDataset
from src.models.model_factory import ModelBuildConfig, build_model


def unwrap_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["seg"]
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def load_state(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)


@torch.no_grad()
def write_split(
    source_manifest: Path,
    split: str,
    out_npz_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    dataset = QaTaNPZDataset(str(source_manifest), split=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    rows: list[pd.Series] = []
    model.eval()
    out_npz_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(loader):
        x = batch["image"].to(device, non_blocking=True)
        prob = torch.sigmoid(unwrap_logits(model(x))).cpu().numpy().astype(np.float32)
        imgs = batch["image"].numpy().astype(np.float32)
        masks = batch["mask"].numpy().astype(np.float32)

        start = batch_idx * batch_size
        for i in range(imgs.shape[0]):
            source_row = dataset.df.iloc[start + i].copy()
            case_id = str(source_row["id"])
            stacked = np.concatenate([imgs[i], prob[i]], axis=0).astype(np.float32)
            out_npz = (out_npz_dir / f"{case_id}.npz").resolve()
            np.savez_compressed(out_npz, img=stacked, mask=masks[i])
            source_row["npz_path"] = str(out_npz)
            rows.append(source_row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data_dir", default="data/processed/qata_lowdice_specialist")
    parser.add_argument("--out_data_dir", default="data/processed/qata_lowdice_specialist_v18prior")
    parser.add_argument("--v18_model", default="transunet2d_v18")
    parser.add_argument("--v18_ckpt", default="outputs/transunet2d_v18_freq_ft8_best.pt")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    source_manifest = Path(args.source_data_dir) / "manifest.csv"
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)

    out_dir = Path(args.out_data_dir)
    out_npz_dir = out_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ModelBuildConfig(name=args.v18_model, in_channels=1, out_channels=1)).to(device)
    load_state(model, Path(args.v18_ckpt))

    frames = []
    for split in ("train", "val", "test"):
        part = write_split(
            source_manifest=source_manifest,
            split=split,
            out_npz_dir=out_npz_dir,
            model=model,
            device=device,
            batch_size=args.batch,
            num_workers=args.num_workers,
        )
        frames.append(part)
        print(f"{split}: {len(part)}")

    manifest = pd.concat(frames, ignore_index=True)
    out_manifest = out_dir / "manifest.csv"
    manifest.to_csv(out_manifest, index=False, encoding="utf-8")
    print(f"manifest={out_manifest}")


if __name__ == "__main__":
    main()
