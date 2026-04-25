from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.qata_npz import QaTaNPZDataset
from src.models.model_factory import ModelBuildConfig, build_model
from src.models.transunet2d_v11 import build_transunet2d_v11


def unwrap_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["seg"]
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum(axis=(1, 2, 3))
    fp = np.logical_and(pred, ~gt).sum(axis=(1, 2, 3))
    fn = np.logical_and(~pred, gt).sum(axis=(1, 2, 3))
    pred_sum = pred.sum(axis=(1, 2, 3))
    gt_sum = gt.sum(axis=(1, 2, 3))
    union = np.logical_or(pred, gt).sum(axis=(1, 2, 3))
    eps = 1e-6
    dice = ((2 * tp + eps) / (pred_sum + gt_sum + eps)).mean()
    iou = ((tp + eps) / (union + eps)).mean()
    precision = ((tp + eps) / (tp + fp + eps)).mean()
    recall = ((tp + eps) / (tp + fn + eps)).mean()
    return float(dice), float(iou), float(precision), float(recall)


def build_named_model(name: str):
    if name == "transunet2d_v11":
        return build_transunet2d_v11(in_channels=1, out_channels=1)
    return build_model(ModelBuildConfig(name=name, in_channels=1, out_channels=1))


def collect_probs(args: argparse.Namespace) -> tuple[list[np.ndarray], np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    names = args.models.split(",")
    ckpts = args.ckpts.split(",")
    if len(names) != len(ckpts):
        raise ValueError("--models and --ckpts must have the same length")

    models = []
    for name, ckpt_path in zip(names, ckpts):
        model = build_named_model(name).to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append(model)
        print(f"loaded {name}: {ckpt_path}")

    ds = QaTaNPZDataset(str(Path(args.data_dir) / "manifest.csv"), split=args.split)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_probs = [[] for _ in models]
    gts = []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(device, non_blocking=True)
            for i, model in enumerate(models):
                all_probs[i].append(torch.sigmoid(unwrap_logits(model(x))).cpu().numpy())
            gts.append(batch["mask"].numpy())

    return [np.concatenate(p, axis=0) for p in all_probs], np.concatenate(gts, axis=0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/qata")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--models", default="transunet2d_v11,uctransnet2d")
    parser.add_argument(
        "--ckpts",
        default="outputs/transunet2d_v11_qata_exp_best.pt,outputs/uctransnet2d_qata_best.pt",
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--thr_min", type=float, default=0.30)
    parser.add_argument("--thr_max", type=float, default=0.60)
    parser.add_argument("--thr_step", type=float, default=0.01)
    args = parser.parse_args()

    probs, gt = collect_probs(args)
    if len(probs) != 2:
        raise ValueError("This script currently sweeps two-model ensembles only")

    best = None
    for w in [float(x) for x in args.weights.split(",") if x.strip()]:
        fused = w * probs[0] + (1.0 - w) * probs[1]
        for th in np.arange(args.thr_min, args.thr_max + args.thr_step / 2, args.thr_step):
            pred = (fused > float(th)).astype(np.uint8)
            result = metrics(pred, gt)
            if best is None or result[0] > best[2][0]:
                best = (w, float(th), result)

    assert best is not None
    w, th, result = best
    print(
        f"best weight_model0={w:.3f} thr={th:.3f} "
        f"dice={result[0]:.6f} iou={result[1]:.6f} "
        f"precision={result[2]:.6f} recall={result[3]:.6f}"
    )


if __name__ == "__main__":
    main()
