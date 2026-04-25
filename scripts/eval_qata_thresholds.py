from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.qata_npz import QaTaNPZDataset
from src.models.transunet2d_v11 import build_transunet2d_v11


def dice_iou_precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum(axis=(1, 2, 3))
    fp = np.logical_and(pred, ~gt).sum(axis=(1, 2, 3))
    fn = np.logical_and(~pred, gt).sum(axis=(1, 2, 3))
    inter = tp
    pred_sum = pred.sum(axis=(1, 2, 3))
    gt_sum = gt.sum(axis=(1, 2, 3))
    union = np.logical_or(pred, gt).sum(axis=(1, 2, 3))
    eps = 1e-6
    dice = ((2 * inter + eps) / (pred_sum + gt_sum + eps)).mean()
    iou = ((inter + eps) / (union + eps)).mean()
    precision = ((tp + eps) / (tp + fp + eps)).mean()
    recall = ((tp + eps) / (tp + fn + eps)).mean()
    return float(dice), float(iou), float(precision), float(recall)


def remove_small_components(pred: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return pred

    out = np.zeros_like(pred, dtype=np.uint8)
    for i in range(pred.shape[0]):
        mask = pred[i, 0].astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask, dtype=np.uint8)
        for label in range(1, n_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == label] = 1
        out[i, 0] = keep
    return out


def load_probabilities(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = build_transunet2d_v11(in_channels=1, out_channels=1).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = QaTaNPZDataset(str(Path(args.data_dir) / "manifest.csv"), split=args.split)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    probs: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(device, non_blocking=True)
            logits = predict_logits(model, x, args.tta_mode)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            gts.append(batch["mask"].numpy())

    return np.concatenate(probs, axis=0), np.concatenate(gts, axis=0).astype(np.uint8)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: torch.Tensor, tta_mode: str) -> torch.Tensor:
    seg = model(x)["seg"]
    if tta_mode == "none":
        return seg

    logits = [seg]
    if tta_mode in ("h", "all"):
        logits.append(torch.flip(model(torch.flip(x, [3]))["seg"], [3]))
    if tta_mode in ("v", "all"):
        logits.append(torch.flip(model(torch.flip(x, [2]))["seg"], [2]))
    if tta_mode in ("hv", "all"):
        logits.append(torch.flip(model(torch.flip(x, [2, 3]))["seg"], [2, 3]))
    return torch.stack(logits, dim=0).mean(dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/qata")
    parser.add_argument("--ckpt", default="outputs/transunet2d_v11_qata_exp_best.pt")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tta", action="store_true", help="legacy alias for --tta_mode all")
    parser.add_argument("--tta_mode", default="none", choices=("none", "h", "v", "hv", "all"))
    parser.add_argument("--thr_min", type=float, default=0.20)
    parser.add_argument("--thr_max", type=float, default=0.80)
    parser.add_argument("--thr_step", type=float, default=0.01)
    parser.add_argument("--min_areas", default="0,8,16,32,64,128,256")
    args = parser.parse_args()
    if args.tta:
        args.tta_mode = "all"

    prob, gt = load_probabilities(args)
    print(f"samples={prob.shape[0]} shape={prob.shape[2]}x{prob.shape[3]}")

    min_areas = [int(x) for x in args.min_areas.split(",") if x.strip()]
    thresholds = np.arange(args.thr_min, args.thr_max + args.thr_step / 2, args.thr_step)
    best: tuple[float, int, tuple[float, float, float, float]] | None = None

    for threshold in thresholds:
        raw_pred = (prob > float(threshold)).astype(np.uint8)
        for min_area in min_areas:
            pred = remove_small_components(raw_pred, min_area)
            metrics = dice_iou_precision_recall(pred, gt)
            if best is None or metrics[0] > best[2][0]:
                best = (float(threshold), min_area, metrics)

    assert best is not None
    th, min_area, metrics = best
    print(
        "best "
        f"thr={th:.3f} min_area={min_area} "
        f"dice={metrics[0]:.6f} iou={metrics[1]:.6f} "
        f"precision={metrics[2]:.6f} recall={metrics[3]:.6f}"
    )

    for threshold in (0.40, 0.45, 0.50, 0.55, th):
        raw_pred = (prob > float(threshold)).astype(np.uint8)
        pred = remove_small_components(raw_pred, min_area if threshold == th else 0)
        metrics = dice_iou_precision_recall(pred, gt)
        print(
            f"setting thr={threshold:.3f} min_area={min_area if threshold == th else 0} "
            f"dice={metrics[0]:.6f} iou={metrics[1]:.6f} "
            f"precision={metrics[2]:.6f} recall={metrics[3]:.6f}"
        )


if __name__ == "__main__":
    main()
