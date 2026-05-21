from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.qata_npz import QaTaNPZDataset
from src.models.model_factory import ModelBuildConfig, build_model


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            out[labels == label] = 1
    return out


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = float(np.logical_and(pred_b, gt_b).sum())
    fp = float(np.logical_and(pred_b, ~gt_b).sum())
    fn = float(np.logical_and(~pred_b, gt_b).sum())
    pred_area = float(pred_b.sum())
    gt_area = float(gt_b.sum())
    union = float(np.logical_or(pred_b, gt_b).sum())
    eps = 1e-6
    return {
        "dice": float((2 * tp + eps) / (pred_area + gt_area + eps)),
        "iou": float((tp + eps) / (union + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_area": gt_area,
        "pred_area": pred_area,
    }


def failure_type(m: dict[str, float]) -> str:
    precision = float(m["precision"])
    recall = float(m["recall"])
    pred_area = int(m["pred_area"])
    gt_area = int(m["gt_area"])
    tp = int(m["tp"])
    dice = float(m["dice"])
    if gt_area > 0 and pred_area == 0:
        return "missed_all"
    if gt_area > 0 and pred_area > 0 and tp == 0:
        return "wrong_location_no_overlap"
    if recall < 0.30:
        return "mostly_missed_fn"
    if precision < 0.30:
        return "over_segmented_fp"
    if recall < 0.55:
        return "under_segmented_fn"
    if dice < 0.75:
        return "boundary_or_shift_error"
    return "moderate_error"


@torch.no_grad()
def collect_v18_errors(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ModelBuildConfig(args.model, in_channels=1, out_channels=1)).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        ds = QaTaNPZDataset(str(Path(args.manifest).resolve()), split=split)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        for batch_idx, batch in enumerate(dl):
            x = batch["image"].to(device, non_blocking=True)
            prob = torch.sigmoid(model(x)["seg"]).cpu().numpy()
            gt_batch = batch["mask"].numpy()
            for i in range(prob.shape[0]):
                ds_idx = batch_idx * args.batch + i
                item = ds.df.iloc[ds_idx]
                pred = remove_small_components((prob[i, 0] > args.threshold).astype(np.uint8), args.min_area)
                gt = (gt_batch[i, 0] > 0.5).astype(np.uint8)
                m = metrics(pred, gt)
                rows.append(
                    {
                        "id": str(item["id"]),
                        "split": split,
                        "npz_path": str(item["npz_path"]),
                        **m,
                        "failure_type": failure_type(m),
                    }
                )
    return pd.DataFrame(rows)


def select_profile(errors: pd.DataFrame, profile: str, low_thr: float) -> pd.Series:
    low = errors["dice"] < low_thr
    if profile == "actual_fp":
        return low & errors["failure_type"].isin(["over_segmented_fp", "wrong_location_no_overlap"])
    if profile == "actual_fn":
        return low & errors["failure_type"].isin(["missed_all", "mostly_missed_fn", "under_segmented_fn"])
    if profile == "actual_boundary":
        return low & errors["failure_type"].isin(["boundary_or_shift_error"])
    if profile == "actual_all":
        return low
    if profile == "actual_hard":
        return errors["dice"] < 0.50
    raise ValueError(f"Unknown profile: {profile}")


def write_manifest(
    source_manifest: pd.DataFrame,
    errors: pd.DataFrame,
    low_ids: set[str],
    profile: str,
    low_thr: float,
    out_root: Path,
) -> None:
    selected_ids: set[str] = set()
    for split in ("train", "val"):
        part = errors[errors["split"].eq(split)]
        selected_ids.update(part.loc[select_profile(part, profile, low_thr), "id"].astype(str).tolist())

    selected = source_manifest[
        (
            source_manifest["split"].isin(["train", "val"])
            & source_manifest["id"].astype(str).isin(selected_ids)
        )
        | (
            source_manifest["split"].eq("test")
            & source_manifest["id"].astype(str).isin(low_ids)
        )
    ].copy()

    target = out_root / f"qata_lowdice_{profile}"
    target.mkdir(parents=True, exist_ok=True)
    selected.to_csv(target / "manifest.csv", index=False, encoding="utf-8")
    profile_errors = errors[errors["id"].astype(str).isin(selected_ids | low_ids)].copy()
    profile_errors.to_csv(target / "v18_error_profile.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"{profile}: {target / 'manifest.csv'}")
    print(selected["split"].value_counts().to_string())
    print(errors.loc[errors["id"].astype(str).isin(selected_ids), "failure_type"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/qata/manifest.csv")
    parser.add_argument("--lowdice_csv", default="outputs/low_dice_v18_freq_ft8_lt060.csv")
    parser.add_argument("--model", default="transunet2d_v18")
    parser.add_argument("--ckpt", default="outputs/transunet2d_v18_freq_ft8_best.pt")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min_area", type=int, default=0)
    parser.add_argument("--low_thr", type=float, default=0.65)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_root", default="data/processed")
    args = parser.parse_args()

    source_manifest = pd.read_csv(args.manifest)
    errors = collect_v18_errors(args)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    errors.to_csv(out_root / "qata_v18_error_profile.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    low_df = pd.read_csv(args.lowdice_csv)
    low_ids = set(str(x) for x in low_df["case_id"].tolist())
    for profile in ("actual_fp", "actual_fn", "actual_boundary", "actual_all", "actual_hard"):
        write_manifest(source_manifest, errors, low_ids, profile, args.low_thr, out_root)


if __name__ == "__main__":
    main()
