from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    pred_area = float(pred.sum())
    gt_area = float(gt.sum())
    eps = 1e-6
    return {
        "dice": float((2 * tp + eps) / (pred_area + gt_area + eps)),
        "iou": float((tp + eps) / (union + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "pred_area": pred_area,
        "gt_area": gt_area,
    }


def collect_specialist(data_dir: Path, pred_dir: Path, min_area: int) -> pd.DataFrame:
    manifest = pd.read_csv(data_dir / "manifest.csv")
    test = manifest[manifest["split"].eq("test")].reset_index(drop=True)
    rows = []
    for idx, item in test.iterrows():
        npz = np.load(Path(str(item["npz_path"])))
        gt = (npz["mask"][0] > 0.5).astype(np.uint8)
        pred = (np.array(Image.open(pred_dir / f"{idx:06d}_pred.png").convert("L")) > 127).astype(np.uint8)
        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred * 255).resize((gt.shape[1], gt.shape[0]), Image.Resampling.NEAREST)
            )
            pred = (pred > 127).astype(np.uint8)
        pred = remove_small_components(pred, min_area)
        rows.append({"case_id": str(item["id"]), "specialist_index": idx, **metrics(pred, gt)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", default="outputs/low_dice_v18_freq_ft8_lt060.csv")
    parser.add_argument("--data_dir", default="data/processed/qata_lowdice_specialist")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--min_area", type=int, default=0)
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_csv)
    baseline = baseline.rename(
        columns={
            "dice": "v18_dice",
            "iou": "v18_iou",
            "precision": "v18_precision",
            "recall": "v18_recall",
            "pred_area": "v18_pred_area",
            "gt_area": "v18_gt_area",
        }
    )
    spec = collect_specialist(Path(args.data_dir), Path(args.pred_dir), args.min_area)
    spec = spec.rename(
        columns={
            "dice": "specialist_dice",
            "iou": "specialist_iou",
            "precision": "specialist_precision",
            "recall": "specialist_recall",
            "pred_area": "specialist_pred_area",
            "gt_area": "specialist_gt_area",
        }
    )
    merged = baseline.merge(spec, on="case_id", how="inner")
    merged["dice_delta"] = merged["specialist_dice"] - merged["v18_dice"]
    merged["recall_delta"] = merged["specialist_recall"] - merged["v18_recall"]
    merged["precision_delta"] = merged["specialist_precision"] - merged["v18_precision"]
    merged["improved"] = merged["dice_delta"] > 0
    merged = merged.sort_values("dice_delta", ascending=False)

    out_csv = Path(args.out_csv or Path(args.pred_dir) / "specialist_vs_v18_lowdice.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig", float_format="%.6f")

    summary = {
        "n": len(merged),
        "v18_mean_dice": float(merged["v18_dice"].mean()),
        "specialist_mean_dice": float(merged["specialist_dice"].mean()),
        "mean_delta": float(merged["dice_delta"].mean()),
        "median_delta": float(merged["dice_delta"].median()),
        "improved_count": int((merged["dice_delta"] > 0).sum()),
        "worse_count": int((merged["dice_delta"] < 0).sum()),
        "same_count": int((merged["dice_delta"] == 0).sum()),
        "v18_zero_count": int((merged["v18_dice"] <= 1e-9).sum()),
        "specialist_zero_count": int((merged["specialist_dice"] <= 1e-9).sum()),
        "delta_gt_0_10": int((merged["dice_delta"] > 0.10).sum()),
        "delta_gt_0_20": int((merged["dice_delta"] > 0.20).sum()),
        "delta_lt_minus_0_10": int((merged["dice_delta"] < -0.10).sum()),
    }
    summary_path = Path(args.summary or Path(args.pred_dir) / "specialist_vs_v18_lowdice_summary.txt")
    with summary_path.open("w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}={value}\n")
        f.write("\nTop improved:\n")
        for _, row in merged.head(12).iterrows():
            f.write(
                f"case={row['case_id']} v18={row['v18_dice']:.4f} "
                f"specialist={row['specialist_dice']:.4f} delta={row['dice_delta']:+.4f} "
                f"type={row.get('failure_type', '')}\n"
            )
        f.write("\nTop worse:\n")
        for _, row in merged.tail(12).sort_values("dice_delta").iterrows():
            f.write(
                f"case={row['case_id']} v18={row['v18_dice']:.4f} "
                f"specialist={row['specialist_dice']:.4f} delta={row['dice_delta']:+.4f} "
                f"type={row.get('failure_type', '')}\n"
            )

    print(f"csv={out_csv}")
    print(f"summary={summary_path}")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
