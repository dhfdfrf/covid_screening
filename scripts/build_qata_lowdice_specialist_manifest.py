from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def mask_features(npz_path: Path) -> dict[str, float]:
    npz = np.load(npz_path)
    mask = (npz["mask"][0] > 0.5).astype(np.uint8)
    h, w = mask.shape
    area = int(mask.sum())
    if area <= 0:
        return {
            "gt_area": 0.0,
            "gt_area_ratio": 0.0,
            "gt_boundary_area_ratio": 0.0,
            "gt_side_balance": 0.0,
            "gt_bilateral": 0.0,
            "gt_bbox_area_ratio": 0.0,
            "lowlike_score": 0.0,
        }

    kernel = np.ones((3, 3), np.uint8)
    boundary = mask - cv2.erode(mask, kernel, iterations=1)
    yy, xx = np.where(mask > 0)
    bbox_w = (int(xx.max()) - int(xx.min()) + 1) / w
    bbox_h = (int(yy.max()) - int(yy.min()) + 1) / h
    left = int(mask[:, : w // 2].sum())
    right = int(mask[:, w // 2 :].sum())
    side_balance = abs(left - right) / max(area, 1)

    return {
        "gt_area": float(area),
        "gt_area_ratio": float(area / (h * w)),
        "gt_boundary_area_ratio": float(boundary.sum() / max(area, 1)),
        "gt_side_balance": float(side_balance),
        "gt_bilateral": float(left > 0 and right > 0),
        "gt_bbox_area_ratio": float(bbox_w * bbox_h),
    }


def score_lowlike(row: dict[str, float], args: argparse.Namespace) -> float:
    score = 0.0
    if row["gt_area_ratio"] <= args.area_ratio_thr:
        score += 1.0
    if row["gt_boundary_area_ratio"] > args.boundary_ratio_thr:
        score += 1.0
    if row["gt_side_balance"] > args.side_balance_thr:
        score += 1.0
    if row["gt_bbox_area_ratio"] <= args.bbox_area_ratio_thr:
        score += 1.0
    if row["gt_bilateral"] < 0.5:
        score += 0.5
    return score


def build_manifest(args: argparse.Namespace) -> None:
    src = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)

    feature_rows = []
    for idx, item in df.iterrows():
        npz_path = Path(str(item["npz_path"]))
        feats = mask_features(npz_path)
        feats["lowlike_score"] = score_lowlike(feats, args)
        feature_rows.append({"source_index": idx, "id": item["id"], "split": item["split"], **feats})
    features = pd.DataFrame(feature_rows)
    merged = df.join(features.drop(columns=["id", "split"]))

    selected_parts = []
    for split in ("train", "val"):
        min_score = args.train_min_score if split == "train" else args.val_min_score
        part = merged[merged["split"].eq(split) & (merged["lowlike_score"] >= min_score)].copy()
        selected_parts.append(part[df.columns])

    low_ids: set[str] = set()
    if args.lowdice_csv:
        low_df = pd.read_csv(args.lowdice_csv)
        if "case_id" in low_df.columns:
            low_ids.update(str(x) for x in low_df["case_id"].tolist())
        if "npz_path" in low_df.columns:
            low_ids.update(Path(str(x)).stem for x in low_df["npz_path"].tolist())

    if low_ids:
        test_part = merged[merged["split"].eq("test") & merged["id"].astype(str).isin(low_ids)].copy()
    else:
        test_part = merged[merged["split"].eq("test") & (merged["lowlike_score"] >= args.test_min_score)].copy()
    selected_parts.append(test_part[df.columns])

    selected = pd.concat(selected_parts, axis=0).reset_index(drop=True)
    selected.to_csv(out_dir / "manifest.csv", index=False, encoding="utf-8")
    features.to_csv(out_dir / "selection_features.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    print(f"source={src}")
    print(f"out_manifest={out_dir / 'manifest.csv'}")
    print(f"out_features={out_dir / 'selection_features.csv'}")
    print("selected_counts:")
    print(selected["split"].value_counts().to_string())
    print("source_lowlike_counts_by_split:")
    for split in ("train", "val", "test"):
        sub = merged[merged["split"].eq(split)]
        print(
            f"{split}: total={len(sub)} "
            f"score>=1={int((sub['lowlike_score'] >= 1).sum())} "
            f"score>=2={int((sub['lowlike_score'] >= 2).sum())} "
            f"score>=3={int((sub['lowlike_score'] >= 3).sum())}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/qata/manifest.csv")
    parser.add_argument("--out_dir", default="data/processed/qata_lowdice_specialist")
    parser.add_argument("--lowdice_csv", default="outputs/low_dice_v18_freq_ft8_lt060.csv")
    parser.add_argument("--area_ratio_thr", type=float, default=0.10236)
    parser.add_argument("--boundary_ratio_thr", type=float, default=0.119962)
    parser.add_argument("--side_balance_thr", type=float, default=0.4645)
    parser.add_argument("--bbox_area_ratio_thr", type=float, default=0.255344)
    parser.add_argument("--train_min_score", type=float, default=2.0)
    parser.add_argument("--val_min_score", type=float, default=2.0)
    parser.add_argument("--test_min_score", type=float, default=2.0)
    args = parser.parse_args()
    build_manifest(args)


if __name__ == "__main__":
    main()
