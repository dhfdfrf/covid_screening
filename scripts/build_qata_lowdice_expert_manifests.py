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
            "gt_comp_count": 0.0,
            "gt_largest_frac": 0.0,
        }

    kernel = np.ones((3, 3), np.uint8)
    boundary = mask - cv2.erode(mask, kernel, iterations=1)
    yy, xx = np.where(mask > 0)
    bbox_w = (int(xx.max()) - int(xx.min()) + 1) / w
    bbox_h = (int(yy.max()) - int(yy.min()) + 1) / h
    left = int(mask[:, : w // 2].sum())
    right = int(mask[:, w // 2 :].sum())
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        comp_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_frac = float(comp_areas.max() / max(area, 1))
    else:
        largest_frac = 0.0

    return {
        "gt_area": float(area),
        "gt_area_ratio": float(area / (h * w)),
        "gt_boundary_area_ratio": float(boundary.sum() / max(area, 1)),
        "gt_side_balance": float(abs(left - right) / max(area, 1)),
        "gt_bilateral": float(left > 0 and right > 0),
        "gt_bbox_area_ratio": float(bbox_w * bbox_h),
        "gt_comp_count": float(max(n_labels - 1, 0)),
        "gt_largest_frac": largest_frac,
    }


def profile_mask(features: pd.DataFrame, profile: str) -> pd.Series:
    if profile == "precision":
        return (
            (features["gt_area_ratio"] <= 0.055)
            & (
                (features["gt_side_balance"] >= 0.45)
                | (features["gt_bbox_area_ratio"] <= 0.22)
                | (features["gt_boundary_area_ratio"] >= 0.14)
            )
        )
    if profile == "recall":
        return (
            (features["gt_area_ratio"] >= 0.045)
            & (features["gt_area_ratio"] <= 0.135)
            & (
                (features["gt_bbox_area_ratio"] <= 0.36)
                | (features["gt_side_balance"] >= 0.35)
                | (features["gt_boundary_area_ratio"] >= 0.10)
            )
        )
    if profile == "boundary":
        return (
            (features["gt_boundary_area_ratio"] >= 0.105)
            & (features["gt_area_ratio"] >= 0.018)
            & (features["gt_area_ratio"] <= 0.14)
        )
    raise ValueError(f"Unknown profile: {profile}")


def build_one(df: pd.DataFrame, features: pd.DataFrame, low_ids: set[str], profile: str, out_dir: Path) -> None:
    mask = profile_mask(features, profile)
    merged = df.join(features.drop(columns=["id", "split"]))
    selected_parts = []
    for split in ("train", "val"):
        selected_parts.append(merged[merged["split"].eq(split) & mask].copy()[df.columns])
    selected_parts.append(merged[merged["split"].eq("test") & merged["id"].astype(str).isin(low_ids)].copy()[df.columns])
    selected = pd.concat(selected_parts, axis=0).reset_index(drop=True)

    target = out_dir / f"qata_lowdice_{profile}"
    target.mkdir(parents=True, exist_ok=True)
    selected.to_csv(target / "manifest.csv", index=False, encoding="utf-8")
    features.to_csv(target / "selection_features.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"{profile}: {target / 'manifest.csv'}")
    print(selected["split"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/qata/manifest.csv")
    parser.add_argument("--lowdice_csv", default="outputs/low_dice_v18_freq_ft8_lt060.csv")
    parser.add_argument("--out_root", default="data/processed")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    feature_rows = []
    for idx, row in df.iterrows():
        feature_rows.append({"id": row["id"], "split": row["split"], **mask_features(Path(str(row["npz_path"])))})
    features = pd.DataFrame(feature_rows)

    low_df = pd.read_csv(args.lowdice_csv)
    low_ids = set(str(x) for x in low_df["case_id"].tolist())
    out_root = Path(args.out_root)
    for profile in ("precision", "recall", "boundary"):
        build_one(df, features, low_ids, profile, out_root)


if __name__ == "__main__":
    main()
