from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = _font(22, True)
FONT_TEXT = _font(15, False)
FONT_SMALL = _font(12, False)


def _norm_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def _load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    pred = (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)
    if pred.shape != shape:
        pred_img = Image.fromarray(pred * 255).resize((shape[1], shape[0]), Image.Resampling.NEAREST)
        pred = (np.array(pred_img) > 127).astype(np.uint8)
    return pred


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(m, kernel, iterations=1)
    return (m - eroded).astype(bool)


def _ring(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    m = mask.astype(np.uint8)
    kernel = np.ones((radius, radius), np.uint8)
    dilated = cv2.dilate(m, kernel, iterations=1).astype(bool)
    return np.logical_and(dilated, ~m.astype(bool))


def _components(mask: np.ndarray) -> tuple[int, int, float]:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return 0, 0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    largest = int(areas.max())
    total = int(areas.sum())
    return int(n_labels - 1), largest, float(largest / max(total, 1))


def _bbox_features(mask: np.ndarray) -> dict[str, float]:
    yy, xx = np.where(mask.astype(bool))
    h, w = mask.shape
    if len(xx) == 0:
        return {
            "bbox_w": 0.0,
            "bbox_h": 0.0,
            "bbox_area_ratio": 0.0,
            "bbox_aspect": 0.0,
            "centroid_x": 0.5,
            "centroid_y": 0.5,
            "bbox_center_x": 0.5,
            "bbox_center_y": 0.5,
        }
    x1, x2 = int(xx.min()), int(xx.max())
    y1, y2 = int(yy.min()), int(yy.max())
    bw = (x2 - x1 + 1) / w
    bh = (y2 - y1 + 1) / h
    return {
        "bbox_w": float(bw),
        "bbox_h": float(bh),
        "bbox_area_ratio": float(bw * bh),
        "bbox_aspect": float(bw / max(bh, 1e-6)),
        "centroid_x": float(xx.mean() / max(w - 1, 1)),
        "centroid_y": float(yy.mean() / max(h - 1, 1)),
        "bbox_center_x": float(((x1 + x2) / 2) / max(w - 1, 1)),
        "bbox_center_y": float(((y1 + y2) / 2) / max(h - 1, 1)),
    }


def _side_features(mask: np.ndarray, prefix: str) -> dict[str, float]:
    h, w = mask.shape
    m = mask.astype(bool)
    total = int(m.sum())
    left = int(m[:, : w // 2].sum())
    right = int(m[:, w // 2 :].sum())
    center = int(m[:, w // 3 : 2 * w // 3].sum())
    denom = max(total, 1)
    return {
        f"{prefix}_left_frac": float(left / denom),
        f"{prefix}_right_frac": float(right / denom),
        f"{prefix}_center_frac": float(center / denom),
        f"{prefix}_side_balance": float(abs(left - right) / denom),
        f"{prefix}_bilateral": float(left > 0 and right > 0),
    }


def _image_features(img: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    h, w = img.shape
    gt_bool = gt.astype(bool)
    ring = _ring(gt, radius=9)
    bg = ~gt_bool
    edge = cv2.Canny((img * 255).astype(np.uint8), 40, 110) > 0

    if gt_bool.any():
        lesion = img[gt_bool]
        lesion_mean = float(lesion.mean())
        lesion_std = float(lesion.std())
        lesion_edge = float(edge[gt_bool].mean())
    else:
        lesion_mean = 0.0
        lesion_std = 0.0
        lesion_edge = 0.0

    if ring.any():
        ring_mean = float(img[ring].mean())
        ring_std = float(img[ring].std())
        ring_edge = float(edge[ring].mean())
    else:
        ring_mean = float(img[bg].mean()) if bg.any() else 0.0
        ring_std = float(img[bg].std()) if bg.any() else 0.0
        ring_edge = float(edge[bg].mean()) if bg.any() else 0.0

    return {
        "image_mean": float(img.mean()),
        "image_std": float(img.std()),
        "lesion_mean": lesion_mean,
        "lesion_std": lesion_std,
        "ring_mean": ring_mean,
        "ring_std": ring_std,
        "lesion_ring_abs_contrast": float(abs(lesion_mean - ring_mean)),
        "lesion_ring_signed_contrast": float(lesion_mean - ring_mean),
        "lesion_edge_density": lesion_edge,
        "ring_edge_density": ring_edge,
        "global_edge_density": float(edge.mean()),
        "lesion_y_lower_half": float((_bbox_features(gt)["centroid_y"] > 0.5)),
        "image_h": float(h),
        "image_w": float(w),
    }


def _metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    tp = float(np.logical_and(gt_b, pred_b).sum())
    fp = float(np.logical_and(~gt_b, pred_b).sum())
    fn = float(np.logical_and(gt_b, ~pred_b).sum())
    gt_area = float(gt_b.sum())
    pred_area = float(pred_b.sum())
    union = float(np.logical_or(gt_b, pred_b).sum())
    eps = 1e-6
    return {
        "dice": float((2.0 * tp + eps) / (gt_area + pred_area + eps)),
        "iou": float((tp + eps) / (union + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def extract_case_features(index: int, npz_path: Path, pred_dir: Path) -> dict[str, float | str | int]:
    npz = np.load(npz_path)
    img = _norm_image(npz["img"])
    gt = (npz["mask"][0] > 0.5).astype(np.uint8)
    pred = _load_mask(pred_dir / f"{index:06d}_pred.png", gt.shape)

    h, w = gt.shape
    gt_area = int(gt.sum())
    pred_area = int(pred.sum())
    gt_comp, gt_largest, gt_largest_frac = _components(gt)
    pred_comp, pred_largest, pred_largest_frac = _components(pred)
    gt_boundary = int(_boundary(gt).sum())
    pred_boundary = int(_boundary(pred).sum())

    gt_bbox = {f"gt_{k}": v for k, v in _bbox_features(gt).items()}
    pred_bbox = {f"pred_{k}": v for k, v in _bbox_features(pred).items()}
    centroid_dist = math.sqrt(
        (float(gt_bbox["gt_centroid_x"]) - float(pred_bbox["pred_centroid_x"])) ** 2
        + (float(gt_bbox["gt_centroid_y"]) - float(pred_bbox["pred_centroid_y"])) ** 2
    )

    row: dict[str, float | str | int] = {
        "index": index,
        "case_id": npz_path.stem,
        "npz_path": str(npz_path.relative_to(ROOT) if npz_path.is_relative_to(ROOT) else npz_path),
        "gt_area": gt_area,
        "pred_area": pred_area,
        "gt_area_ratio": float(gt_area / (h * w)),
        "pred_area_ratio": float(pred_area / (h * w)),
        "area_ratio_error": float(abs(pred_area - gt_area) / max(gt_area, 1)),
        "gt_comp_count": gt_comp,
        "pred_comp_count": pred_comp,
        "gt_largest_area": gt_largest,
        "pred_largest_area": pred_largest,
        "gt_largest_frac": gt_largest_frac,
        "pred_largest_frac": pred_largest_frac,
        "gt_boundary_area_ratio": float(gt_boundary / max(gt_area, 1)),
        "pred_boundary_area_ratio": float(pred_boundary / max(pred_area, 1)),
        "centroid_dist": float(centroid_dist),
    }
    row.update(_metrics(gt, pred))
    row.update(gt_bbox)
    row.update(pred_bbox)
    row.update(_side_features(gt, "gt"))
    row.update(_side_features(pred, "pred"))
    row.update(_image_features(img, gt))
    row["side_balance_error"] = float(abs(float(row["gt_left_frac"]) - float(row["pred_left_frac"])))
    return row


def collect_features(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)
    manifest = pd.read_csv(data_dir / "manifest.csv")
    manifest = manifest[manifest["split"].eq(args.split)].reset_index(drop=True)

    rows = []
    for index, item in manifest.iterrows():
        pred_path = pred_dir / f"{index:06d}_pred.png"
        if not pred_path.exists():
            continue
        rows.append(extract_case_features(index, ROOT / str(item["npz_path"]), pred_dir))

    df = pd.DataFrame(rows)
    df["group"] = "middle"
    df.loc[df["dice"] < args.low_thr, "group"] = "low"
    df.loc[df["dice"] >= args.high_thr, "group"] = "high"
    return df


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "index",
        "group",
        "case_id",
        "npz_path",
        "dice",
        "iou",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
    }
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def intrinsic_feature_columns(cols: list[str]) -> list[str]:
    blocked_prefixes = ("pred_",)
    blocked_names = {
        "pred_area",
        "pred_area_ratio",
        "pred_comp_count",
        "pred_largest_area",
        "pred_largest_frac",
        "pred_boundary_area_ratio",
        "area_ratio_error",
        "centroid_dist",
        "side_balance_error",
    }
    return [
        col
        for col in cols
        if not col.startswith(blocked_prefixes) and col not in blocked_names
    ]


def prediction_error_feature_columns(cols: list[str]) -> list[str]:
    wanted_prefixes = ("pred_",)
    wanted_names = {
        "pred_area",
        "pred_area_ratio",
        "pred_comp_count",
        "pred_largest_area",
        "pred_largest_frac",
        "pred_boundary_area_ratio",
        "area_ratio_error",
        "centroid_dist",
        "side_balance_error",
    }
    return [
        col
        for col in cols
        if col.startswith(wanted_prefixes) or col in wanted_names
    ]


def summarize_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    low = df[df["group"].eq("low")]
    high = df[df["group"].eq("high")]
    rows = []
    for col in feature_cols:
        low_values = low[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        high_values = high[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(low_values) < 2 or len(high_values) < 2:
            continue
        low_mean = float(low_values.mean())
        high_mean = float(high_values.mean())
        low_std = float(low_values.std(ddof=1))
        high_std = float(high_values.std(ddof=1))
        pooled = math.sqrt((low_std * low_std + high_std * high_std) / 2.0)
        effect = (low_mean - high_mean) / max(pooled, 1e-8)
        rows.append(
            {
                "feature": col,
                "low_mean": low_mean,
                "high_mean": high_mean,
                "diff_low_minus_high": low_mean - high_mean,
                "effect_size": effect,
                "abs_effect_size": abs(effect),
                "low_median": float(low_values.median()),
                "high_median": float(high_values.median()),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_effect_size", ascending=False)


def best_threshold_rules(df: pd.DataFrame, feature_cols: list[str], max_rules: int = 30) -> pd.DataFrame:
    sub = df[df["group"].isin(["low", "high"])].copy()
    y = sub["group"].eq("low").to_numpy(dtype=np.uint8)
    rows = []
    for col in feature_cols:
        values = sub[col].astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()
        mask = np.isfinite(values)
        if mask.sum() < 10 or len(np.unique(values[mask])) < 2:
            continue
        vals = values[mask]
        labels = y[mask]
        candidates = np.unique(np.quantile(vals, np.linspace(0.05, 0.95, 19)))
        best = None
        for thr in candidates:
            for op in ("<=", ">"):
                pred_low = vals <= thr if op == "<=" else vals > thr
                tp = float(np.logical_and(pred_low, labels == 1).sum())
                tn = float(np.logical_and(~pred_low, labels == 0).sum())
                fp = float(np.logical_and(pred_low, labels == 0).sum())
                fn = float(np.logical_and(~pred_low, labels == 1).sum())
                tpr = tp / max(tp + fn, 1.0)
                tnr = tn / max(tn + fp, 1.0)
                bal_acc = (tpr + tnr) / 2.0
                precision = tp / max(tp + fp, 1.0)
                recall = tpr
                item = (bal_acc, precision, recall, op, float(thr), tp, fp, fn, tn)
                if best is None or item[0] > best[0]:
                    best = item
        if best is not None:
            bal_acc, precision, recall, op, thr, tp, fp, fn, tn = best
            rows.append(
                {
                    "feature": col,
                    "rule": f"{col} {op} {thr:.6g}",
                    "balanced_accuracy": bal_acc,
                    "precision_low": precision,
                    "recall_low": recall,
                    "tp_low": int(tp),
                    "fp_high_as_low": int(fp),
                    "fn_low_as_high": int(fn),
                    "tn_high": int(tn),
                }
            )
    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False).head(max_rules)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")


def draw_summary(summary: pd.DataFrame, rules: pd.DataFrame, counts: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1500, 920
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((34, 26), "Low-Dice vs High-Dice Feature Differences", font=FONT_TITLE, fill=(20, 24, 32))
    draw.text(
        (34, 58),
        f"low={counts.get('low', 0)}  high={counts.get('high', 0)}  middle={counts.get('middle', 0)}",
        font=FONT_TEXT,
        fill=(60, 60, 60),
    )

    top = summary.head(14).reset_index(drop=True)
    x0, y0 = 40, 105
    draw.text((x0, y0), "Top feature gaps by effect size", font=FONT_TEXT, fill=(25, 25, 25))
    y = y0 + 34
    max_abs = max(float(top["abs_effect_size"].max()) if len(top) else 1.0, 1e-6)
    for _, row in top.iterrows():
        feature = str(row["feature"])
        effect = float(row["effect_size"])
        bar_w = int(300 * abs(effect) / max_abs)
        color = (215, 70, 55) if effect > 0 else (55, 125, 210)
        draw.text((x0, y), feature[:34], font=FONT_SMALL, fill=(30, 30, 30))
        draw.rectangle((x0 + 310, y + 3, x0 + 310 + bar_w, y + 17), fill=color)
        text = (
            f"effect={effect:+.2f}  "
            f"low={float(row['low_mean']):.4f}  high={float(row['high_mean']):.4f}"
        )
        draw.text((x0 + 620, y), text, font=FONT_SMALL, fill=(45, 45, 45))
        y += 36

    x1, y1 = 40, 650
    draw.text((x1, y1), "Best simple rules for identifying low-Dice samples", font=FONT_TEXT, fill=(25, 25, 25))
    y = y1 + 34
    for _, row in rules.head(6).iterrows():
        text = (
            f"{row['rule']}    "
            f"bal_acc={float(row['balanced_accuracy']):.3f}  "
            f"precision={float(row['precision_low']):.3f}  recall={float(row['recall_low']):.3f}"
        )
        draw.text((x1, y), text[:150], font=FONT_SMALL, fill=(45, 45, 45))
        y += 31

    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/qata")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--low_thr", type=float, default=0.60)
    parser.add_argument("--high_thr", type=float, default=0.90)
    parser.add_argument("--out_dir", default="outputs/analysis/v18_low_high")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_features(args)
    feature_cols = numeric_feature_columns(df)
    summary = summarize_features(df, feature_cols)
    rules = best_threshold_rules(df, feature_cols)
    intrinsic_cols = intrinsic_feature_columns(feature_cols)
    prediction_cols = prediction_error_feature_columns(feature_cols)
    intrinsic_summary = summarize_features(df, intrinsic_cols)
    intrinsic_rules = best_threshold_rules(df, intrinsic_cols)
    prediction_summary = summarize_features(df, prediction_cols)
    prediction_rules = best_threshold_rules(df, prediction_cols)
    counts = df["group"].value_counts().to_dict()

    write_table(df, out_dir / "sample_features.csv")
    write_table(summary, out_dir / "feature_differences.csv")
    write_table(rules, out_dir / "low_high_rules.csv")
    write_table(intrinsic_summary, out_dir / "intrinsic_feature_differences.csv")
    write_table(intrinsic_rules, out_dir / "intrinsic_low_high_rules.csv")
    write_table(prediction_summary, out_dir / "prediction_error_differences.csv")
    write_table(prediction_rules, out_dir / "prediction_error_rules.csv")
    draw_summary(summary, rules, counts, out_dir / "feature_difference_summary.png")

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"pred_dir={args.pred_dir}\n")
        f.write(f"split={args.split}\n")
        f.write(f"low_thr={args.low_thr} high_thr={args.high_thr}\n")
        f.write(f"counts={counts}\n\n")
        f.write("Top feature differences:\n")
        for _, row in summary.head(15).iterrows():
            f.write(
                f"- {row['feature']}: effect={row['effect_size']:+.3f}, "
                f"low_mean={row['low_mean']:.6f}, high_mean={row['high_mean']:.6f}\n"
            )
        f.write("\nTop intrinsic image/GT differences:\n")
        for _, row in intrinsic_summary.head(15).iterrows():
            f.write(
                f"- {row['feature']}: effect={row['effect_size']:+.3f}, "
                f"low_mean={row['low_mean']:.6f}, high_mean={row['high_mean']:.6f}\n"
            )
        f.write("\nTop prediction-error differences:\n")
        for _, row in prediction_summary.head(15).iterrows():
            f.write(
                f"- {row['feature']}: effect={row['effect_size']:+.3f}, "
                f"low_mean={row['low_mean']:.6f}, high_mean={row['high_mean']:.6f}\n"
            )
        f.write("\nBest rules:\n")
        for _, row in rules.head(15).iterrows():
            f.write(
                f"- {row['rule']}: bal_acc={row['balanced_accuracy']:.3f}, "
                f"precision={row['precision_low']:.3f}, recall={row['recall_low']:.3f}\n"
            )
        f.write("\nBest intrinsic rules:\n")
        for _, row in intrinsic_rules.head(15).iterrows():
            f.write(
                f"- {row['rule']}: bal_acc={row['balanced_accuracy']:.3f}, "
                f"precision={row['precision_low']:.3f}, recall={row['recall_low']:.3f}\n"
            )

    print(f"samples={len(df)}")
    print(f"counts={counts}")
    print(f"out_dir={out_dir}")
    print("top_differences:")
    for _, row in summary.head(10).iterrows():
        print(
            f"{row['feature']}: effect={row['effect_size']:+.3f} "
            f"low={row['low_mean']:.6f} high={row['high_mean']:.6f}"
        )
    print("top_rules:")
    for _, row in rules.head(8).iterrows():
        print(
            f"{row['rule']} bal_acc={row['balanced_accuracy']:.3f} "
            f"precision={row['precision_low']:.3f} recall={row['recall_low']:.3f}"
        )
    print("top_intrinsic_differences:")
    for _, row in intrinsic_summary.head(8).iterrows():
        print(
            f"{row['feature']}: effect={row['effect_size']:+.3f} "
            f"low={row['low_mean']:.6f} high={row['high_mean']:.6f}"
        )
    print("top_intrinsic_rules:")
    for _, row in intrinsic_rules.head(8).iterrows():
        print(
            f"{row['rule']} bal_acc={row['balanced_accuracy']:.3f} "
            f"precision={row['precision_low']:.3f} recall={row['recall_low']:.3f}"
        )


if __name__ == "__main__":
    main()
