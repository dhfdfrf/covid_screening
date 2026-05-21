from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    data_dir: Path
    model: str
    ckpt: Path


def default_experts() -> list[ExpertSpec]:
    return [
        ExpertSpec(
            name="broad",
            data_dir=Path("data/processed/qata_lowdice_specialist"),
            model="lowdice_refinenet2d",
            ckpt=Path("outputs/lowdice_refinenet2d_slim_b32_ft20_best.pt"),
        ),
        ExpertSpec(
            name="precision",
            data_dir=Path("data/processed/qata_lowdice_precision"),
            model="lowdice_refinenet2d",
            ckpt=Path("outputs/lowdice2_precision_ft20_best.pt"),
        ),
        ExpertSpec(
            name="recall",
            data_dir=Path("data/processed/qata_lowdice_recall"),
            model="lowdice_refinenet2d",
            ckpt=Path("outputs/lowdice3_recall_ft20_best.pt"),
        ),
        ExpertSpec(
            name="boundary",
            data_dir=Path("data/processed/qata_lowdice_boundary"),
            model="lowdice_refinenet2d",
            ckpt=Path("outputs/lowdice4_boundary_ft20_best.pt"),
        ),
    ]


def parse_expert(text: str) -> ExpertSpec:
    parts = text.split(":")
    if len(parts) != 4:
        raise ValueError("--expert must be name:data_dir:model:ckpt")
    return ExpertSpec(parts[0], Path(parts[1]), parts[2], Path(parts[3]))


def parse_min_areas(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_named_model(name: str) -> torch.nn.Module:
    return build_model(ModelBuildConfig(name=name, in_channels=1, out_channels=1))


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: torch.Tensor, tta_mode: str) -> torch.Tensor:
    out = model(x)
    seg = out["seg"] if isinstance(out, dict) else out
    if tta_mode == "none":
        return seg

    logits = [seg]
    if tta_mode in ("h", "all"):
        out_h = model(torch.flip(x, [3]))
        seg_h = out_h["seg"] if isinstance(out_h, dict) else out_h
        logits.append(torch.flip(seg_h, [3]))
    if tta_mode in ("v", "all"):
        out_v = model(torch.flip(x, [2]))
        seg_v = out_v["seg"] if isinstance(out_v, dict) else out_v
        logits.append(torch.flip(seg_v, [2]))
    if tta_mode in ("hv", "all"):
        out_hv = model(torch.flip(x, [2, 3]))
        seg_hv = out_hv["seg"] if isinstance(out_hv, dict) else out_hv
        logits.append(torch.flip(seg_hv, [2, 3]))
    return torch.stack(logits, dim=0).mean(dim=0)


def load_state(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)


def collect_probs(
    spec: ExpertSpec,
    split: str,
    batch_size: int,
    num_workers: int,
    tta_mode: str,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    ds = QaTaNPZDataset(str(spec.data_dir / "manifest.csv"), split=split)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = build_named_model(spec.model).to(device)
    load_state(model, spec.ckpt)
    model.eval()

    probs: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(device, non_blocking=True)
            logits = predict_logits(model, x, tta_mode)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            gts.append(batch["mask"].numpy())

    ids = ds.df["id"].astype(str).tolist()
    return ids, np.concatenate(probs, axis=0), np.concatenate(gts, axis=0).astype(np.uint8)


def remove_small_components_one(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            out[labels == label] = 1
    return out


def apply_threshold(prob: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    raw = (prob > threshold).astype(np.uint8)
    if min_area <= 0:
        return raw
    out = np.zeros_like(raw, dtype=np.uint8)
    for i in range(raw.shape[0]):
        out[i, 0] = remove_small_components_one(raw[i, 0], min_area)
    return out


def metric_arrays(pred: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = np.logical_and(pred_b, gt_b).sum(axis=(1, 2, 3)).astype(np.float64)
    fp = np.logical_and(pred_b, ~gt_b).sum(axis=(1, 2, 3)).astype(np.float64)
    fn = np.logical_and(~pred_b, gt_b).sum(axis=(1, 2, 3)).astype(np.float64)
    pred_area = pred_b.sum(axis=(1, 2, 3)).astype(np.float64)
    gt_area = gt_b.sum(axis=(1, 2, 3)).astype(np.float64)
    union = np.logical_or(pred_b, gt_b).sum(axis=(1, 2, 3)).astype(np.float64)
    eps = 1e-6
    return {
        "dice": (2.0 * tp + eps) / (pred_area + gt_area + eps),
        "iou": (tp + eps) / (union + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_area": pred_area,
        "gt_area": gt_area,
    }


def tune_threshold(
    prob: np.ndarray,
    gt: np.ndarray,
    thresholds: np.ndarray,
    min_areas: list[int],
) -> tuple[float, int, float]:
    best_threshold = float(thresholds[0])
    best_min_area = int(min_areas[0])
    best_dice = -1.0
    for threshold in thresholds:
        for min_area in min_areas:
            pred = apply_threshold(prob, float(threshold), int(min_area))
            mean_dice = float(metric_arrays(pred, gt)["dice"].mean())
            if mean_dice > best_dice:
                best_threshold = float(threshold)
                best_min_area = int(min_area)
                best_dice = mean_dice
    return best_threshold, best_min_area, best_dice


def evaluate_expert(
    spec: ExpertSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    thresholds = np.arange(args.thr_min, args.thr_max + args.thr_step / 2.0, args.thr_step)
    min_areas = parse_min_areas(args.min_areas)

    tune_ids, tune_prob, tune_gt = collect_probs(
        spec,
        split=args.tune_split,
        batch_size=args.batch,
        num_workers=args.num_workers,
        tta_mode=args.tta_mode,
        device=device,
    )
    del tune_ids
    threshold, min_area, tune_dice = tune_threshold(tune_prob, tune_gt, thresholds, min_areas)

    eval_ids, eval_prob, eval_gt = collect_probs(
        spec,
        split=args.eval_split,
        batch_size=args.batch,
        num_workers=args.num_workers,
        tta_mode=args.tta_mode,
        device=device,
    )
    pred = apply_threshold(eval_prob, threshold, min_area)
    metrics = metric_arrays(pred, eval_gt)

    rows = {"case_id": eval_ids}
    for key, value in metrics.items():
        rows[f"{spec.name}_{key}"] = value
    df = pd.DataFrame(rows)
    info: dict[str, float | int | str] = {
        "name": spec.name,
        "threshold": threshold,
        "min_area": min_area,
        "tune_dice": tune_dice,
        "eval_dice": float(df[f"{spec.name}_dice"].mean()),
        "eval_low_lt_060": int((df[f"{spec.name}_dice"] < 0.60).sum()),
        "eval_zero": int((df[f"{spec.name}_dice"] <= 1e-9).sum()),
        "ckpt": str(spec.ckpt),
    }
    return df, info


def gate_choice(failure_type: str, available: set[str]) -> str:
    routing = {
        "over_segmented_fp": ("actual_fp", "actual_all", "precision", "broad"),
        "wrong_location_no_overlap": ("actual_boundary", "precision", "actual_all", "broad"),
        "under_segmented_fn": ("actual_fn", "boundary", "recall", "broad"),
        "mostly_missed_fn": ("recall", "actual_fn", "boundary", "broad"),
        "missed_all": ("actual_all", "actual_fn", "broad", "recall"),
        "boundary_or_shift_error": ("basev18", "actual_all", "broad", "actual_boundary", "boundary"),
        "moderate_error": ("basev18", "actual_all", "broad", "boundary"),
    }.get(str(failure_type), ("broad",))
    for preferred in routing:
        if preferred in available:
            return preferred
    if "broad" in available:
        return "broad"
    return sorted(available)[0]


def add_gate_columns(merged: pd.DataFrame, expert_names: list[str]) -> pd.DataFrame:
    available = set(expert_names)
    merged["failure_type_gate_model"] = [
        gate_choice(ft, available) for ft in merged.get("failure_type", pd.Series([""] * len(merged)))
    ]
    merged["failure_type_gate_dice"] = [
        float(row[f"{row['failure_type_gate_model']}_dice"]) for _, row in merged.iterrows()
    ]

    dice_cols = ["v18_dice"] + [f"{name}_dice" for name in expert_names]
    merged["oracle_best_model"] = merged[dice_cols].idxmax(axis=1).str.replace("_dice", "", regex=False)
    merged["oracle_best_dice"] = merged[dice_cols].max(axis=1)

    expert_cols = [f"{name}_dice" for name in expert_names]
    merged["expert_oracle_model"] = merged[expert_cols].idxmax(axis=1).str.replace("_dice", "", regex=False)
    merged["expert_oracle_dice"] = merged[expert_cols].max(axis=1)
    return merged


def summarize_metric(df: pd.DataFrame, column: str, baseline: str = "v18_dice") -> dict[str, float | int]:
    delta = df[column] - df[baseline]
    return {
        "mean_dice": float(df[column].mean()),
        "median_dice": float(df[column].median()),
        "low_lt_060": int((df[column] < 0.60).sum()),
        "zero": int((df[column] <= 1e-9).sum()),
        "mean_delta_vs_v18": float(delta.mean()),
        "improved_vs_v18": int((delta > 1e-9).sum()),
        "worse_vs_v18": int((delta < -1e-9).sum()),
        "delta_gt_010": int((delta > 0.10).sum()),
        "delta_gt_020": int((delta > 0.20).sum()),
    }


def write_summary(
    path: Path,
    merged: pd.DataFrame,
    expert_infos: list[dict[str, float | int | str]],
    expert_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"n={len(merged)}\n")
        f.write("\nFailure types from v18 low-dice analysis:\n")
        for failure_type, count in merged["failure_type"].value_counts().items():
            mean_v18 = merged.loc[merged["failure_type"].eq(failure_type), "v18_dice"].mean()
            f.write(f"{failure_type}: count={count} v18_mean_dice={mean_v18:.4f}\n")

        f.write("\nExpert threshold settings tuned on validation split:\n")
        for info in expert_infos:
            f.write(
                f"{info['name']}: threshold={float(info['threshold']):.3f} "
                f"min_area={int(info['min_area'])} tune_dice={float(info['tune_dice']):.4f} "
                f"eval_dice={float(info['eval_dice']):.4f} "
                f"low_lt_060={int(info['eval_low_lt_060'])} zero={int(info['eval_zero'])}\n"
            )

        metric_cols = ["v18_dice"] + [f"{name}_dice" for name in expert_names]
        metric_cols += ["failure_type_gate_dice", "expert_oracle_dice", "oracle_best_dice"]
        f.write("\nOverall metrics on v18 low-dice test subset:\n")
        for col in metric_cols:
            label = col.replace("_dice", "")
            summary = summarize_metric(merged, col)
            f.write(
                f"{label}: mean={summary['mean_dice']:.4f} median={summary['median_dice']:.4f} "
                f"low_lt_060={summary['low_lt_060']} zero={summary['zero']} "
                f"delta={summary['mean_delta_vs_v18']:+.4f} "
                f"improved={summary['improved_vs_v18']} worse={summary['worse_vs_v18']} "
                f"delta_gt_010={summary['delta_gt_010']} delta_gt_020={summary['delta_gt_020']}\n"
            )

        f.write("\nGate model distribution:\n")
        for model, count in merged["failure_type_gate_model"].value_counts().items():
            f.write(f"{model}: {count}\n")

        f.write("\nOracle best model distribution:\n")
        for model, count in merged["oracle_best_model"].value_counts().items():
            f.write(f"{model}: {count}\n")

        f.write("\nLowest cases after failure-type gate:\n")
        low = merged.sort_values("failure_type_gate_dice").head(20)
        for _, row in low.iterrows():
            f.write(
                f"case={row['case_id']} type={row.get('failure_type', '')} "
                f"v18={row['v18_dice']:.4f} gate={row['failure_type_gate_dice']:.4f} "
                f"gate_model={row['failure_type_gate_model']} "
                f"oracle={row['oracle_best_dice']:.4f}/{row['oracle_best_model']}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", default="outputs/low_dice_v18_freq_ft8_lt060.csv")
    parser.add_argument("--expert", action="append", default=[], help="name:data_dir:model:ckpt")
    parser.add_argument("--tune_split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--eval_split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tta_mode", default="none", choices=("none", "h", "v", "hv", "all"))
    parser.add_argument("--thr_min", type=float, default=0.05)
    parser.add_argument("--thr_max", type=float, default=0.90)
    parser.add_argument("--thr_step", type=float, default=0.05)
    parser.add_argument("--min_areas", default="0,16,32,64,128,256")
    parser.add_argument("--out_csv", default="outputs/lowdice_expert_gate_eval.csv")
    parser.add_argument("--summary", default="outputs/lowdice_expert_gate_eval_summary.txt")
    args = parser.parse_args()

    specs = [parse_expert(item) for item in args.expert] if args.expert else default_experts()
    specs = [spec for spec in specs if spec.ckpt.exists() and (spec.data_dir / "manifest.csv").exists()]
    if not specs:
        raise FileNotFoundError("No expert checkpoints/manifests were found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print("experts=" + ",".join(spec.name for spec in specs))

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

    merged = baseline.copy()
    expert_infos: list[dict[str, float | int | str]] = []
    expert_names: list[str] = []
    for spec in specs:
        print(f"evaluating {spec.name} ...")
        expert_df, info = evaluate_expert(spec, args, device)
        merged = merged.merge(expert_df, on="case_id", how="inner")
        expert_infos.append(info)
        expert_names.append(spec.name)
        print(
            f"{spec.name}: threshold={float(info['threshold']):.3f} "
            f"min_area={int(info['min_area'])} eval_dice={float(info['eval_dice']):.4f}"
        )

    merged = add_gate_columns(merged, expert_names)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig", float_format="%.6f")

    summary_path = Path(args.summary)
    write_summary(summary_path, merged, expert_infos, expert_names)
    print(f"csv={out_csv}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
