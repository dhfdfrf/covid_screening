from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_lowdice_expert_gate import (
    ExpertSpec,
    apply_threshold,
    collect_probs,
    default_experts,
    metric_arrays,
    parse_expert,
    parse_min_areas,
)


def candidate_weights(names: list[str]) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for name in names:
        candidates.append({name: 1.0})

    for a, b in itertools.combinations(names, 2):
        for wa in (0.25, 0.50, 0.75):
            candidates.append({a: wa, b: 1.0 - wa})

    for combo in itertools.combinations(names, 3):
        candidates.append({name: 1.0 / 3.0 for name in combo})
        for perm in set(itertools.permutations((0.50, 0.30, 0.20), 3)):
            candidates.append({name: weight for name, weight in zip(combo, perm)})

    if len(names) >= 4:
        candidates.append({name: 1.0 / len(names) for name in names})
        preferred = [
            {"broad": 0.55, "boundary": 0.25, "recall": 0.15, "precision": 0.05},
            {"broad": 0.50, "boundary": 0.35, "recall": 0.15},
            {"broad": 0.60, "boundary": 0.20, "recall": 0.20},
            {"broad": 0.65, "boundary": 0.25, "precision": 0.10},
            {"broad": 0.70, "recall": 0.20, "precision": 0.10},
        ]
        for item in preferred:
            filtered = {k: v for k, v in item.items() if k in names and v > 0}
            total = sum(filtered.values())
            if total > 0:
                candidates.append({k: v / total for k, v in filtered.items()})

    unique: dict[tuple[tuple[str, float], ...], dict[str, float]] = {}
    for weights in candidates:
        total = sum(weights.values())
        normalized = {k: float(v) / total for k, v in weights.items() if v > 0}
        key = tuple(sorted((k, round(v, 4)) for k, v in normalized.items()))
        unique[key] = normalized
    return list(unique.values())


def parse_weights(text: str, names: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in text.split(";"):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in names:
            raise ValueError(f"Unknown model in --weights: {key}")
        weights[key] = float(value)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("--weights must sum to a positive value")
    return {key: value / total for key, value in weights.items() if value > 0}


def ensemble_prob(prob_by_name: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    out: np.ndarray | None = None
    for name, weight in weights.items():
        part = prob_by_name[name] * float(weight)
        out = part if out is None else out + part
    assert out is not None
    return out


def tune_setting(
    prob: np.ndarray,
    gt: np.ndarray,
    thresholds: np.ndarray,
    min_areas: list[int],
) -> tuple[float, int, float]:
    best = (float(thresholds[0]), int(min_areas[0]), -1.0)
    for threshold in thresholds:
        for min_area in min_areas:
            pred = apply_threshold(prob, float(threshold), int(min_area))
            dice = float(metric_arrays(pred, gt)["dice"].mean())
            if dice > best[2]:
                best = (float(threshold), int(min_area), dice)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", default="outputs/low_dice_v18_freq_ft8_lt060.csv")
    parser.add_argument("--expert", action="append", default=[], help="name:data_dir:model:ckpt")
    parser.add_argument("--data_dir", default="data/processed/qata_lowdice_specialist")
    parser.add_argument("--tune_split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--eval_split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tta_mode", default="none", choices=("none", "h", "v", "hv", "all"))
    parser.add_argument("--thr_min", type=float, default=0.05)
    parser.add_argument("--thr_max", type=float, default=0.90)
    parser.add_argument("--thr_step", type=float, default=0.05)
    parser.add_argument("--min_areas", default="0,16,32,64,128,256")
    parser.add_argument("--weights", default="", help="Optional fixed ensemble, e.g. broad=0.5;boundary=0.3;recall=0.2")
    parser.add_argument("--out_csv", default="outputs/lowdice_soft_ensemble_eval.csv")
    parser.add_argument("--summary", default="outputs/lowdice_soft_ensemble_eval_summary.txt")
    args = parser.parse_args()

    base_specs = [parse_expert(item) for item in args.expert] if args.expert else default_experts()
    data_dir = Path(args.data_dir)
    specs = [
        ExpertSpec(spec.name, data_dir, spec.model, spec.ckpt)
        for spec in base_specs
        if spec.ckpt.exists() and (data_dir / "manifest.csv").exists()
    ]
    if not specs:
        raise FileNotFoundError("No expert checkpoints/manifests were found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = [spec.name for spec in specs]
    print(f"device={device}")
    print("experts=" + ",".join(names))

    val_prob: dict[str, np.ndarray] = {}
    test_prob: dict[str, np.ndarray] = {}
    val_gt: np.ndarray | None = None
    test_gt: np.ndarray | None = None
    test_ids: list[str] | None = None

    for spec in specs:
        ids_val, prob_val, gt_val = collect_probs(
            spec, args.tune_split, args.batch, args.num_workers, args.tta_mode, device
        )
        del ids_val
        ids_test, prob_test, gt_test = collect_probs(
            spec, args.eval_split, args.batch, args.num_workers, args.tta_mode, device
        )
        val_prob[spec.name] = prob_val
        test_prob[spec.name] = prob_test
        val_gt = gt_val if val_gt is None else val_gt
        test_gt = gt_test if test_gt is None else test_gt
        test_ids = ids_test if test_ids is None else test_ids

    assert val_gt is not None and test_gt is not None and test_ids is not None
    thresholds = np.arange(args.thr_min, args.thr_max + args.thr_step / 2.0, args.thr_step)
    min_areas = parse_min_areas(args.min_areas)

    rows = []
    best_row: dict[str, object] | None = None
    weight_candidates = [parse_weights(args.weights, names)] if args.weights else candidate_weights(names)
    for weights in weight_candidates:
        prob = ensemble_prob(val_prob, weights)
        threshold, min_area, val_dice = tune_setting(prob, val_gt, thresholds, min_areas)
        test_ens = ensemble_prob(test_prob, weights)
        test_pred = apply_threshold(test_ens, threshold, min_area)
        metrics = metric_arrays(test_pred, test_gt)
        mean_dice = float(metrics["dice"].mean())
        row = {
            "name": "+".join(f"{k}{v:.2f}" for k, v in sorted(weights.items())),
            "weights": ";".join(f"{k}={v:.4f}" for k, v in sorted(weights.items())),
            "threshold": threshold,
            "min_area": min_area,
            "val_dice": val_dice,
            "test_dice": mean_dice,
            "test_low_lt_060": int((metrics["dice"] < 0.60).sum()),
            "test_zero": int((metrics["dice"] <= 1e-9).sum()),
            "test_precision": float(metrics["precision"].mean()),
            "test_recall": float(metrics["recall"].mean()),
        }
        rows.append(row)
        if best_row is None or mean_dice > float(best_row["test_dice"]):
            best_row = row

    results = pd.DataFrame(rows).sort_values("test_dice", ascending=False)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    assert best_row is not None
    best_weights = dict(item.split("=") for item in str(best_row["weights"]).split(";"))
    best_weights = {k: float(v) for k, v in best_weights.items()}
    best_prob = ensemble_prob(test_prob, best_weights)
    best_pred = apply_threshold(best_prob, float(best_row["threshold"]), int(best_row["min_area"]))
    best_metrics = metric_arrays(best_pred, test_gt)

    per_case = pd.DataFrame({"case_id": test_ids})
    for key, value in best_metrics.items():
        per_case[f"ensemble_{key}"] = value
    baseline = pd.read_csv(args.baseline_csv).rename(columns={"dice": "v18_dice"})
    per_case = baseline.merge(per_case, on="case_id", how="inner")
    per_case["ensemble_delta"] = per_case["ensemble_dice"] - per_case["v18_dice"]
    per_case_path = out_path.with_name(out_path.stem + "_best_cases.csv")
    per_case.to_csv(per_case_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    summary_path = Path(args.summary)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"best={best_row['name']}\n")
        f.write(f"weights={best_row['weights']}\n")
        f.write(f"threshold={float(best_row['threshold']):.3f}\n")
        f.write(f"min_area={int(best_row['min_area'])}\n")
        f.write(f"val_dice={float(best_row['val_dice']):.4f}\n")
        f.write(f"test_dice={float(best_row['test_dice']):.4f}\n")
        f.write(f"test_low_lt_060={int(best_row['test_low_lt_060'])}\n")
        f.write(f"test_zero={int(best_row['test_zero'])}\n")
        f.write(f"v18_mean={per_case['v18_dice'].mean():.4f}\n")
        f.write(f"delta_vs_v18={per_case['ensemble_delta'].mean():+.4f}\n")
        f.write(f"improved_vs_v18={(per_case['ensemble_delta'] > 1e-9).sum()}\n")
        f.write(f"worse_vs_v18={(per_case['ensemble_delta'] < -1e-9).sum()}\n")
        f.write("\nTop 12 ensembles:\n")
        for _, row in results.head(12).iterrows():
            f.write(
                f"{row['name']}: test={row['test_dice']:.4f} val={row['val_dice']:.4f} "
                f"thr={row['threshold']:.3f} min_area={int(row['min_area'])} "
                f"low={int(row['test_low_lt_060'])} zero={int(row['test_zero'])}\n"
            )

    print(f"csv={out_path}")
    print(f"cases={per_case_path}")
    print(f"summary={summary_path}")
    print(f"best={best_row['name']} test_dice={float(best_row['test_dice']):.4f}")


if __name__ == "__main__":
    main()
