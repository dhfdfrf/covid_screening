from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

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
    tune_threshold,
)


def mask_features(prefix: str, prob: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for i in range(prob.shape[0]):
        p = prob[i, 0].astype(np.float32)
        m = pred[i, 0].astype(np.uint8)
        h, w = m.shape
        area = float(m.sum())
        soft_area = float(p.sum())
        entropy = float((-(p * np.log(p + 1e-6) + (1.0 - p) * np.log(1.0 - p + 1e-6))).mean())

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        comp_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32) if n_labels > 1 else np.array([], dtype=np.float32)
        largest = float(comp_areas.max()) if comp_areas.size else 0.0
        small_area = float(comp_areas[comp_areas < 64].sum()) if comp_areas.size else 0.0

        if area > 0:
            ys, xs = np.where(m > 0)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox_w = float((x1 - x0 + 1) / max(w, 1))
            bbox_h = float((y1 - y0 + 1) / max(h, 1))
            bbox_area = float((x1 - x0 + 1) * (y1 - y0 + 1))
            cx = float(xs.mean() / max(w - 1, 1))
            cy = float(ys.mean() / max(h - 1, 1))
            touch = float(x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1)
        else:
            bbox_w = bbox_h = bbox_area = cx = cy = touch = 0.0

        rows.append(
            {
                f"{prefix}_area": area,
                f"{prefix}_area_ratio": area / float(h * w),
                f"{prefix}_soft_area": soft_area,
                f"{prefix}_soft_area_ratio": soft_area / float(h * w),
                f"{prefix}_prob_mean": float(p.mean()),
                f"{prefix}_prob_std": float(p.std()),
                f"{prefix}_prob_max": float(p.max()),
                f"{prefix}_prob_p95": float(np.percentile(p, 95)),
                f"{prefix}_entropy": entropy,
                f"{prefix}_n_components": float(max(n_labels - 1, 0)),
                f"{prefix}_largest_component": largest,
                f"{prefix}_largest_ratio": largest / max(area, 1.0),
                f"{prefix}_small_component_ratio": small_area / max(area, 1.0),
                f"{prefix}_bbox_area_ratio": bbox_area / float(h * w),
                f"{prefix}_bbox_w_ratio": bbox_w,
                f"{prefix}_bbox_h_ratio": bbox_h,
                f"{prefix}_center_x": cx,
                f"{prefix}_center_y": cy,
                f"{prefix}_touch_border": touch,
            }
        )
    return pd.DataFrame(rows)


def pair_features(names: list[str], preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    n = next(iter(preds.values())).shape[0]
    for i in range(n):
        row: dict[str, float] = {}
        masks = {name: preds[name][i, 0].astype(bool) for name in names}
        areas = {name: float(masks[name].sum()) for name in names}
        for a_idx, a in enumerate(names):
            for b in names[a_idx + 1 :]:
                inter = float(np.logical_and(masks[a], masks[b]).sum())
                union = float(np.logical_or(masks[a], masks[b]).sum())
                row[f"{a}_{b}_pred_iou"] = (inter + 1e-6) / (union + 1e-6)
                row[f"{a}_{b}_area_ratio"] = (areas[a] + 1.0) / (areas[b] + 1.0)
        stack = np.stack([masks[name] for name in names], axis=0)
        vote = stack.sum(axis=0)
        row["vote_any_area"] = float((vote >= 1).sum())
        row["vote_major_area"] = float((vote >= max(2, len(names) // 2 + 1)).sum())
        row["vote_all_area"] = float((vote == len(names)).sum())
        row["model_area_mean"] = float(np.mean([areas[name] for name in names]))
        row["model_area_std"] = float(np.std([areas[name] for name in names]))
        row["model_area_min"] = float(np.min([areas[name] for name in names]))
        row["model_area_max"] = float(np.max([areas[name] for name in names]))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_split(
    specs: list[ExpertSpec],
    split: str,
    settings: dict[str, tuple[float, int]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    pred_cache: dict[str, np.ndarray] = {}
    id_cache: dict[str, list[str]] = {}
    common_ids: list[str] | None = None

    for spec in specs:
        ids, prob, gt = collect_probs(
            spec,
            split=split,
            batch_size=args.batch,
            num_workers=args.num_workers,
            tta_mode=args.tta_mode,
            device=device,
        )
        threshold, min_area = settings[spec.name]
        pred = apply_threshold(prob, threshold, min_area)
        pred_cache[spec.name] = pred
        id_cache[spec.name] = ids
        metrics = metric_arrays(pred, gt)

        metric_df = pd.DataFrame({"case_id": ids})
        for key, value in metrics.items():
            metric_df[f"{spec.name}_{key}"] = value
        metric_frames.append(metric_df)

        features = pd.DataFrame({"case_id": ids})
        features = pd.concat([features, mask_features(spec.name, prob, pred)], axis=1)
        features[f"{spec.name}_threshold"] = threshold
        features[f"{spec.name}_min_area"] = float(min_area)
        feature_frames.append(features)
        common_ids = ids if common_ids is None else common_ids

    metrics_merged = metric_frames[0]
    features_merged = feature_frames[0]
    for frame in metric_frames[1:]:
        metrics_merged = metrics_merged.merge(frame, on="case_id", how="inner")
    for frame in feature_frames[1:]:
        features_merged = features_merged.merge(frame, on="case_id", how="inner")

    # Pairwise agreement features require the same case ordering after the inner join.
    order = features_merged["case_id"].astype(str).tolist()
    aligned_preds = {}
    for spec in specs:
        index = {case_id: idx for idx, case_id in enumerate(id_cache[spec.name])}
        aligned_preds[spec.name] = np.stack([pred_cache[spec.name][index[case_id]] for case_id in order], axis=0)
    pair_df = pd.concat([pd.DataFrame({"case_id": order}), pair_features([s.name for s in specs], aligned_preds)], axis=1)
    features_merged = features_merged.merge(pair_df, on="case_id", how="inner")
    return metrics_merged, features_merged


def choose_dice(df: pd.DataFrame, model_col: str) -> np.ndarray:
    values = []
    for _, row in df.iterrows():
        values.append(float(row[f"{row[model_col]}_dice"]))
    return np.asarray(values, dtype=np.float32)


def add_gate_result(
    out: pd.DataFrame,
    name: str,
    chosen: np.ndarray,
) -> None:
    out[f"{name}_model"] = chosen
    out[f"{name}_dice"] = choose_dice(out, f"{name}_model")


def guarded_regression_choices(
    pred_calib: np.ndarray,
    pred_test: np.ndarray,
    calib_true_dice: np.ndarray,
    expert_names: list[str],
    default_name: str = "broad",
) -> tuple[np.ndarray, float, float]:
    names = np.asarray(expert_names, dtype=object)
    default_idx = int(expert_names.index(default_name)) if default_name in expert_names else 0
    best_margin = 0.0
    best_val_dice = -1.0

    for margin in np.linspace(-0.08, 0.20, 29):
        best_idx = pred_calib.argmax(axis=1)
        default_score = pred_calib[:, default_idx]
        best_score = pred_calib[np.arange(len(pred_calib)), best_idx]
        chosen_idx = np.where(best_score - default_score > margin, best_idx, default_idx)
        val_dice = float(calib_true_dice[np.arange(len(chosen_idx)), chosen_idx].mean())
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_margin = float(margin)

    best_idx_test = pred_test.argmax(axis=1)
    default_score_test = pred_test[:, default_idx]
    best_score_test = pred_test[np.arange(len(pred_test)), best_idx_test]
    chosen_idx_test = np.where(best_score_test - default_score_test > best_margin, best_idx_test, default_idx)
    return names[chosen_idx_test], best_margin, best_val_dice


def guarded_classification_choices(
    proba_calib: np.ndarray,
    pred_calib: np.ndarray,
    proba_test: np.ndarray,
    pred_test: np.ndarray,
    calib_true_dice: np.ndarray,
    expert_names: list[str],
    default_name: str = "broad",
) -> tuple[np.ndarray, float, float]:
    names = np.asarray(expert_names, dtype=object)
    default_idx = int(expert_names.index(default_name)) if default_name in expert_names else 0
    best_threshold = 0.0
    best_val_dice = -1.0
    conf_calib = proba_calib.max(axis=1)
    conf_test = proba_test.max(axis=1)

    for threshold in np.linspace(0.0, 0.95, 20):
        chosen_idx = np.where(conf_calib >= threshold, pred_calib, default_idx)
        val_dice = float(calib_true_dice[np.arange(len(chosen_idx)), chosen_idx].mean())
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_threshold = float(threshold)

    chosen_idx_test = np.where(conf_test >= best_threshold, pred_test, default_idx)
    return names[chosen_idx_test], best_threshold, best_val_dice


def encoded_predictions_to_expert_indices(
    encoded: np.ndarray,
    encoder: LabelEncoder,
    expert_names: list[str],
) -> np.ndarray:
    name_to_idx = {name: idx for idx, name in enumerate(expert_names)}
    decoded = encoder.inverse_transform(encoded)
    return np.asarray([name_to_idx[name] for name in decoded], dtype=np.int64)


def summarize(path: Path, df: pd.DataFrame, expert_names: list[str], gate_names: list[str], settings: dict[str, tuple[float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"n={len(df)}\n")
        f.write("\nExpert settings:\n")
        for name in expert_names:
            threshold, min_area = settings[name]
            f.write(f"{name}: threshold={threshold:.3f} min_area={min_area}\n")

        f.write("\nMetrics on v18 low-dice test subset:\n")
        cols = ["v18_dice"] + [f"{name}_dice" for name in expert_names] + [f"{name}_dice" for name in gate_names]
        cols += ["expert_oracle_dice", "oracle_with_v18_dice"]
        for col in cols:
            delta = df[col] - df["v18_dice"]
            f.write(
                f"{col.replace('_dice', '')}: mean={df[col].mean():.4f} "
                f"median={df[col].median():.4f} low_lt_060={(df[col] < 0.60).sum()} "
                f"zero={(df[col] <= 1e-9).sum()} delta={delta.mean():+.4f} "
                f"improved={(delta > 1e-9).sum()} worse={(delta < -1e-9).sum()}\n"
            )

        f.write("\nGate choices:\n")
        for name in gate_names:
            f.write(f"{name}:\n")
            for model, count in df[f"{name}_model"].value_counts().items():
                f.write(f"  {model}: {count}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", default="outputs/low_dice_v18_freq_ft8_lt060.csv")
    parser.add_argument("--expert", action="append", default=[], help="name:data_dir:model:ckpt")
    parser.add_argument("--tune_split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--gate_train_split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--gate_calib_split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--eval_split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--gate_data_dir",
        default="data/processed/qata_lowdice_specialist",
        help="Common manifest used to train/evaluate the gate for all experts.",
    )
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tta_mode", default="none", choices=("none", "h", "v", "hv", "all"))
    parser.add_argument("--thr_min", type=float, default=0.05)
    parser.add_argument("--thr_max", type=float, default=0.90)
    parser.add_argument("--thr_step", type=float, default=0.05)
    parser.add_argument("--min_areas", default="0,16,32,64,128,256")
    parser.add_argument("--default_model", default="broad")
    parser.add_argument("--out_csv", default="outputs/lowdice_learned_gate_eval.csv")
    parser.add_argument("--summary", default="outputs/lowdice_learned_gate_eval_summary.txt")
    args = parser.parse_args()

    specs = [parse_expert(item) for item in args.expert] if args.expert else default_experts()
    specs = [spec for spec in specs if spec.ckpt.exists() and (spec.data_dir / "manifest.csv").exists()]
    if not specs:
        raise FileNotFoundError("No expert checkpoints/manifests were found.")
    gate_data_dir = Path(args.gate_data_dir)
    if not (gate_data_dir / "manifest.csv").exists():
        raise FileNotFoundError(f"Missing gate manifest: {gate_data_dir / 'manifest.csv'}")
    specs = [ExpertSpec(spec.name, gate_data_dir, spec.model, spec.ckpt) for spec in specs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print("experts=" + ",".join(spec.name for spec in specs))

    settings: dict[str, tuple[float, int]] = {}
    thresholds = np.arange(args.thr_min, args.thr_max + args.thr_step / 2.0, args.thr_step)
    min_areas = parse_min_areas(args.min_areas)
    for spec in specs:
        ids, prob, gt = collect_probs(
            spec,
            split=args.tune_split,
            batch_size=args.batch,
            num_workers=args.num_workers,
            tta_mode=args.tta_mode,
            device=device,
        )
        del ids
        threshold, min_area, tune_dice = tune_threshold(prob, gt, thresholds, min_areas)
        settings[spec.name] = (threshold, min_area)
        print(f"{spec.name}: threshold={threshold:.3f} min_area={min_area} tune_dice={tune_dice:.4f}")

    train_metrics, train_features = evaluate_split(specs, args.gate_train_split, settings, args, device)
    if args.gate_calib_split == args.gate_train_split:
        calib_metrics, calib_features = train_metrics, train_features
    else:
        calib_metrics, calib_features = evaluate_split(specs, args.gate_calib_split, settings, args, device)
    test_metrics, test_features = evaluate_split(specs, args.eval_split, settings, args, device)

    expert_names = [spec.name for spec in specs]
    train = train_metrics.merge(train_features, on="case_id", how="inner")
    calib = calib_metrics.merge(calib_features, on="case_id", how="inner")
    test = test_metrics.merge(test_features, on="case_id", how="inner")

    dice_cols = [f"{name}_dice" for name in expert_names]
    train["best_expert"] = train[dice_cols].idxmax(axis=1).str.replace("_dice", "", regex=False)
    feature_cols = [c for c in train_features.columns if c != "case_id"]
    x_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_calib = calib[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    encoder = LabelEncoder()
    y_cls = encoder.fit_transform(train["best_expert"].to_numpy())
    y_reg = train[dice_cols].to_numpy(dtype=np.float32)
    y_reg_calib = calib[dice_cols].to_numpy(dtype=np.float32)

    gates: dict[str, np.ndarray] = {}
    clf_rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=2026,
        n_jobs=-1,
    )
    clf_rf.fit(x_train, y_cls)
    rf_pred_calib = clf_rf.predict(x_calib)
    rf_pred_test = clf_rf.predict(x_test)
    gates["rf_class_gate"] = encoder.inverse_transform(rf_pred_test)
    rf_class_guard, rf_conf, rf_class_guard_val = guarded_classification_choices(
        clf_rf.predict_proba(x_calib),
        encoded_predictions_to_expert_indices(rf_pred_calib, encoder, expert_names),
        clf_rf.predict_proba(x_test),
        encoded_predictions_to_expert_indices(rf_pred_test, encoder, expert_names),
        y_reg_calib,
        expert_names,
        default_name=args.default_model,
    )
    gates["rf_class_guard_gate"] = rf_class_guard

    clf_et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=2026,
        n_jobs=-1,
    )
    clf_et.fit(x_train, y_cls)
    et_pred_calib = clf_et.predict(x_calib)
    et_pred_test = clf_et.predict(x_test)
    gates["et_class_gate"] = encoder.inverse_transform(et_pred_test)
    et_class_guard, et_conf, et_class_guard_val = guarded_classification_choices(
        clf_et.predict_proba(x_calib),
        encoded_predictions_to_expert_indices(et_pred_calib, encoder, expert_names),
        clf_et.predict_proba(x_test),
        encoded_predictions_to_expert_indices(et_pred_test, encoder, expert_names),
        y_reg_calib,
        expert_names,
        default_name=args.default_model,
    )
    gates["et_class_guard_gate"] = et_class_guard

    reg_rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=5,
        random_state=2026,
        n_jobs=-1,
    )
    reg_rf.fit(x_train, y_reg)
    rf_val_pred = reg_rf.predict(x_calib)
    rf_pred = reg_rf.predict(x_test)
    gates["rf_reg_gate"] = np.asarray(expert_names, dtype=object)[rf_pred.argmax(axis=1)]
    rf_guard, rf_margin, rf_guard_val = guarded_regression_choices(
        rf_val_pred,
        rf_pred,
        y_reg_calib,
        expert_names,
        default_name=args.default_model,
    )
    gates["rf_guard_gate"] = rf_guard

    reg_et = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=4,
        random_state=2026,
        n_jobs=-1,
    )
    reg_et.fit(x_train, y_reg)
    et_val_pred = reg_et.predict(x_calib)
    et_pred = reg_et.predict(x_test)
    gates["et_reg_gate"] = np.asarray(expert_names, dtype=object)[et_pred.argmax(axis=1)]
    et_guard, et_margin, et_guard_val = guarded_regression_choices(
        et_val_pred,
        et_pred,
        y_reg_calib,
        expert_names,
        default_name=args.default_model,
    )
    gates["et_guard_gate"] = et_guard

    print(f"rf_class_guard conf={rf_conf:.3f} calib_dice={rf_class_guard_val:.4f}")
    print(f"et_class_guard conf={et_conf:.3f} calib_dice={et_class_guard_val:.4f}")
    print(f"rf_guard margin={rf_margin:.3f} val_dice={rf_guard_val:.4f}")
    print(f"et_guard margin={et_margin:.3f} val_dice={et_guard_val:.4f}")

    gate_frame = pd.DataFrame({"case_id": test["case_id"].astype(str)})
    for name, chosen in gates.items():
        gate_frame[f"{name}_model"] = chosen
    test = test.merge(gate_frame, on="case_id", how="inner")

    baseline = pd.read_csv(args.baseline_csv).rename(
        columns={
            "dice": "v18_dice",
            "iou": "v18_iou",
            "precision": "v18_precision",
            "recall": "v18_recall",
            "pred_area": "v18_pred_area",
            "gt_area": "v18_gt_area",
        }
    )
    out = baseline.merge(test, on="case_id", how="inner")
    for name in gates:
        out[f"{name}_dice"] = choose_dice(out, f"{name}_model")

    out["expert_oracle_model"] = out[dice_cols].idxmax(axis=1).str.replace("_dice", "", regex=False)
    out["expert_oracle_dice"] = out[dice_cols].max(axis=1)
    all_cols = ["v18_dice"] + dice_cols
    out["oracle_with_v18_model"] = out[all_cols].idxmax(axis=1).str.replace("_dice", "", regex=False)
    out["oracle_with_v18_dice"] = out[all_cols].max(axis=1)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    summary_path = Path(args.summary)
    summarize(summary_path, out, expert_names, list(gates.keys()), settings)
    print(f"train_gate_cases={len(train)}")
    print(f"calib_gate_cases={len(calib)}")
    print(f"test_cases={len(out)}")
    print(f"csv={out_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
