from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

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
FONT_SMALL = _font(13, False)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    return (np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    return (np.array(img.resize(size, Image.Resampling.NEAREST)) > 127).astype(np.uint8)


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    p = np.pad(m, 1, mode="constant")
    eroded = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return np.logical_xor(m, eroded)


def _draw_boundary(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.copy()
    b = _boundary(mask)
    yy, xx = np.where(b)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y = np.clip(yy + dy, 0, out.shape[0] - 1)
            x = np.clip(xx + dx, 0, out.shape[1] - 1)
            out[y, x] = color
    return out


def _overlay(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    pred_bool = pred.astype(bool)
    base[pred_bool] = base[pred_bool] * 0.5 + np.array([230, 45, 42]) * 0.5
    out = base.astype(np.uint8)
    out = _draw_boundary(out, gt, (255, 220, 30))
    out = _draw_boundary(out, pred, (0, 220, 255))
    return Image.fromarray(out)


def _mask_image(mask: np.ndarray) -> Image.Image:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask.astype(bool)] = 255
    return Image.fromarray(out)


def _fit(img: Image.Image, size: int, black: bool = False) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.Resampling.NEAREST if black else Image.Resampling.LANCZOS)
    bg = Image.new("RGB", (size, size), (0, 0, 0) if black else (245, 245, 245))
    bg.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return bg


def _metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    tp = int(np.logical_and(gt_b, pred_b).sum())
    fp = int(np.logical_and(~gt_b, pred_b).sum())
    fn = int(np.logical_and(gt_b, ~pred_b).sum())
    gt_area = int(gt_b.sum())
    pred_area = int(pred_b.sum())
    eps = 1e-6
    return {
        "dice": float((2 * tp + eps) / (gt_area + pred_area + eps)),
        "iou": float((tp + eps) / (tp + fp + fn + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_area": gt_area,
        "pred_area": pred_area,
    }


def _failure_type(m: dict[str, float | int]) -> str:
    dice = float(m["dice"])
    precision = float(m["precision"])
    recall = float(m["recall"])
    pred_area = int(m["pred_area"])
    gt_area = int(m["gt_area"])
    tp = int(m["tp"])

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


def collect_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)
    manifest = data_dir / "manifest.csv"
    df = pd.read_csv(manifest)
    df = df[df["split"].eq(args.split)].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for idx, row in df.iterrows():
        pred_path = pred_dir / f"{idx:06d}_pred.png"
        if not pred_path.exists():
            continue
        npz_path = ROOT / str(row["npz_path"])
        npz = np.load(npz_path)
        gt = (npz["mask"][0] > 0.5).astype(np.uint8)
        pred = (np.array(Image.open(pred_path).convert("L")) > 127).astype(np.uint8)
        if pred.shape != gt.shape:
            pred = _resize_mask(pred, (gt.shape[1], gt.shape[0]))

        m = _metrics(gt, pred)
        rows.append(
            {
                "index": idx,
                "case_id": npz_path.stem,
                "npz_path": str(row["npz_path"]),
                **m,
                "failure_type": _failure_type(m),
            }
        )

    rows.sort(key=lambda item: float(item["dice"]))
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "index",
        "case_id",
        "dice",
        "iou",
        "precision",
        "recall",
        "gt_area",
        "pred_area",
        "tp",
        "fp",
        "fn",
        "failure_type",
        "npz_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            formatted = {k: row[k] for k in keys}
            for key in ("dice", "iou", "precision", "recall"):
                formatted[key] = f"{float(row[key]):.6f}"
            writer.writerow(formatted)


def make_sheet(rows: list[dict[str, object]], args: argparse.Namespace, path: Path) -> None:
    data_dir = Path(args.data_dir)
    manifest = pd.read_csv(data_dir / "manifest.csv")
    manifest = manifest[manifest["split"].eq(args.split)].reset_index(drop=True)
    pred_dir = Path(args.pred_dir)

    selected = rows[: args.max_sheet]
    if not selected:
        return

    panel = 138
    row_h = panel + 58
    gap = 12
    cols = 4
    canvas_w = cols * (panel * 4 + gap * 5) + gap
    canvas_h = 58 + int(np.ceil(len(selected) / cols)) * row_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 16), f"Lowest Dice cases: {Path(args.pred_dir).name}", font=FONT_TITLE, fill=(20, 24, 32))

    for n, row in enumerate(selected):
        block_x = gap + (n % cols) * (panel * 4 + gap * 5)
        block_y = 58 + (n // cols) * row_h
        idx = int(row["index"])
        npz = np.load(ROOT / str(manifest.loc[idx, "npz_path"]))
        gray = _to_uint8(npz["img"])
        gt = (npz["mask"][0] > 0.5).astype(np.uint8)
        pred = (np.array(Image.open(pred_dir / f"{idx:06d}_pred.png").convert("L")) > 127).astype(np.uint8)
        if pred.shape != gt.shape:
            pred = _resize_mask(pred, (gt.shape[1], gt.shape[0]))
        imgs = [
            Image.fromarray(gray),
            _mask_image(gt),
            _mask_image(pred),
            _overlay(gray, gt, pred),
        ]
        labels = ["Orig", "GT", "Pred", "Overlay"]
        for j, (img, label) in enumerate(zip(imgs, labels)):
            x = block_x + gap + j * (panel + gap)
            y = block_y
            draw.text((x + 2, y), label, font=FONT_SMALL, fill=(35, 35, 35))
            canvas.paste(_fit(img, panel, black=(label in ("GT", "Pred"))), (x, y + 18))
        info = f"idx={idx} Dice={float(row['dice']):.3f} {row['failure_type']}"
        draw.text((block_x + gap, block_y + panel + 22), info, font=FONT_TEXT, fill=(30, 30, 30))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/qata")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--low_thr", type=float, default=0.60)
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--sheet", default="")
    parser.add_argument("--max_sheet", type=int, default=24)
    args = parser.parse_args()

    rows = collect_cases(args)
    low_rows = [row for row in rows if float(row["dice"]) < args.low_thr]

    out_csv = Path(args.out_csv) if args.out_csv else Path(args.pred_dir) / f"low_dice_lt_{args.low_thr:.2f}.csv"
    write_csv(low_rows, out_csv)
    if args.sheet:
        make_sheet(low_rows, args, Path(args.sheet))

    print(f"samples={len(rows)}")
    print(f"low_dice_lt_{args.low_thr:.2f}={len(low_rows)}")
    print(f"csv={out_csv}")
    if args.sheet:
        print(f"sheet={args.sheet}")

    bins = [
        ("dice=0", lambda x: float(x["dice"]) <= 1e-6),
        ("0<dice<0.30", lambda x: 1e-6 < float(x["dice"]) < 0.30),
        ("0.30<=dice<0.60", lambda x: 0.30 <= float(x["dice"]) < 0.60),
        ("0.60<=dice<0.75", lambda x: 0.60 <= float(x["dice"]) < 0.75),
        ("dice>=0.75", lambda x: float(x["dice"]) >= 0.75),
    ]
    for name, pred in bins:
        print(f"{name}: {sum(1 for row in rows if pred(row))}")

    by_type: dict[str, int] = {}
    for row in low_rows:
        by_type[str(row["failure_type"])] = by_type.get(str(row["failure_type"]), 0) + 1
    for name, count in sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{name}: {count}")

    print("lowest:")
    for row in rows[: min(15, len(rows))]:
        print(
            f"idx={row['index']:>3} dice={float(row['dice']):.3f} "
            f"p={float(row['precision']):.3f} r={float(row['recall']):.3f} "
            f"gt={row['gt_area']} pred={row['pred_area']} {row['failure_type']}"
        )


if __name__ == "__main__":
    main()
