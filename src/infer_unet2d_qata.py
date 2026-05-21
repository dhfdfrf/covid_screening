import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.qata_npz import QaTaNPZDataset
from src.models.model_factory import ModelBuildConfig, build_model, available_models
from src.utils.metrics import dice_np, iou_np, precision_np, recall_np
from src.utils.viz import overlay_and_save


def _parse_hw(s: str):
    s = (s or "").strip()
    if not s:
        return None
    if "," in s:
        a, b = s.split(",", 1)
        return (int(a), int(b))
    v = int(s)
    return (v, v)


def unwrap_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["seg"]
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    out = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            out[labels == label] = 1
    return out


@torch.no_grad()
def predict_logits(model, x, tta_mode: str):
    seg = unwrap_logits(model(x))
    if tta_mode == "none":
        return seg

    logits = [seg]
    if tta_mode in ("h", "all"):
        logits.append(torch.flip(unwrap_logits(model(torch.flip(x, [3]))), [3]))
    if tta_mode in ("v", "all"):
        logits.append(torch.flip(unwrap_logits(model(torch.flip(x, [2]))), [2]))
    if tta_mode in ("hv", "all"):
        logits.append(torch.flip(unwrap_logits(model(torch.flip(x, [2, 3]))), [2, 3]))
    return torch.stack(logits, dim=0).mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min_area", type=int, default=0)
    ap.add_argument(
        "--tta_mode",
        type=str,
        default="none",
        choices=("none", "h", "v", "hv", "all"),
        help="test-time augmentation mode; all is usually not recommended for this project",
    )

    ap.add_argument(
        "--model",
        type=str,
        default="unet2d",
        help=f"choices: {', '.join(list(available_models()) + ['transunet2d_v3', 'transunet2d_v4', 'transunet2d_v5', 'transunet2d_v6', 'transunet2d_v7', 'transunet2d_v8', 'transunet2d_v9', 'transunet2d_v10', 'transunet2d_v11', 'transunet2d_v12', 'transunet2d_v13', 'transunet2d_v14', 'transunet2d_v16', 'transunet2d_v17', 'transunet2d_v18', 'transunet2d_v19', 'transunet2d_v20'])}",
    )
    ap.add_argument(
        "--img_size",
        type=str,
        default="",
        help="optional, e.g. '512,512'",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default="covid-19 infection region",
        help="used by lavt2d",
    )

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    manifest = data_dir / "manifest.csv"
    assert manifest.exists(), f"Missing manifest: {manifest}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_test = QaTaNPZDataset(str(manifest), split="test")
    dl = DataLoader(
        ds_test,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    img_size = _parse_hw(args.img_size)
    model_name = args.model.strip().lower()

    if model_name in ("transunet2d_v20", "transunet_v20", "transunet2dv20"):
        from src.models.transunet2d_v20 import build_transunet2d_v20
        model = build_transunet2d_v20(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v19", "transunet_v19", "transunet2dv19"):
        from src.models.transunet2d_v19 import build_transunet2d_v19
        model = build_transunet2d_v19(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v18", "transunet_v18", "transunet2dv18"):
        from src.models.transunet2d_v18 import build_transunet2d_v18
        model = build_transunet2d_v18(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v17", "transunet_v17", "transunet2dv17"):
        from src.models.transunet2d_v17 import build_transunet2d_v17
        model = build_transunet2d_v17(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v16", "transunet_v16", "transunet2dv16"):
        from src.models.transunet2d_v16 import build_transunet2d_v16
        model = build_transunet2d_v16(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v14", "transunet_v14", "transunet2dv14"):
        from src.models.transunet2d_v14 import build_transunet2d_v14
        model = build_transunet2d_v14(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v13", "transunet_v13", "transunet2dv13"):
        from src.models.transunet2d_v13 import build_transunet2d_v13
        model = build_transunet2d_v13(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v12", "transunet_v12", "transunet2dv12"):
        from src.models.transunet2d_v12 import build_transunet2d_v12
        model = build_transunet2d_v12(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v11", "transunet_v11", "transunet2dv11"):
        from src.models.transunet2d_v11 import build_transunet2d_v11
        model = build_transunet2d_v11(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v10", "transunet_v10", "transunet2dv10"):
        from src.models.transunet2d_v10 import build_transunet2d_v10
        model = build_transunet2d_v10(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v9", "transunet_v9", "transunet2dv9"):
        from src.models.transunet2d_v9 import build_transunet2d_v9
        model = build_transunet2d_v9(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v8", "transunet_v8", "transunet2dv8"):
        from src.models.transunet2d_v8 import build_transunet2d_v8
        model = build_transunet2d_v8(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v7", "transunet_v7", "transunet2dv7"):
        from src.models.transunet2d_v7 import build_transunet2d_v7
        model = build_transunet2d_v7(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v6", "transunet_v6", "transunet2dv6"):
        from src.models.transunet2d_v6 import build_transunet2d_v6
        model = build_transunet2d_v6(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v5", "transunet_v5", "transunet2dv5"):
        from src.models.transunet2d_v5 import build_transunet2d_v5
        model = build_transunet2d_v5(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v4", "transunet_v4", "transunet2dv4"):
        from src.models.transunet2d_v4 import build_transunet2d_v4
        model = build_transunet2d_v4(
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif model_name in ("transunet2d_v3", "transunet_v3", "transunet2dv3"):
        from src.models.transunet2d_v3 import build_transunet2d_v3
        model = build_transunet2d_v3(
            in_channels=1,
            out_channels=1,
        ).to(device)
    else:
        model = build_model(
            ModelBuildConfig(
                name=args.model,
                in_channels=1,
                out_channels=1,
                image_size=img_size,
                prompt=args.prompt,
            )
        ).to(device)

    if args.ckpt is None or str(args.ckpt).strip() == "":
        args.ckpt = f"outputs/{args.model}_qata_best.pt"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    dices = []
    ious = []
    precisions = []
    recalls = []

    idx_global = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="Infer"):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].cpu().numpy()

            logits = predict_logits(model, x, args.tta_mode)

            prob = torch.sigmoid(logits).cpu().numpy()
            pred = (prob > args.thr).astype(np.uint8)

            for i in range(pred.shape[0]):
                img = batch["image"][i, 0].cpu().numpy()
                gt = y[i, 0].astype(np.uint8)
                pm = pred[i, 0].astype(np.uint8)
                pm = remove_small_components(pm, args.min_area)

                dices.append(dice_np(pm, gt))
                ious.append(iou_np(pm, gt))
                precisions.append(precision_np(pm, gt))
                recalls.append(recall_np(pm, gt))

                mask_path = out_dir / f"{idx_global:06d}_pred.png"
                cv2.imwrite(str(mask_path), (pm * 255).astype(np.uint8))

                overlay_path = out_dir / f"{idx_global:06d}_overlay.png"
                overlay_and_save(img, pm, overlay_path)

                idx_global += 1

    print(f"Model={args.model}")
    print(f"Checkpoint={args.ckpt}")
    print(f"Test mean Dice={float(np.mean(dices)):.4f} (n={len(dices)})")
    print(f"Test mean IoU={float(np.mean(ious)):.4f}")
    print(f"Test mean Precision={float(np.mean(precisions)):.4f}")
    print(f"Test mean Recall={float(np.mean(recalls)):.4f}")


if __name__ == "__main__":
    main()
