import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.seed import seed_everything
from src.utils.metrics import dice_coeff_from_logits
from src.datasets.qata_npz import QaTaNPZDataset
from src.models.model_factory import ModelBuildConfig, build_model, available_models


def _parse_hw(s: str):
    s = (s or "").strip()
    if not s:
        return None
    if "," in s:
        a, b = s.split(",", 1)
        return (int(a), int(b))
    v = int(s)
    return (v, v)


def dice_focal_loss(logits, target, eps=1e-6, alpha=0.25, gamma=2.0):
    probs = torch.sigmoid(logits)

    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    pt = probs * target + (1 - probs) * (1 - target)
    focal = (alpha * (1 - pt) ** gamma * bce).mean()

    inter = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    dice_loss = 1 - dice.mean()

    return dice_loss + focal


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0,
    alpha: float = 0.6,
    beta: float = 0.4,
    gamma: float = 0.75,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    p = probs.flatten(1)
    t = target.flatten(1)
    tp = (p * t).sum(dim=1)
    fp = (p * (1.0 - t)).sum(dim=1)
    fn = ((1.0 - p) * t).sum(dim=1)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1.0 - tversky).pow(gamma).mean()


def tversky_bce_loss(logits: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0.01):
    target = target * (1.0 - label_smoothing) + (1.0 - target) * label_smoothing
    return 0.6 * focal_tversky_loss(logits, target) + 0.4 * F.binary_cross_entropy_with_logits(logits, target)


def boundary_target_from_mask(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp_(0.0, 1.0)


def deep_boundary_loss(outputs, target, boundary_weight: float = 0.1):
    if not isinstance(outputs, dict):
        return dice_focal_loss(unwrap_logits(outputs), target)

    loss = dice_focal_loss(outputs["seg"], target)
    for key, weight in (("ds2", 0.2), ("ds3", 0.1), ("ds4", 0.05)):
        if key in outputs:
            loss = loss + weight * dice_focal_loss(outputs[key], target)

    if boundary_weight > 0 and "boundary" in outputs:
        boundary = boundary_target_from_mask(target)
        loss = loss + boundary_weight * dice_focal_loss(outputs["boundary"], boundary)

    return loss


def compute_loss(outputs, target, loss_mode: str, boundary_weight: float):
    if loss_mode == "tversky_bce":
        return tversky_bce_loss(unwrap_logits(outputs), target)
    if loss_mode == "deep_boundary":
        return deep_boundary_loss(outputs, target, boundary_weight=boundary_weight)
    return dice_focal_loss(unwrap_logits(outputs), target)


def unwrap_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["seg"]
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def evaluate(model, loader, device, loss_mode: str, boundary_weight: float):
    model.eval()
    dices = []
    losses = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            outputs = model(x)
            logits = unwrap_logits(outputs)

            loss = compute_loss(outputs, y, loss_mode, boundary_weight)
            dices.append(dice_coeff_from_logits(logits, y))
            losses.append(float(loss.item()))

    return float(sum(dices) / len(dices)), float(sum(losses) / len(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument(
        "--loss",
        type=str,
        default="dice_focal",
        choices=("dice_focal", "tversky_bce", "deep_boundary"),
        help="training loss; deep_boundary enables ds2/ds3/ds4 and boundary head when present",
    )
    ap.add_argument("--boundary_weight", type=float, default=0.1)
    ap.add_argument("--resume", type=str, default="", help="optional checkpoint to load model weights from")
    ap.add_argument("--run_tag", type=str, default="", help="optional output tag for checkpoint/tensorboard")

    ap.add_argument(
        "--model",
        type=str,
        default="unet2d",
        help=f"model name, choices: {', '.join(list(available_models()) + ['transunet2d_v3', 'transunet2d_v4', 'transunet2d_v5', 'transunet2d_v6', 'transunet2d_v7', 'transunet2d_v8', 'transunet2d_v9', 'transunet2d_v10', 'transunet2d_v11', 'transunet2d_v12'])}",
    )
    ap.add_argument(
        "--img_size",
        type=str,
        default="",
        help="optional, e.g. '512,512' (only some models need it)",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default="covid-19 infection region",
        help="used by lavt2d (if you run it)",
    )

    args = ap.parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    manifest = data_dir / "manifest.csv"
    assert manifest.exists(), f"Missing manifest: {manifest}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu =", torch.cuda.get_device_name(0))

    ds_train = QaTaNPZDataset(str(manifest), split="train")
    ds_val = QaTaNPZDataset(str(manifest), split="val")

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    img_size = _parse_hw(args.img_size)
    model_name = args.model.strip().lower()

    if model_name in ("transunet2d_v12", "transunet_v12", "transunet2dv12"):
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
    elif model_name in ("transunet2d_v2", "transunet_v2", "transunet2dv2"):
        from src.models.transunet2d_v2 import build_transunet2d_v2
        model = build_transunet2d_v2(
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

    if args.resume.strip():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"resumed model weights from {args.resume}")

    print(f"LOSS = {args.loss}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6
    )

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=(device.type == "cuda"))

    run_dir = Path("outputs")
    run_tag = args.run_tag.strip() or f"{args.model}_qata_exp"
    ckpt_path = run_dir / f"{run_tag}_best.pt"
    tb_dir = run_dir / "tensorboard" / run_tag
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))

    best_val_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for _, batch in enumerate(pbar):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(amp_device, enabled=(device.type == "cuda")):
                outputs = model(x)
                logits = unwrap_logits(outputs)
                loss = compute_loss(outputs, y, args.loss, args.boundary_weight)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(loss=float(loss.item()))

        sched.step()

        val_dice, val_loss = evaluate(model, dl_val, device, args.loss, args.boundary_weight)
        writer.add_scalar("val/dice", val_dice, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)

        print(f"[Epoch {epoch}] val_dice={val_dice:.4f} val_loss={val_loss:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                },
                ckpt_path,
            )
            print(f"Saved best -> {ckpt_path}")

    writer.close()
    print(f"Done. Best val dice = {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
