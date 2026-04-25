from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from src.data.dataset import QaTaCOV19Dataset
from src.models.model_factory import ModelBuildConfig, build_model
from src.utils.common import device_auto, ensure_dir, save_checkpoint, set_seed
from src.utils.losses import DiceBCELoss
from src.utils.metrics import segmentation_metrics_from_logits


def evaluate(model, loader, device):
    model.eval()
    dice_sum = 0.0
    iou_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            logits = model(image)
            if isinstance(logits, tuple):
                logits = logits[0]
            metric = segmentation_metrics_from_logits(logits, mask)
            dice_sum += metric["dice"]
            iou_sum += metric["iou"]
            n += 1
    return {"dice": dice_sum / max(n, 1), "iou": iou_sum / max(n, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["transunet2d", "swin_unet2d", "uctransnet2d"])
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints/base")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = device_auto()
    dataset = QaTaCOV19Dataset(args.image_dir, args.mask_dir, tuple(args.image_size))
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(ModelBuildConfig(name=args.model, image_size=tuple(args.image_size), return_features=False)).to(device)
    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_dice = -1.0
    ensure_dir(args.save_dir)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(image)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())

        metrics = evaluate(model, val_loader, device)
        print(f"[{args.model}] epoch={epoch} loss={loss_sum/max(len(train_loader),1):.4f} dice={metrics['dice']:.4f} iou={metrics['iou']:.4f}")
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            save_path = str(Path(args.save_dir) / f"{args.model}_best.pt")
            save_checkpoint({"model": model.state_dict(), "metrics": metrics, "args": vars(args)}, save_path)
            print(f"saved to {save_path}")


if __name__ == "__main__":
    main()
