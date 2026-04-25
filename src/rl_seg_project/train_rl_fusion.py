from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, random_split

from src.data.dataset import QaTaCOV19Dataset
from src.rl.fusion_agent import PPOBatch, RLDynamicFusionSegmenter, compute_reward, ppo_update
from src.utils.common import device_auto, ensure_dir, save_checkpoint, set_seed
from src.utils.losses import dice_score_from_probs, iou_score_from_probs


def load_branch_ckpt(module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = module.load_state_dict(state, strict=False)
    print(f"loaded {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dice_sum = 0.0
    iou_sum = 0.0
    n = 0
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        out = model.deterministic_forward(image)
        probs = torch.sigmoid(out["fused_logits"])
        pred = (probs >= 0.5).float()
        dice = dice_score_from_probs(pred, mask).mean().item()
        iou = iou_score_from_probs(pred, mask).mean().item()
        dice_sum += dice
        iou_sum += iou
        n += 1
    return {"dice": dice_sum / max(n, 1), "iou": iou_sum / max(n, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints/rl")
    parser.add_argument("--trans_ckpt", type=str, required=True)
    parser.add_argument("--swin_ckpt", type=str, required=True)
    parser.add_argument("--uct_ckpt", type=str, required=True)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = device_auto()
    dataset = QaTaCOV19Dataset(args.image_dir, args.mask_dir, tuple(args.image_size))
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = RLDynamicFusionSegmenter(image_size=tuple(args.image_size), freeze_backbones=True).to(device)
    load_branch_ckpt(model.transunet, args.trans_ckpt, device)
    load_branch_ckpt(model.swin, args.swin_ckpt, device)
    load_branch_ckpt(model.uctransnet, args.uct_ckpt, device)
    optimizer = torch.optim.AdamW(list(model.policy.parameters()) + list(model.adapters.parameters()), lr=args.lr, weight_decay=1e-4)

    best_dice = -1.0
    ensure_dir(args.save_dir)

    for epoch in range(1, args.epochs + 1):
        model.train()
        reward_sum = 0.0
        dice_sum = 0.0
        for batch in train_loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            with torch.no_grad():
                with autocast(enabled=device.type == "cuda"):
                    out = model(image)
            reward, reward_info = compute_reward(out["fused_logits"], out["branch_logits"], mask)
            returns = reward.detach()
            advantages = returns - out["value"].detach()
            ppo_batch = PPOBatch(
                states=out["state"].detach(),
                raw_actions=out["raw_actions"].detach(),
                old_log_probs=out["log_prob"].detach(),
                returns=returns.detach(),
                advantages=advantages.detach(),
            )
            stats = ppo_update(model, optimizer, ppo_batch, epochs=args.ppo_epochs)
            reward_sum += float(reward.mean().item())
            dice_sum += reward_info["dice"]

        val_metrics = evaluate(model, val_loader, device)
        print(
            f"[RL] epoch={epoch} train_reward={reward_sum/max(len(train_loader),1):.4f} "
            f"train_dice={dice_sum/max(len(train_loader),1):.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} {stats}"
        )
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            save_path = str(Path(args.save_dir) / "rl_fusion_best.pt")
            save_checkpoint({
                "policy": model.policy.state_dict(),
                "adapters": model.adapters.state_dict(),
                "metrics": val_metrics,
                "args": vars(args),
            }, save_path)
            print(f"saved to {save_path}")


if __name__ == "__main__":
    main()
