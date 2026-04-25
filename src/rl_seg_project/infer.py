from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.rl.fusion_agent import RLDynamicFusionSegmenter
from src.utils.common import device_auto


def load_gray(path: str, image_size):
    img = Image.open(path).convert("L").resize(image_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return ten


def load_branch_ckpt(module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    module.load_state_dict(ckpt.get("model", ckpt), strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--trans_ckpt", type=str, required=True)
    parser.add_argument("--swin_ckpt", type=str, required=True)
    parser.add_argument("--uct_ckpt", type=str, required=True)
    parser.add_argument("--rl_ckpt", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="prediction.png")
    args = parser.parse_args()

    device = device_auto()
    model = RLDynamicFusionSegmenter(image_size=tuple(args.image_size), freeze_backbones=True).to(device)
    load_branch_ckpt(model.transunet, args.trans_ckpt, device)
    load_branch_ckpt(model.swin, args.swin_ckpt, device)
    load_branch_ckpt(model.uctransnet, args.uct_ckpt, device)
    rl = torch.load(args.rl_ckpt, map_location=device)
    model.policy.load_state_dict(rl["policy"])
    model.adapters.load_state_dict(rl["adapters"])
    model.eval()

    image = load_gray(args.image, tuple(args.image_size)).to(device)
    with torch.no_grad():
        out = model.deterministic_forward(image)
        prob = torch.sigmoid(out["fused_logits"])[0, 0].cpu().numpy()
        mask = (prob >= 0.5).astype(np.uint8) * 255
    Image.fromarray(mask).save(args.save_path)
    print("weights:", out["weights"][0].tolist())
    print("saved:", args.save_path)


if __name__ == "__main__":
    main()
