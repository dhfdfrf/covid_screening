from __future__ import annotations

from typing import Dict

import torch

from src.utils.losses import dice_score_from_probs, iou_score_from_probs


@torch.no_grad()
def segmentation_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    dice = dice_score_from_probs(pred, target).mean().item()
    iou = iou_score_from_probs(pred, target).mean().item()
    return {"dice": dice, "iou": iou}
