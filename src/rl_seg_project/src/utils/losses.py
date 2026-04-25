from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_score_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = probs.float()
    target = target.float()
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    return (2.0 * inter + eps) / (denom + eps)


def iou_score_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = probs.float()
    target = target.float()
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims) - inter
    return (inter + eps) / (union + eps)


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        dice = 1.0 - dice_score_from_probs(probs, target).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice
