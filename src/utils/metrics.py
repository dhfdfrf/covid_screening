import numpy as np
import torch


@torch.no_grad()
def dice_coeff_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """
    logits: (B,1,H,W) raw logits
    target: (B,1,H,W) {0,1}
    """
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()

    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return float(dice.mean().item())


def dice_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    return float((2 * inter + eps) / (union + eps))


def iou_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float((inter + eps) / (union + eps))


def precision_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    tp = (pred & gt).sum()
    fp = (pred & (1 - gt)).sum()
    return float((tp + eps) / (tp + fp + eps))


def recall_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    tp = (pred & gt).sum()
    fn = ((1 - pred) & gt).sum()
    return float((tp + eps) / (tp + fn + eps))