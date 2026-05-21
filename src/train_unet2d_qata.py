import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def balanced_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = probs * target + (1.0 - probs) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def combo_seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        0.55 * soft_dice_loss(logits, target)
        + 0.25 * balanced_focal_loss(logits, target)
        + 0.20 * focal_tversky_loss(logits, target, alpha=0.7, beta=0.3)
    )


def _target_sample_weights(
    target: torch.Tensor,
    small_thr: float = 0.015,
    medium_thr: float = 0.08,
) -> torch.Tensor:
    area_ratio = target.flatten(1).mean(dim=1)
    weights = torch.ones_like(area_ratio)
    weights = torch.where(area_ratio < medium_thr, torch.full_like(weights, 1.45), weights)
    weights = torch.where(area_ratio < small_thr, torch.full_like(weights, 2.25), weights)
    return weights


def _weighted_mean_per_sample(loss_per_sample: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.detach()
    return (loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def sample_weighted_soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    loss = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)
    return _weighted_mean_per_sample(loss, _target_sample_weights(target))


def sample_weighted_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.35,
    beta: float = 0.65,
    gamma: float = 0.75,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    p = probs.flatten(1)
    t = target.flatten(1)
    tp = (p * t).sum(dim=1)
    fp = (p * (1.0 - t)).sum(dim=1)
    fn = ((1.0 - p) * t).sum(dim=1)
    score = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    loss = (1.0 - score).pow(gamma)
    return _weighted_mean_per_sample(loss, _target_sample_weights(target))


def boundary_weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    boundary_band = boundary_target_from_mask(target, kernel_size=5)
    pixel_weight = 1.0 + 2.5 * target + 2.0 * boundary_band
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = (bce * pixel_weight).flatten(1).sum(dim=1) / pixel_weight.flatten(1).sum(dim=1).clamp_min(1e-6)
    return _weighted_mean_per_sample(loss, _target_sample_weights(target))


def soft_centroid_alignment_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    b, _, h, w = probs.shape
    yy = torch.linspace(0.0, 1.0, h, device=probs.device, dtype=probs.dtype).view(1, 1, h, 1)
    xx = torch.linspace(0.0, 1.0, w, device=probs.device, dtype=probs.dtype).view(1, 1, 1, w)

    p_sum = probs.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    t_sum = target.sum(dim=(1, 2, 3)).clamp_min(1e-6)

    px = (probs * xx).sum(dim=(1, 2, 3)) / p_sum
    py = (probs * yy).sum(dim=(1, 2, 3)) / p_sum
    tx = (target * xx).sum(dim=(1, 2, 3)) / t_sum
    ty = (target * yy).sum(dim=(1, 2, 3)) / t_sum

    valid = (target.sum(dim=(1, 2, 3)) > 0).float()
    loss = ((px - tx).square() + (py - ty).square()) * valid
    if valid.sum() < 1:
        return logits.new_tensor(0.0)
    return _weighted_mean_per_sample(loss, _target_sample_weights(target) * valid)


def hard_region_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        0.40 * sample_weighted_soft_dice_loss(logits, target)
        + 0.30 * sample_weighted_tversky_loss(logits, target)
        + 0.22 * boundary_weighted_bce_loss(logits, target)
        + 0.08 * soft_centroid_alignment_loss(logits, target)
    )


def relative_area_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mode: str = "symmetric",
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred_sum = probs.sum(dim=(1, 2, 3))
    target_sum = target.sum(dim=(1, 2, 3)).clamp_min(1.0)
    diff = (pred_sum - target_sum) / target_sum
    if mode == "over":
        loss = diff.clamp_min(0.0)
    elif mode == "under":
        loss = (-diff).clamp_min(0.0)
    else:
        loss = diff.abs()
    return _weighted_mean_per_sample(loss, _target_sample_weights(target))


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


def boundary_combo_loss(outputs, target, boundary_weight: float = 0.05):
    if not isinstance(outputs, dict):
        return combo_seg_loss(unwrap_logits(outputs), target)

    loss = combo_seg_loss(outputs["seg"], target)
    for key, weight in (("ds2", 0.15), ("ds3", 0.08), ("ds4", 0.04)):
        if key in outputs:
            loss = loss + weight * combo_seg_loss(outputs[key], target)

    if boundary_weight > 0 and "boundary" in outputs:
        boundary = boundary_target_from_mask(target)
        loss = loss + boundary_weight * combo_seg_loss(outputs["boundary"], boundary)

    return loss


def prior_boundary_combo_loss(outputs, target, boundary_weight: float = 0.05):
    loss = boundary_combo_loss(outputs, target, boundary_weight=boundary_weight)
    if isinstance(outputs, dict) and "prior" in outputs:
        loss = loss + 0.15 * combo_seg_loss(outputs["prior"], target)
    return loss


def prior_calibration_combo_loss(outputs, target, boundary_weight: float = 0.05):
    loss = boundary_combo_loss(outputs, target, boundary_weight=boundary_weight)
    if isinstance(outputs, dict) and "prior" in outputs:
        loss = loss + 0.05 * combo_seg_loss(outputs["prior"], target)
    if isinstance(outputs, dict) and "calibration" in outputs:
        loss = loss + 0.01 * outputs["calibration"].abs().mean()
    return loss


def frequency_prior_combo_loss(outputs, target, boundary_weight: float = 0.05):
    loss = boundary_combo_loss(outputs, target, boundary_weight=boundary_weight)
    if isinstance(outputs, dict) and "prior" in outputs:
        loss = loss + 0.03 * combo_seg_loss(outputs["prior"], target)
    return loss


def posterior_calibration_loss(outputs, target, boundary_weight: float = 0.05):
    loss = boundary_combo_loss(outputs, target, boundary_weight=boundary_weight)
    if isinstance(outputs, dict) and "calibration" in outputs:
        loss = loss + 0.02 * outputs["calibration"].abs().mean()
    return loss


def frequency_aux_loss(outputs, target, boundary_weight: float = 0.05):
    loss = boundary_combo_loss(outputs, target, boundary_weight=boundary_weight)
    if isinstance(outputs, dict) and "freq_aux" in outputs:
        loss = loss + 0.05 * combo_seg_loss(outputs["freq_aux"], target)
    return loss


def hard_case_combo_loss(outputs, target, boundary_weight: float = 0.08):
    if not isinstance(outputs, dict):
        return hard_region_loss(unwrap_logits(outputs), target)

    loss = hard_region_loss(outputs["seg"], target)
    for key, weight in (("ds2", 0.16), ("ds3", 0.08), ("ds4", 0.04)):
        if key in outputs:
            loss = loss + weight * hard_region_loss(outputs[key], target)

    if boundary_weight > 0 and "boundary" in outputs:
        boundary = boundary_target_from_mask(target, kernel_size=5)
        loss = loss + boundary_weight * hard_region_loss(outputs["boundary"], boundary)
    if "prior" in outputs:
        loss = loss + 0.04 * hard_region_loss(outputs["prior"], target)
    if "freq_aux" in outputs:
        loss = loss + 0.06 * hard_region_loss(outputs["freq_aux"], target)
    return loss


def mild_hard_case_combo_loss(outputs, target, boundary_weight: float = 0.05):
    """Conservative hard-case regularization for low-Dice samples.

    The v21 hard loss improves the training objective but can dominate the
    segmentation loss. This variant keeps the stable boundary-combo objective
    as the anchor and adds only lightweight penalties for false negatives,
    boundary pixels and large centroid shifts.
    """
    logits = unwrap_logits(outputs)
    loss = boundary_combo_loss(outputs, target, boundary_weight=boundary_weight)
    loss = loss + 0.08 * sample_weighted_tversky_loss(
        logits,
        target,
        alpha=0.40,
        beta=0.60,
        gamma=0.75,
    )
    loss = loss + 0.04 * boundary_weighted_bce_loss(logits, target)
    loss = loss + 0.015 * soft_centroid_alignment_loss(logits, target)
    if isinstance(outputs, dict) and "prior" in outputs:
        loss = loss + 0.02 * combo_seg_loss(outputs["prior"], target)
    if isinstance(outputs, dict) and "freq_aux" in outputs:
        loss = loss + 0.03 * combo_seg_loss(outputs["freq_aux"], target)
    return loss


def precision_hard_case_combo_loss(outputs, target, boundary_weight: float = 0.06):
    logits = unwrap_logits(outputs)
    loss = boundary_combo_loss(outputs, target, boundary_weight=boundary_weight)
    loss = loss + 0.18 * sample_weighted_tversky_loss(
        logits,
        target,
        alpha=0.72,
        beta=0.28,
        gamma=0.85,
    )
    loss = loss + 0.10 * relative_area_loss(logits, target, mode="over")
    loss = loss + 0.04 * boundary_weighted_bce_loss(logits, target)
    return loss


def recall_hard_case_combo_loss(outputs, target, boundary_weight: float = 0.04):
    logits = unwrap_logits(outputs)
    loss = boundary_combo_loss(outputs, target, boundary_weight=boundary_weight)
    loss = loss + 0.18 * sample_weighted_tversky_loss(
        logits,
        target,
        alpha=0.25,
        beta=0.75,
        gamma=0.70,
    )
    loss = loss + 0.08 * relative_area_loss(logits, target, mode="under")
    loss = loss + 0.025 * soft_centroid_alignment_loss(logits, target)
    return loss


def boundary_shift_combo_loss(outputs, target, boundary_weight: float = 0.10):
    logits = unwrap_logits(outputs)
    loss = boundary_combo_loss(outputs, target, boundary_weight=boundary_weight)
    loss = loss + 0.08 * boundary_weighted_bce_loss(logits, target)
    loss = loss + 0.06 * soft_centroid_alignment_loss(logits, target)
    loss = loss + 0.04 * relative_area_loss(logits, target, mode="symmetric")
    return loss


def compute_loss(outputs, target, loss_mode: str, boundary_weight: float):
    if loss_mode == "tversky_bce":
        return tversky_bce_loss(unwrap_logits(outputs), target)
    if loss_mode == "deep_boundary":
        return deep_boundary_loss(outputs, target, boundary_weight=boundary_weight)
    if loss_mode == "boundary_combo":
        return boundary_combo_loss(outputs, target, boundary_weight=boundary_weight)
    if loss_mode == "prior_boundary_combo":
        return prior_boundary_combo_loss(outputs, target, boundary_weight=boundary_weight)
    if loss_mode == "prior_calibration_combo":
        return prior_calibration_combo_loss(outputs, target, boundary_weight=boundary_weight)
    if loss_mode == "frequency_prior_combo":
        return frequency_prior_combo_loss(outputs, target, boundary_weight=boundary_weight)
    if loss_mode == "posterior_calibration":
        return posterior_calibration_loss(outputs, target, boundary_weight=boundary_weight)
    if loss_mode == "frequency_aux":
        return frequency_aux_loss(outputs, target, boundary_weight=boundary_weight)
    if loss_mode == "hard_case_combo":
        return hard_case_combo_loss(outputs, target, boundary_weight=boundary_weight)
    if loss_mode == "mild_hard_case_combo":
        return mild_hard_case_combo_loss(outputs, target, boundary_weight=boundary_weight)
    if loss_mode == "precision_hard_case_combo":
        return precision_hard_case_combo_loss(outputs, target, boundary_weight=boundary_weight)
    if loss_mode == "recall_hard_case_combo":
        return recall_hard_case_combo_loss(outputs, target, boundary_weight=boundary_weight)
    if loss_mode == "boundary_shift_combo":
        return boundary_shift_combo_loss(outputs, target, boundary_weight=boundary_weight)
    return dice_focal_loss(unwrap_logits(outputs), target)


def unwrap_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["seg"]
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def load_model_state(model: torch.nn.Module, state_dict: dict, partial: bool):
    if not partial:
        return model.load_state_dict(state_dict), []

    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    incompatible = model.load_state_dict(compatible, strict=False)
    return incompatible, skipped


def make_ema_model(model: torch.nn.Module) -> torch.nn.Module:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for key, ema_value in ema_state.items():
        model_value = model_state[key].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(model_value)


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


@torch.no_grad()
def evaluate_threshold_sweep(
    model,
    loader,
    device,
    loss_mode: str,
    boundary_weight: float,
    thr_min: float,
    thr_max: float,
    thr_step: float,
):
    model.eval()
    thresholds = torch.arange(
        thr_min,
        thr_max + thr_step * 0.5,
        thr_step,
        device=device,
    )
    dice_sums = torch.zeros_like(thresholds)
    sample_count = 0
    losses = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        outputs = model(x)
        logits = unwrap_logits(outputs)
        losses.append(float(compute_loss(outputs, y, loss_mode, boundary_weight).item()))

        prob = torch.sigmoid(logits)
        gt_sum = y.sum(dim=(1, 2, 3))
        sample_count += int(y.shape[0])
        for i, threshold in enumerate(thresholds):
            pred = (prob > threshold).float()
            inter = (pred * y).sum(dim=(1, 2, 3))
            pred_sum = pred.sum(dim=(1, 2, 3))
            dice = (2.0 * inter + 1e-6) / (pred_sum + gt_sum + 1e-6)
            dice_sums[i] += dice.sum()

    dice = dice_sums / max(sample_count, 1)
    best_idx = int(torch.argmax(dice).item())
    return (
        float(dice[best_idx].item()),
        float(thresholds[best_idx].item()),
        float(sum(losses) / max(len(losses), 1)),
    )


def build_scheduler(opt, epochs: int, warmup_epochs: int):
    warmup_epochs = max(0, min(warmup_epochs, epochs - 1))
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=epochs,
            eta_min=1e-6,
        )

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


def build_lesion_balanced_sampler(
    dataset: QaTaNPZDataset,
    small_thr: float,
    medium_thr: float,
    empty_weight: float,
    small_weight: float,
    medium_weight: float,
    large_weight: float,
) -> WeightedRandomSampler:
    weights = []
    bins = {"empty": 0, "small": 0, "medium": 0, "large": 0}
    for path in dataset.paths:
        npz_path = path if path.is_absolute() else ROOT / path
        mask = np.load(npz_path)["mask"]
        area_ratio = float((mask > 0.5).mean())
        if area_ratio <= 0.0:
            weight = empty_weight
            bins["empty"] += 1
        elif area_ratio < small_thr:
            weight = small_weight
            bins["small"] += 1
        elif area_ratio < medium_thr:
            weight = medium_weight
            bins["medium"] += 1
        else:
            weight = large_weight
            bins["large"] += 1
        weights.append(weight)
    print(
        "LESION_BALANCED_SAMPLER "
        f"empty={bins['empty']} small={bins['small']} "
        f"medium={bins['medium']} large={bins['large']} "
        f"weights(empty/small/medium/large)="
        f"{empty_weight}/{small_weight}/{medium_weight}/{large_weight}"
    )
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


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
        choices=(
            "dice_focal",
            "tversky_bce",
            "deep_boundary",
            "boundary_combo",
            "prior_boundary_combo",
            "prior_calibration_combo",
            "frequency_prior_combo",
            "posterior_calibration",
            "frequency_aux",
            "hard_case_combo",
            "mild_hard_case_combo",
            "precision_hard_case_combo",
            "recall_hard_case_combo",
            "boundary_shift_combo",
        ),
        help="training loss; hard_case_combo is designed for low-Dice hard cases",
    )
    ap.add_argument("--boundary_weight", type=float, default=0.1)
    ap.add_argument("--resume", type=str, default="", help="optional checkpoint to load model weights from")
    ap.add_argument("--resume_partial", action="store_true", help="load matching keys only; useful when warm-starting v13 from v11/v12")
    ap.add_argument("--grad_clip", type=float, default=0.0, help="clip gradient norm after AMP unscale; 0 disables")
    ap.add_argument("--warmup_epochs", type=int, default=0, help="linear warmup epochs before cosine decay")
    ap.add_argument("--augment", action="store_true", help="enable safe train-time CXR augmentations")
    ap.add_argument("--hflip_prob", type=float, default=0.5, help="horizontal flip probability when --augment is enabled")
    ap.add_argument("--intensity_prob", type=float, default=0.8, help="intensity augmentation probability when --augment is enabled")
    ap.add_argument("--noise_prob", type=float, default=0.25, help="noise augmentation probability when --augment is enabled")
    ap.add_argument(
        "--sampler",
        type=str,
        default="shuffle",
        choices=("shuffle", "lesion_balanced"),
        help="lesion_balanced oversamples small/medium lesion masks to reduce low-Dice misses",
    )
    ap.add_argument("--sampler_small_thr", type=float, default=0.015, help="mask area ratio below this is treated as small lesion")
    ap.add_argument("--sampler_medium_thr", type=float, default=0.08, help="mask area ratio below this is treated as medium lesion")
    ap.add_argument("--sampler_empty_weight", type=float, default=0.35)
    ap.add_argument("--sampler_small_weight", type=float, default=3.0)
    ap.add_argument("--sampler_medium_weight", type=float, default=1.8)
    ap.add_argument("--sampler_large_weight", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.0, help="EMA decay for model weights; 0 disables")
    ap.add_argument(
        "--select_metric",
        type=str,
        default="fixed_dice",
        choices=("fixed_dice", "best_threshold"),
        help="checkpoint selection metric on validation split",
    )
    ap.add_argument("--select_thr_min", type=float, default=0.20)
    ap.add_argument("--select_thr_max", type=float, default=0.75)
    ap.add_argument("--select_thr_step", type=float, default=0.02)
    ap.add_argument("--run_tag", type=str, default="", help="optional output tag for checkpoint/tensorboard")

    ap.add_argument(
        "--model",
        type=str,
        default="unet2d",
        help=f"model name, choices: {', '.join(list(available_models()) + ['transunet2d_v3', 'transunet2d_v4', 'transunet2d_v5', 'transunet2d_v6', 'transunet2d_v7', 'transunet2d_v8', 'transunet2d_v9', 'transunet2d_v10', 'transunet2d_v11', 'transunet2d_v12', 'transunet2d_v13', 'transunet2d_v14', 'transunet2d_v16', 'transunet2d_v17', 'transunet2d_v18', 'transunet2d_v19', 'transunet2d_v20'])}",
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

    ds_train = QaTaNPZDataset(
        str(manifest),
        split="train",
        augment=args.augment,
        hflip_prob=args.hflip_prob,
        intensity_prob=args.intensity_prob,
        noise_prob=args.noise_prob,
    )
    ds_val = QaTaNPZDataset(str(manifest), split="val")

    train_sampler = None
    train_shuffle = True
    if args.sampler == "lesion_balanced":
        train_sampler = build_lesion_balanced_sampler(
            ds_train,
            small_thr=args.sampler_small_thr,
            medium_thr=args.sampler_medium_thr,
            empty_weight=args.sampler_empty_weight,
            small_weight=args.sampler_small_weight,
            medium_weight=args.sampler_medium_weight,
            large_weight=args.sampler_large_weight,
        )
        train_shuffle = False

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=train_shuffle,
        sampler=train_sampler,
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
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        incompatible, skipped = load_model_state(model, state_dict, partial=args.resume_partial)
        if args.resume_partial:
            print(
                "partially resumed model weights from "
                f"{args.resume}; missing={len(incompatible.missing_keys)} "
                f"unexpected={len(incompatible.unexpected_keys)} "
                f"skipped_shape={len(skipped)}"
            )
        else:
            print(f"resumed model weights from {args.resume}")

    print(f"LOSS = {args.loss}")
    print(f"WARMUP_EPOCHS = {args.warmup_epochs}")
    print(f"GRAD_CLIP = {args.grad_clip}")
    print(f"AUGMENT = {args.augment}")
    if args.augment:
        print(f"AUGMENT_PROBS = hflip:{args.hflip_prob} intensity:{args.intensity_prob} noise:{args.noise_prob}")
    print(f"SAMPLER = {args.sampler}")
    print(f"EMA_DECAY = {args.ema_decay}")
    print(f"SELECT_METRIC = {args.select_metric}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = build_scheduler(opt, epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=(device.type == "cuda"))

    run_dir = Path("outputs")
    run_tag = args.run_tag.strip() or f"{args.model}_qata_exp"
    ckpt_path = run_dir / f"{run_tag}_best.pt"
    tb_dir = run_dir / "tensorboard" / run_tag
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))

    ema_model = make_ema_model(model) if args.ema_decay > 0 else None
    best_score = -1.0

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
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            if ema_model is not None:
                update_ema_model(ema_model, model, args.ema_decay)

            pbar.set_postfix(loss=float(loss.item()))

        sched.step()

        eval_model = ema_model if ema_model is not None else model
        val_dice, val_loss = evaluate(eval_model, dl_val, device, args.loss, args.boundary_weight)
        val_best_dice = val_dice
        val_best_thr = 0.5
        if args.select_metric == "best_threshold":
            val_best_dice, val_best_thr, val_loss = evaluate_threshold_sweep(
                eval_model,
                dl_val,
                device,
                args.loss,
                args.boundary_weight,
                args.select_thr_min,
                args.select_thr_max,
                args.select_thr_step,
            )

        writer.add_scalar("val/dice", val_dice, epoch)
        writer.add_scalar("val/best_threshold_dice", val_best_dice, epoch)
        writer.add_scalar("val/best_threshold", val_best_thr, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)

        score = val_best_dice if args.select_metric == "best_threshold" else val_dice
        print(
            f"[Epoch {epoch}] val_dice={val_dice:.4f} "
            f"val_best_dice={val_best_dice:.4f} val_best_thr={val_best_thr:.3f} "
            f"val_loss={val_loss:.4f}"
        )

        if score > best_score:
            best_score = score
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": eval_model.state_dict(),
                    "model_name": args.model,
                    "loss": args.loss,
                    "boundary_weight": args.boundary_weight,
                    "ema_decay": args.ema_decay,
                    "augment": args.augment,
                    "select_metric": args.select_metric,
                    "selection_score": score,
                    "val_dice": val_dice,
                    "val_best_dice": val_best_dice,
                    "val_best_threshold": val_best_thr,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"Saved best -> {ckpt_path}")

    writer.close()
    print(f"Done. Best selection score = {best_score:.4f}")


if __name__ == "__main__":
    main()
