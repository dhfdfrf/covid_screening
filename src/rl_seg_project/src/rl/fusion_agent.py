from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model_factory import ModelBuildConfig, build_model
from src.rl.policy import PolicyNet, PPOBatch
from src.utils.losses import dice_score_from_probs


class FeatureAdapter(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.LazyLinear(out_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        pooled = feat.mean(dim=(-2, -1))
        return self.proj(pooled)


class RLDynamicFusionSegmenter(nn.Module):
    def __init__(self, image_size=(224, 224), freeze_backbones: bool = True):
        super().__init__()
        self.transunet = build_model(ModelBuildConfig(name="transunet2d", image_size=image_size, return_features=True))
        self.swin = build_model(ModelBuildConfig(name="swin_unet2d", image_size=image_size, return_features=True))
        self.uctransnet = build_model(ModelBuildConfig(name="uctransnet2d", image_size=image_size, return_features=True))

        self.adapters = nn.ModuleList([FeatureAdapter(128), FeatureAdapter(128), FeatureAdapter(128)])
        self.policy = PolicyNet(in_dim=128 * 3, hidden_dim=256, num_actions=3)
        self.freeze_backbones = freeze_backbones
        if freeze_backbones:
            self.set_backbone_requires_grad(False)

    def set_backbone_requires_grad(self, flag: bool):
        for net in [self.transunet, self.swin, self.uctransnet]:
            for p in net.parameters():
                p.requires_grad = flag

    def extract(self, image: torch.Tensor):
        if self.freeze_backbones:
            with torch.no_grad():
                logit_t, feat_t = self.transunet(image)
                logit_s, feat_s = self.swin(image)
                logit_u, feat_u = self.uctransnet(image)
        else:
            logit_t, feat_t = self.transunet(image)
            logit_s, feat_s = self.swin(image)
            logit_u, feat_u = self.uctransnet(image)
        state = torch.cat([
            self.adapters[0](feat_t),
            self.adapters[1](feat_s),
            self.adapters[2](feat_u),
        ], dim=-1)
        logits = torch.stack([logit_t, logit_s, logit_u], dim=1)
        return logits, state

    def fuse_with_weights(self, logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        w = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fused = (w * logits).sum(dim=1)
        return fused

    def forward(self, image: torch.Tensor):
        logits, state = self.extract(image)
        weights, raw_actions, log_prob, entropy, value = self.policy.sample_action(state)
        fused_logits = self.fuse_with_weights(logits, weights)
        return {
            "branch_logits": logits,
            "state": state,
            "weights": weights,
            "raw_actions": raw_actions,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "fused_logits": fused_logits,
        }

    @torch.no_grad()
    def deterministic_forward(self, image: torch.Tensor):
        logits, state = self.extract(image)
        mu, _, value = self.policy(state)
        weights = torch.softmax(mu, dim=-1)
        fused_logits = self.fuse_with_weights(logits, weights)
        return {
            "branch_logits": logits,
            "state": state,
            "weights": weights,
            "value": value,
            "fused_logits": fused_logits,
        }


def compute_reward(fused_logits: torch.Tensor, branch_logits: torch.Tensor, target: torch.Tensor, alpha_iou: float = 0.2):
    probs = torch.sigmoid(fused_logits)
    base_probs = torch.sigmoid(branch_logits).mean(dim=1)
    dice = dice_score_from_probs(probs, target)
    base_dice = dice_score_from_probs(base_probs, target)
    inter = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter + 1e-6
    iou = inter / union
    reward = (dice - base_dice) + alpha_iou * iou
    return reward, {"dice": dice.mean().item(), "baseline_dice": base_dice.mean().item(), "iou": iou.mean().item()}


def ppo_update(model: RLDynamicFusionSegmenter, optimizer: torch.optim.Optimizer, batch: PPOBatch, clip_eps: float = 0.2, vf_coef: float = 0.5, ent_coef: float = 0.01, epochs: int = 4):
    stats = {}
    for _ in range(epochs):
        weights, log_prob, entropy, value = model.policy.evaluate_action(batch.states, batch.raw_actions)
        ratio = torch.exp(log_prob - batch.old_log_probs)
        surr1 = ratio * batch.advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch.advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(value, batch.returns)
        entropy_loss = -entropy.mean()
        loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
        optimizer.step()
        stats = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.mean().item()),
            "avg_w0": float(weights[:, 0].mean().item()),
            "avg_w1": float(weights[:, 1].mean().item()),
            "avg_w2": float(weights[:, 2].mean().item()),
        }
    return stats
