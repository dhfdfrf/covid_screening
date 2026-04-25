from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_actions: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Parameter(torch.full((num_actions,), -1.0))
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = self.mu(h)
        value = self.value(h).squeeze(-1)
        std = self.log_std.exp().expand_as(mu)
        return mu, std, value

    def sample_action(self, x: torch.Tensor):
        mu, std, value = self.forward(x)
        dist = Normal(mu, std)
        raw = dist.rsample()
        log_prob = dist.log_prob(raw).sum(dim=-1)
        weights = torch.softmax(raw, dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return weights, raw, log_prob, entropy, value

    def evaluate_action(self, x: torch.Tensor, raw_action: torch.Tensor):
        mu, std, value = self.forward(x)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        weights = torch.softmax(raw_action, dim=-1)
        return weights, log_prob, entropy, value


@dataclass
class PPOBatch:
    states: torch.Tensor
    raw_actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
