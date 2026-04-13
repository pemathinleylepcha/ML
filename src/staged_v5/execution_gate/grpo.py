from __future__ import annotations

import torch
import torch.nn.functional as F

from staged_v5.execution_gate.contracts import NeuralGateAction


def _scale_rewards(
    rewards: torch.Tensor,
    actions: torch.Tensor,
    filled_mask: torch.Tensor | None,
    positive_fill_reward_boost: float,
) -> torch.Tensor:
    scaled_rewards = rewards.detach().clone()
    if filled_mask is None:
        return scaled_rewards
    positive_mask = filled_mask.bool() & (scaled_rewards > 0.0)
    non_market_mask = actions != int(NeuralGateAction.MARKET_NOW)
    boost_mask = positive_mask & non_market_mask
    scaled_rewards[boost_mask] = scaled_rewards[boost_mask] * positive_fill_reward_boost
    return scaled_rewards


def grpo_update_step(
    gate_network,
    optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    old_log_probs: torch.Tensor,
    *,
    reference_logits: torch.Tensor | None = None,
    filled_mask: torch.Tensor | None = None,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.01,
    positive_fill_reward_boost: float = 2.0,
    zero_variance_skip_epsilon: float = 1e-5,
) -> dict[str, float | bool]:
    scaled_rewards = _scale_rewards(rewards, actions, filled_mask, positive_fill_reward_boost)
    std_reward = scaled_rewards.std(unbiased=False) + 1e-8
    if float(std_reward.item()) < float(zero_variance_skip_epsilon):
        return {"skipped": True, "zero_variance": True, "loss": 0.0, "mean_reward": float(rewards.mean().item())}

    advantages = torch.clamp((scaled_rewards - scaled_rewards.mean()) / std_reward, -3.0, 3.0)
    logits = gate_network(states)
    dist = torch.distributions.Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    kl_penalty = torch.tensor(0.0, device=states.device)
    if reference_logits is not None:
        kl_penalty = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.log_softmax(reference_logits, dim=-1),
            log_target=True,
            reduction="batchmean",
        )
    loss = policy_loss + kl_beta * kl_penalty
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        "skipped": False,
        "zero_variance": False,
        "loss": float(loss.item()),
        "mean_reward": float(rewards.mean().item()),
        "kl_penalty": float(kl_penalty.item()),
    }
