"""Small PPO math helpers. No chess engine, search, or move labels."""

import numpy as np
import torch


def compute_gae(rewards, values, bootstrap_value, gamma=1.0, gae_lambda=0.95):
    """GAE over consecutive decisions by ONE player (opponent turns in between).

    rewards[-1] is the true terminal outcome for completed games.
    Truncated games instead bootstrap from the value of the next decision state.
    """
    if len(rewards) != len(values):
        raise ValueError('Rewards and values must have the same length.')
    advantages = np.zeros(len(rewards), dtype=np.float32)
    following_value = float(bootstrap_value)
    following_advantage = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        delta = float(rewards[i]) + gamma * following_value - float(values[i])
        following_advantage = delta + gamma * gae_lambda * following_advantage
        advantages[i] = following_advantage
        following_value = float(values[i])
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def masked_distribution(logits, legal_mask):
    """Only legal actions have nonzero sampling probability."""
    if logits.shape != legal_mask.shape:
        raise ValueError('Logits and legal mask must have matching shapes.')
    if not bool(legal_mask.any(dim=-1).all()):
        raise ValueError('Every position must have at least one legal action.')
    masked = logits.masked_fill(~legal_mask, -1e9)
    return torch.distributions.Categorical(logits=masked)


def ppo_objective(logits, values, legal_masks, actions, old_log_probs,
                  old_values, advantages, returns, clip=0.1,
                  value_weight=0.5, entropy_weight=0.01):
    dist = masked_distribution(logits, legal_masks)
    new_log_probs = dist.log_prob(actions)
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    # Conservative clipped critic update, relative to rollout value estimates.
    value_clipped = old_values + torch.clamp(values - old_values, -clip, clip)
    value_loss = 0.5 * torch.maximum((values - returns).square(),
                                     (value_clipped - returns).square()).mean()
    entropy = dist.entropy().mean()
    total = policy_loss + value_weight * value_loss - entropy_weight * entropy
    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > clip).float().mean().item()
        approx_kl = (old_log_probs - new_log_probs).mean().item()
    return total, {
        'policy_loss': float(policy_loss.detach()),
        'value_loss': float(value_loss.detach()),
        'entropy': float(entropy.detach()),
        'clip_fraction': clip_fraction,
        'approx_kl': approx_kl,
    }
