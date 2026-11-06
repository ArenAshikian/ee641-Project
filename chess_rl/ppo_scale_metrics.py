"""Detailed PPO minibatch metrics for the new experiments only.

Does not modify the old V0-V100 PPO trainer or its saved output files.
"""

import numpy as np
import torch

from chess_rl.ppo_math import masked_distribution, ppo_objective


def explained_variance(returns, predictions):
    """1 - var(target - prediction)/var(target); negative is worse than baseline."""
    targets = np.asarray(returns, dtype=np.float64)
    old = np.asarray(predictions, dtype=np.float64)
    if len(targets) < 2 or targets.var() < 1e-12:
        return None
    return float(1.0 - np.var(targets - old) / np.var(targets))


def summarize_experiences(experiences):
    raw_advantages = np.asarray([item['advantage'] for item in experiences], dtype=np.float64)
    returns = np.asarray([item['return'] for item in experiences], dtype=np.float64)
    values = np.asarray([item['value'] for item in experiences], dtype=np.float64)
    return {
        'raw_advantage_mean': float(raw_advantages.mean()),
        'raw_advantage_std': float(raw_advantages.std()),
        'raw_advantage_abs_mean': float(np.abs(raw_advantages).mean()),
        'return_mean': float(returns.mean()),
        'return_std': float(returns.std()),
        'rollout_value_mse': float(np.mean((returns - values)**2)),
        'rollout_value_explained_variance': explained_variance(returns, values),
    }


def detailed_ppo_update(model, optimizer, experiences, device, epochs=2,
                        batch_size=64, clip=0.1, value_weight=0.5,
                        entropy_weight=0.01, seed=1):
    """Same PPO objective as V0-V100, with sample-weighted epoch-wide diagnostics."""
    if not experiences:
        raise ValueError('No learner actions to update.')
    initial = summarize_experiences(experiences)
    model.train()
    # Maintain the exact same BN running statistics during behavior sampling and update.
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.eval()
    states = np.stack([s['board'] for s in experiences])
    masks = np.stack([s['mask'] for s in experiences])
    actions = torch.tensor([s['action'] for s in experiences], device=device, dtype=torch.long)
    old_log_probs = torch.tensor([s['old_log_prob'] for s in experiences], device=device)
    old_values = torch.tensor([s['value'] for s in experiences], device=device)
    adv = torch.tensor([s['advantage'] for s in experiences], device=device)
    returns = torch.tensor([s['return'] for s in experiences], device=device)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    rng = np.random.default_rng(seed)
    metric_names = ('policy_loss', 'value_loss', 'entropy', 'clip_fraction', 'approx_kl')
    sums = {name: 0.0 for name in metric_names}
    sums['approx_kl_positive'] = 0.0
    total_samples = 0
    steps = 0
    clipped_steps = 0
    max_kl = 0.0
    for epoch in range(epochs):
        indices_list = np.array_split(rng.permutation(len(experiences)),
                                      max(1, (len(experiences) + batch_size - 1)//batch_size))
        for indices in indices_list:
            if len(indices) < 2:
                continue
            x = torch.from_numpy(states[indices].astype(np.float32)).to(device)
            legal = torch.from_numpy(masks[indices]).to(device)
            idx = torch.from_numpy(indices).to(device)
            logits, new_values = model(x)
            loss, metrics = ppo_objective(
                logits, new_values, legal, actions[idx], old_log_probs[idx],
                old_values[idx], adv[idx], returns[idx], clip=clip,
                value_weight=value_weight, entropy_weight=entropy_weight,
            )
            with torch.no_grad():
                dist = masked_distribution(logits, legal)
                delta = dist.log_prob(actions[idx]) - old_log_probs[idx]
                ratio = torch.exp(delta)
                # KL estimator used by Schulman, nonnegative for any single sample.
                kl_positive = ((ratio - 1.0) - delta).mean().item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            count = len(indices)
            for name in metric_names:
                sums[name] += metrics[name] * count
            sums['approx_kl_positive'] += kl_positive * count
            max_kl = max(max_kl, float(kl_positive))
            total_samples += count
            steps += 1
            clipped_steps += int(metrics['clip_fraction'] > 0.1)
    model.eval()
    if steps == 0:
        raise RuntimeError('No optimizer update was performed.')
    aggregate = {key: val / total_samples for key, val in sums.items()}
    aggregate.update(initial)
    aggregate.update({'optimizer_steps': steps, 'transitions': len(experiences),
                      'minibatches_clip_fraction_gt_0_1': clipped_steps,
                      'max_minibatch_approx_kl_positive': max_kl})
    return aggregate
