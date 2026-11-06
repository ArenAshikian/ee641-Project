# PPO batch smoke test: GPU precision audit

This patch changes **only** `chess_rl/ppo_batch_bench.py`, the read-only benchmark script. The batched collector, original PPO implementation, network, optimizer, checkpoints, and Lichess bot are all unchanged.

## Why the earlier test failed

The previous benchmark demanded absolute agreement of policy log probabilities within 0.0001 between batch-size-one and batched CUDA inference on a 10-block neural network. Different CUDA convolution algorithms may result in small float32 differences. The error does not by itself establish that the batched collector is incorrect. The four CPU unit tests and the exact move/action/mask/state comparisons passed before the error.

## What the revised smoke test checks

* Exact greedy move, action ID, legal mask and board encoder on the same positions.
* Critic values within 0.005, and the selected action's log probabilities within 0.02. The maximum observed differences are printed.
* **Every legal action's probability** via full-distribution total variation (TV) <= 0.02, reported separately. Any deviation beyond the bound fails the test, with diagnostic values.
* Four real sampled games, their outcomes, and GAE data; no training or model modifications.

The bounds are engineering tolerances, not a scientific assertion of numerically identical trajectory distributions. In particular, different RNG ordering produces different sampled games, and the numerical results should be inspected before integrating the collector into continued PPO training.

## Installation

Extract this ZIP into `B:\EE641_Chess_RL_Starter\`, replacing `chess_rl\ppo_batch_bench.py` only. Do not replace model files.

Then run:

```cmd
run.cmd ppo-batch-smoke
```

If it passes, run:

```cmd
run.cmd ppo-batch-bench
```

If it fails, share the numerical audit/errors. Do not continue PPO training before the safety check passes.
