# EE 641 Chess RL: Safe PPO Speed Test (No Training)

**Why this update:** the previous latest-only PPO branch generated 512 self-play games one at a time, issuing one small GPU call per move. This update implements a separate batched rollout collector to investigate speed without changing the PPO algorithm, old weights or any current Lichess bot. Do not launch another 60k+ training games until the benchmark is verified.

## Installation

Unzip `EE641_PPO_Batch_Benchmark_Update.zip` into `B:\EE641_Chess_RL_Starter\`. Replace only `run.cmd` if prompted. This update adds:

- `chess_rl\ppo_batch.py` (new, independent parallel rollout collector)
- `chess_rl\ppo_batch_bench.py` (read-only test and timing)
- `chess_rl\research_snapshot.py` (copies October results and records checkpoint SHA-256s)
- `tests\test_ppo_batch.py` (CPU correctness tests)
- `README_PPO_BATCH.md` (this document)
- Updated `run.cmd`, preserving every previous command, including Maia, Lichess and PPO scaling

The files `ppo_train.py`, `ppo_scale.py`, `ppo_scale_metrics.py`, model weights, optimizer states and any earlier evaluation results are **not overwritten**. The Lichess bot continues to use the same frozen Latest S49 checkpoint.

## Step 1: Preserve the independent confirmation result

```
run.cmd research-snapshot
```

This copies the October tournament manifest, 3,072-game data, summary, exact October starting positions, and key benchmark source files into a timestamped folder under `research_archive\`. It verifies that the primary 256 matched opening pairs are complete. It also records SHA-256 hashes for V0, V100 and Latest S49. **The checkpoint weights are not duplicated**; separately back up the `checkpoints\` directory if you need full recoverability. Never share Lichess tokens; this script does not read them.

## Step 2: Verify batched collector

```
run.cmd ppo-batch-smoke
```

This runs four small CPU unit tests, compares serial versus batched decisions on the frozen checkpoint, and completes **four short simulated games** with both collectors. It does not perform any gradient steps. The checkpoint hash is pinned to the exact model previously used in Lichess:

`75bec742766a4c9709211ade12719064bc0eb6405a4931314cd7d2178f04a748`

It checks legal actions, board states, old PPO log probabilities, critic values, terminal/cutoff bookkeeping and advantage/return outputs. As with all GPU inference, a very close numerical difference may occur because different batch sizes use different CUDA kernels; the unit tests use a small tolerance.

## Step 3: Benchmark a realistic small workload

```
run.cmd ppo-batch-bench
```

This processes 32 human-opening self-play games under a move-limit cap of 80, comparing the original serial collector with 32 concurrent games through the batched collector. Both use the same frozen S49 policy; **neither trains or modifies weights**. It prints total seconds, positions processed per second, and speedup. Saves `outputs\ppo_batch_benchmark.json`.

**Interpretation:** sampled games will not be identical because batching consumes random draws in another order. The game throughput comparison is therefore approximate. The stronger correctness checks compare deterministic greedy moves, masks, log probabilities and values on the **same exact positions**, and the one-game test compares rollout rewards and bootstrapping.

## What happens after a passing benchmark

1. If CUDA speed actually improves, integrate batching into a **new** isolated continuation from Latest S49, with the same PPO hyperparameters and optimizer state.
2. Keep `checkpoints\ppo_scale_latest\s0049.pt` read-only; save new weights to a new directory and record the new rollout collector version in the training manifest.
3. Train in stages toward at least 50,000, 75,000 and 100,000 cumulative PPO games. Because each iteration is 512 games, the planned exact steps may cross these round numbers slightly.
4. Use a separate development-test opponent set during training. Do **not** repeatedly use the October pre-registered confirmation dataset as a checkpoint selector. Reserve new independent human openings for the next final confirmation.
5. Keep the original H3 live-game fine-tuning comparison as the next separate experiment.

**No chess search:** both collectors run the model on the current position once per move. Python-chess only generates legal moves, checks game completion and preserves board history. The new code never imports Stockfish, Lc0, Maia, minimax or MCTS.

**Validation limitations:** locally syntax-checked and exercised the collector with a simulated chess-board interface, but the runtime used to build this ZIP lacks the `python-chess` package and RTX 3090. The smoke test on your machine is required before trusting speed or moving into further training.
