# EE 641: Search-Free PPO V10 -> V100

This update **continues from the PPO V10 checkpoint**. It does not start from V0 again.

## Installation

Copy the files from this ZIP into `B:\EE641_Chess_RL_Starter\` and replace `run.cmd`.
Keep your existing `.venv`, `checkpoints`, and `outputs` folders.

## Run

Open CMD in the project folder and type:

```cmd
run.cmd ppo-long
```

The launcher first runs all the project tests.

## What it does

- Resumes from `checkpoints\ppo_v0010.pt`, including the optimizer state.
- If `ppo_v0010.pt` is missing, it can recover V10 from `ppo_train_latest.pt` **only if** that checkpoint is exactly iteration 10.
- Trains versions 11 through 100, **128 games per iteration**. It does not retrain the first 10 iterations.
- This is **11,520 new games**, plus the **320 games** already played for V1–V10: **11,840 total**, not 12,800. The first 10 iterations had 32 games each.
- Continues the same PPO update method, legal action mask, terminal reward of +1/0/-1, and value bootstrap for a truncated game.
- Uses 50% recent frozen snapshot, 50% uniformly drawn from archived V0, V10, V20, etc. All opponents are neural policies.
- Saves the learner and AdamW optimizer state after **every iteration**, so restarting the same command resumes the last completed iteration. If interrupted during a game batch, that incomplete iteration is replayed.
- Freezes and permanently saves `ppo_v0020.pt`, `ppo_v0030.pt`, ..., `ppo_v0100.pt`; existing archives are never overwritten.
- At V10 and every tenth version, evaluates against frozen V0, V10, the 1M supervised model (if present), and random legal moves. Uses the **same 50 starting positions with colors reversed**, so 100 games per opponent and version.
- Records actual completed-game wins, draws, losses, truncations, game termination reasons, and a paired-bootstrap interval for completed paired-opening scores.

## No search

**Training:** No minimax, alpha-beta, MCTS, chess engine, engine evaluations, or expert-move labels. Each move uses one neural-network forward pass plus a legal-action mask.

**Evaluation:** Also no search or engine. Fixed opponents and random moves only. `python-chess` determines legal moves and valid game termination.

## Output files

- `checkpoints\ppo_long_latest.pt` : latest resumable learner and optimizer (separate from `ppo_train_latest.pt`).
- `checkpoints\ppo_v0020.pt`, `ppo_v0030.pt`, ... : permanent archives.
- `outputs\ppo_long_metrics.csv` : iterations 11–100 and PPO metrics.
- `outputs\ppo_long_eval_v0010.json`, `ppo_long_eval_v0020.json`, ... : fixed-opponent evaluations.
- `outputs\ppo_long_games_v0010.csv`, etc. : results of each individual evaluation game.
- `outputs\ppo_long_evaluation_summary.csv` : all V10/V20/... results in one table.

To rerun or recover only missing evaluations of archived models:

```cmd
run.cmd ppo-long-eval
```

**Do not run `run.cmd ppo-train` to continue this new experiment.** That older command belongs to the separate 32-game-per-iteration experiment.

## Interpreting results

Self-play win rates against a changing opponent pool **are not an estimate of Elo or of improvement over V0**. Focus on V10/V20/V30/... scores against the same frozen V0 and V10 opponents and identical starting positions. A 100-game tournament is still preliminary. The paired-bootstrap intervals show uncertainty and only include complete color-swapped pairs with no artificial cutoffs.

The underlying `ppo_update` helper currently reports **last-minibatch** policy loss, value loss, and approximate KL. These are diagnostics, not proof of chess improvement. V10/V20/... matched-opponent performance is the primary experimental comparison.

Python source is intentionally kept small and uses only dependencies already installed for the previous PPO update.
