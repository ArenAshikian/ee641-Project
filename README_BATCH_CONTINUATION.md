# EE641 chess PPO: isolated batched continuation from frozen Latest S49

This update adds `chess_rl/ppo_batch_continue.py`, a small unit test and five new `run.cmd` commands. It uses the already-installed `chess_rl/ppo_batch.py` from the successful 7.12x benchmark. Extract into `B:\EE641_Chess_RL_Starter` and replace `run.cmd` if asked.

## Safety

- Frozen starting checkpoint: `checkpoints\ppo_scale_latest\s0049.pt` (SHA256 `75bec742766a4c9709211ade12719064bc0eb6405a4931314cd7d2178f04a748`).
- Existing Latest S49 and Lichess bot checkpoint are NEVER overwritten. The original September PPO and October Maia evidence remain untouched.
- New results live under `checkpoints\ppo_batch_continue\` and `outputs\ppo_batch_continue_metrics.csv`.
- Real training resumes from the saved AdamW optimizer state, not weights alone.
- Only latest-only self-play is continued: a frozen snapshot plays both sides during each rollout iteration, followed by the existing `detailed_ppo_update` PPO update.
- No Stockfish, tree search, MCTS, opening book, or engine moves; human-game opening sequences are *only* initial positions, not labels or rewards.
- Other training hyperparameters remain the same: 512 games/step, 240-ply limit, PPO epochs=2, PPO minibatch=64, LR=1e-5, gamma=1.0, GAE lambda=0.95, clip=0.1. Only GPU rollout collection is batched (32 positions/inference), and the random seed changes to make a reproducible new branch.
- Outcomes and PPO samples are saved after every full training step. Ctrl+C may lose *at most the current unfinished iteration*. Restart the same milestone command to resume.
- Training samples are stochastic and **will not** reproduce the exact same chess trajectories as sequential rollouts. The observed 7.12x benchmark is a rollout throughput improvement only, not a guarantee of end-to-end PPO runtime.

## Commands

Close `run.cmd lichess-play` first to avoid competing for GPU resources.

1. Verify a REAL optimization step in GPU memory only; no files changed:

   `run.cmd ppo-continue-smoke`

2. After smoke passes, continue Latest S49 to the first preserved milestone:

   `run.cmd ppo-continue-50k`

   This runs 26 new 512-game iterations. It will stop at **50,240 cumulative games**, then preserve the checkpoint as `checkpoints\ppo_batch_continue\s0026.pt`.

3. Later resume to about 75k:

   `run.cmd ppo-continue-75k`

   Ends at 75,328 cumulative games, saves `s0075.pt`.

4. Later resume to about 100k:

   `run.cmd ppo-continue-100k`

   Ends at 100,416 cumulative games, saves `s0124.pt`.

5. Anytime, inspect saved progress:

   `run.cmd ppo-continue-report`

## Research caution

The October Maia 1900 confirmation was preregistered and should stay **sealed**. Do not repeatedly evaluate these new checkpoints against those October test openings to pick a winner. We should use a separate development set for model selection and new unseen confirmation openings for any final claim. One statistically significant p=0.0493 comparison is encouraging but borderline, not proof that longer training will necessarily help.

## Validation

Code compiled and archive was inspected on the authoring side. The model checkpoint, GPU, source datasets, and all existing Python dependencies are on your Windows PC, so the full smoke and end-to-end training cannot be executed here. Do not train until `run.cmd ppo-continue-smoke` passes.
