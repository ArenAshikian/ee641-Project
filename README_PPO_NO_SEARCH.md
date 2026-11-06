# EE 641: Search-free PPO pilot

This adds a small PPO self-play implementation to the chess project. **No search** is used to choose moves, generate training signals, or evaluate the models. This includes no minimax, alpha-beta, MCTS, Stockfish, engine scores, opening-book queries, or tablebases. `python-chess` is only used for legal moves and official game rules. Each actual played move requires **one network forward pass**.

## Install

Extract the ZIP, copy its `chess_rl` folder, `tests` folder, and `run.cmd` into your existing `B:\EE641_Chess_RL_Starter\`. Agree to replace `run.cmd`. Keep `checkpoints\v0_supervised.pt`, your earlier checkpoints, the Lichess archive, and `.venv`. No new dependencies are needed.

## First run

Open CMD from your project directory and enter:

    run.cmd ppo-pilot

It runs project tests and:

1. Evaluates permanent V0 against frozen V0, your earlier 1M model (if available), and a random-legal opponent. Uses identical opening move **sequences** for both colors and preserves the opening move history for the 61-plane encoder. Four opening pairs per opponent are only a smoke test, not a strength estimate.
2. Runs **2 PPO iterations**, each with **8 full chess games** (half playing White, half Black), against either a frozen copy of the current policy or a historical frozen V0 opponent.
3. Updates only the **learner** with PPO: masked action probabilities, clipped objective, learned value estimates, generalized advantage estimation (GAE), legal moves, entropy bonus, and terminal win/draw/loss. The opponent is never updated during a rollout.
4. Evaluates the pilot again on the same fixed openings and opponents, saving the results separately.

A player win is +1, draw 0, and loss -1. A 240-ply cutoff is **truncation, not a draw**. The rollout continues to the learner's next decision boundary and bootstraps the value function at that position. This avoids treating a cut-off game as a genuine result.

### Pilot output

- `checkpoints\ppo_pilot_latest.pt` : resumable pilot model and optimizer, separate from the main experiment.
- `outputs\ppo_pilot_metrics.csv` : losses, approximate KL, entropy, wins/draws/losses, truncated games, timings.
- `outputs\ppo_pilot_v0_baseline.json` : pre-PPO paired-color match results.
- `outputs\ppo_pilot_after.json` : post-PPO paired-color match results.

**The 20-million-position V0 (`checkpoints\v0_supervised.pt`) is never overwritten.**

## Only after the pilot works

Run:

    run.cmd ppo-train

This starts a separate experiment initialized from V0: 10 iterations, 32 games per iteration, two PPO optimization epochs per iteration. It saves `checkpoints\ppo_train_latest.pt` after every iteration, and `checkpoints\ppo_v0010.pt` permanently at iteration 10. Rerun the same command to resume. For later experiments, we can increase the number of iterations while keeping the configuration fixed. Historical frozen opponents come 50% of the time from the archive pool (V0, V10, V20, etc.); 50% from a frozen snapshot of the current learner at the start of the iteration. No overwritten old snapshots or chess engine guidance.

After the main run, use:

    run.cmd ppo-eval

This performs 20 paired-color opening matches per available opponent, comparing the current PPO policy against frozen V0 and 1M models. Outputs `outputs\ppo_trained_eval.json`.

## Important limitations

- The first pilot and ten-iteration run are **correctness and feasibility experiments**, not sufficient to establish chess improvement. The real experiment must use many more games and balanced, independent fixed-opponent matches with confidence intervals. Elo and best-move accuracy are not measured here.
- Evaluation is also search-free. If we later choose to analyze centipawn loss using a chess engine, that must be a **separate, explicitly opt-in offline evaluation process**, never used in training. It is not included in this update.
- The game state uses the existing 61-plane representation and 4,672 actions. Opening move history is preserved when setting up starting boards. Repetition claims and official draws use `python-chess`.
- Training batches keep the pretrained BatchNorm running statistics fixed so rollout and update log-probabilities are comparable.
- Real inference is sequential and can be much slower than supervised batch training. Estimate self-play speed from the pilot rather than assuming it will match supervised training throughput.
- Self-play wins/draws/losses are relative to the pool, not independently verified playing strength. Many draws, unstable KL, worsening fixed-opponent results, or excessive truncations are warning signals.
- The rollout policy and opponent are frozen during each PPO iteration. Archived models are used unchanged. Checkpoint files record iteration, hyperparameters, source V0 SHA-256, metrics, and optimizer state.
- The code was assembled with AI assistance; review and disclose as required by your course policy.

## Recommended sequence

1. `run.cmd ppo-pilot`, inspect its full output and JSON/CSV files.
2. Review PPO math and test behavior, then only if stable run `run.cmd ppo-train`.
3. Evaluate against fixed, never-trained-on checkpoints and compare with V0.
4. Increase self-play games and iterations based on measured time and quality, not an arbitrary epoch count.
