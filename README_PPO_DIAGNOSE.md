# Search-Free PPO Plateau Diagnosis

Extract this ZIP into `B:\EE641_Chess_RL_Starter\`, replacing `run.cmd`.

Run one command from the project directory:

    run.cmd ppo-diagnose

This runs the existing tests and four new diagnostic tests. Then it analyzes
V0 and V100 using 64 sampled games for each model against frozen V0 (128 total)
from the same openings and both colors. No Stockfish, minimax, MCTS, heuristic
search, engine annotations, or engine labels are used. Moves use one model pass
and python-chess legality checks only.

The command measures:

- Stochastic game score, to compare with the separate greedy championship.
- Value prediction MSE and R2 against game outcomes (+1, 0, -1), compared with
  a constant outcome baseline, separately for White and Black.
- V0 versus V100 greedy action agreement and policy entropy on identical
  sampled game positions. High entropy means more spread over legal moves.
- Total learner decisions, optimizer steps, and opponent mixture using your
  existing `outputs/ppo_long_metrics.csv`.

The saved report is `outputs/ppo_plateau_diagnostics.json`.

Cautions:
- Value accuracy is measured against eventual game outcomes, which is a noisy
  signal. Many positions within each game are correlated. Negative R2 means
  worse than a constant-prediction baseline on these sampled positions.
- 64 games per candidate are only exploratory; do not treat small score changes
  as proof of stronger chess.
- Previously logged PPO KL, entropy, losses, and clip fractions are taken from
  the LAST minibatch only, so the old training logs do not give epoch-wide means.
- The historical PPO training logs do not include W/D/L by named opponent.
- This command does not train, update, delete, or replace any checkpoints or
  training files. It is safe to rerun.

If results suggest a weak value network or training/evaluation mismatch, the
next step is a controlled ablation branching from the same archived checkpoint,
not further uninterrupted PPO training.
