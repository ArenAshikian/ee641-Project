# EE 641: V0 vs V100, greedy vs sampled (no search)

This update is a **read-only evaluation**. It does not start PPO training, modify model weights, download any data, or use a chess engine. Every move is chosen from a **single network forward pass** plus a legal-move mask.

## Install and run

1. Extract this ZIP into `B:\EE641_Chess_RL_Starter\`, keeping the `chess_rl` and `tests` subdirectories.
2. Allow Windows to replace `run.cmd`. Do not delete `.venv`, data, checkpoints, or earlier evaluation files.
3. In CMD, run:

```cmd
run.cmd ppo-policy-test
```

It first runs the project's tests, then evaluates four conditions against a frozen **greedy V0** opponent:

| Candidate | Candidate move choice | Opponent | Games |
| --- | --- | --- | ---: |
| V0 | Greedy (highest legal logit) | V0 greedy | 400 |
| V100 | Greedy | V0 greedy | 400 |
| V0 | Sampled from legal-masked softmax | V0 greedy | 400 |
| V100 | Sampled | V0 greedy | 400 |

Each condition uses the **same 200 opening positions**, one game with the candidate as White and one as Black. All candidates receive the same per-opening/color random seed for stochastic sampling. The opponent stays greedy V0 in every case to isolate the candidate's move-selection method. There is **no look-ahead, minimax, MCTS, alpha-beta, Stockfish, or best-move labels**.

The starting positions are newly generated, exclude known previous training/evaluation **starting positions** where available, and exclude the checkpoint championship openings. This cannot guarantee that the positions never appeared as intermediate states elsewhere.

## Metrics and how to interpret them

- Each condition reports wins, draws, losses, truncations, score, and a 95% bootstrap confidence interval that resamples the whole White/Black **opening pair**.
- Artificial 240-ply limits are **truncations, not draws**. A pair with a truncated game is omitted from paired statistics.
- The primary analysis matches all four conditions on opening pairs completed in all four conditions, and calculates:
  1. **Greedy gain:** `V100 greedy score - V0 greedy score`.
  2. **Sampled gain:** `V100 sampled score - V0 sampled score`.
  3. **Difference-of-differences:** `sampled gain - greedy gain`.
- Positive sampled gain with little greedy gain would support the hypothesis that PPO helped sampled play more than deterministic play. The confidence intervals and opponent consistency matter. This experiment does **not** by itself prove the mechanism or whether PPO increased chess strength generally.
- This differs from the earlier 64-game diagnosis, in which **both players sampled**. Here the V0 opponent is **always greedy**, so don't compare the earlier 68% score directly with this run.

## Output files

```text
outputs\ppo_policy_comparison_manifest.json
outputs\ppo_policy_comparison_games.csv
outputs\ppo_policy_comparison_summary.csv
outputs\ppo_policy_comparison_summary.json
```

The manifest includes SHA-256 hashes for both checkpoints, the seed and all openings. Results are written **after every game**. If the command is interrupted, run `run.cmd ppo-policy-test` again; it skips completed games. It refuses to combine results from different checkpoint files or experiment settings.

Your `v0_supervised.pt`, `ppo_v0100.pt`, PPO optimizer states, previous results, and any Lichess dataset files remain unchanged.
