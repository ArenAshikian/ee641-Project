# EE 641 Chess: Frozen-Checkpoint Championship

This is a **search-free evaluation only**. It does not train or modify any model.

## Installation

Extract the ZIP into `B:\EE641_Chess_RL_Starter\`, preserving the `chess_rl` and `tests` subfolders, and replace `run.cmd` when prompted. Keep `checkpoints`, `outputs`, `.venv`, and the dataset.

## Run

```cmd
run.cmd championship
```

The launcher first runs all project tests. Then it loads the frozen checkpoints and plays five matchups:

| Candidate | Opponent | Games |
| --- | --- | ---: |
| V100 | V0 | 400 |
| V100 | V10 | 400 |
| V100 | V30 | 400 |
| V100 | V50 | 400 |
| V30 | V50 | 400 |

Each matchup uses **the same 200 starting positions**, with **one game per color** from each position. All 2,000 games use the one-pass neural policies, no engine, no MCTS/minimax/alpha-beta, no look-ahead, no Stockfish labels. The chess rules library is used for legal moves and terminal conditions only. The players use deterministic legal-masked argmax, not sampled moves. Rules-based checkmate/stalemate/repetition and claimable draws end a game normally. The **240-ply limit is a truncation, not a draw**.

### Independent openings

The championship uses a new fixed seed and 10/12/14-ply random-legal starting positions. It checks and excludes the prior PPO training and fixed-evaluation **starting positions** produced by the original experiment. It does **not** prove that these board positions never occurred anywhere inside an earlier full game or the original Lichess dataset. They are new *evaluation starts*.

### Metrics

- Wins, draws, losses, incomplete/truncated games, and game termination reasons.
- Game score = (wins + 0.5 × draws) / completed games.
- Paired score averages the two color-swapped games for each fully completed opening pair.
- A 95% bootstrap confidence interval is calculated by resampling **opening pairs**, not individual games, so the two colors are kept together. It excludes incomplete pairs with truncation.
- Compare fixed-opponent scores cautiously, particularly when confidence intervals overlap. This is an exploratory five-comparison study and not a proof that the checkpoint with the highest point estimate is definitively strongest.

## Output files

- `outputs\championship_manifest.json`: seed, all 200 positions, model file SHA-256 hashes, and settings.
- `outputs\championship_games.csv`: every game, saved immediately after it finishes.
- `outputs\championship_summary.csv`: one row for each matchup, with scores and intervals.
- `outputs\championship_summary.json`: detailed scores and termination reasons.

The command is **resumable**. If interrupted, run `run.cmd championship` again. It checks model hashes and settings, skips saved games, and continues with missing games. Do not modify the five checkpoint files between restarts or mix results from different experimental settings.

The original archived models, PPO optimizer states, training metrics, and prior evaluation results are not changed.
