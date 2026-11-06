# EE641 Chess RL: pre-V0 evaluation and benchmark

This is an **update** for the existing `B:\EE641_Chess_RL_Starter` folder. It is NOT a replacement project.

## Install and run

1. Extract `EE641_PreV0_Update.zip`.
2. Copy the contents (including the `chess_rl` and `tests` folders) into `B:\EE641_Chess_RL_Starter`. Select **Replace** for the existing `run.cmd`.
3. Open Command Prompt from `B:\EE641_Chess_RL_Starter` and run:

```cmd
run.cmd prep-v0
```

**Don't delete the `.venv`, September archive, processed data, or checkpoints.** This update requires no new Python packages and does not change `requirements.txt`.

## What the command checks

1. All existing tests plus new evaluation tests.
2. A matched **real-data** training-speed benchmark using the already-extracted 1M data shards, comparing 6 residual blocks x 96 channels with 10 blocks x 128 channels. Both use identical real September positions and the same loss, optimizer, and batch size. The benchmark excludes extraction, data-loading overhead, evaluation, and self-play; it is **not** an end-to-end PPO speed estimate.
3. A reproducible tournament: the trained 1M model against the 100k model and a uniform-random legal-move player, **100 games per opponent** (50 opening positions with colors reversed). The openings are deterministic shallow randomized moves, not engine or opening-book labels. Score means win=1, draw=0.5, loss=0; artificial game truncation is not a draw and is excluded from completed-game score. Reports a color-paired bootstrap confidence interval.
4. **Optional Stockfish shallow-search opponent** if a local Stockfish executable is installed and discoverable. Search is **evaluation only** and never influences the neural model's move choice or training labels. If Stockfish is not installed, this opponent is skipped without error. You can set the `STOCKFISH_PATH` environment variable to the executable path before running if desired.

## Outputs

- `outputs/v0_architecture_benchmark.json`
- `outputs/v0_fixed_opponent_games.csv` (each game is appended immediately, so an interrupted run can resume)
- `outputs/v0_fixed_opponent_summary.json`

The existing 3k/100k/1M checkpoints and data are never replaced by these scripts. This run **does not** train the permanent supervised V0 model or PPO.

To run only one part:

```cmd
run.cmd benchmark-v0
run.cmd eval-v0
```

## Notes for the project report

- Current 1M data samples are from an initial stretch of the September file, not an even sample across the full month.
- Whole-game train/validation split reduces direct game leakage, though identical chess openings can still be shared by independent games.
- Win-rate estimates from these shallow randomized openings are diagnostic, not an Elo rating or a complete measure of chess skill.
- Evaluations are independent of training data labels. If using Stockfish it is only an offline, external opponent.
- Keep the logs and model checkpoints as experiment evidence for the EE641 project.
