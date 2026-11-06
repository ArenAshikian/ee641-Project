# EE 641 Chess RL: 100,000-position supervised experiment

This is an **update** to the `EE641_Chess_RL_Starter` project, not a replacement.
It keeps your September Lichess archive, CUDA-enabled PyTorch environment,
prior tests, and 3,000-position pilot results.

## Install

1. Extract `EE641_100k_Update.zip`.
2. Copy the **contents** into `B:\EE641_Chess_RL_Starter\`.
3. Choose **Replace files** if Windows asks for `run.cmd` and `chess_rl\lichess_data.py`.
4. From CMD in `B:\EE641_Chess_RL_Starter\`, run:

```cmd
run.cmd train-100k
```

## What happens

- Reads `data\lichess\lichess_db_standard_rated_2026-09.pgn.zst` in place.
- Filters rated standard blitz/rapid games with both players rated 1800-2200.
- Excludes BOT players, time forfeits/abandonments, and moves with low post-move clock.
- Samples 100,000 positions across whole games, splitting train/validation
  deterministically by game ID. No engine evaluation labels are used.
- Writes small compressed shards under `data\processed\supervised_100000\`.
- Trains a 4-block x 64-channel ResNet policy/value model for 3 epochs.
- Reports random-legal and most-common-action validation baselines, human-move
  accuracy, loss, and actual throughput on your computer.

## Outputs

- `data\processed\supervised_100000\stats.json`
- `data\processed\supervised_100000\train_*.npz` and `val_*.npz`
- `outputs\supervised_100000_metrics.csv`
- `checkpoints\supervised_100000.pt`

This **does not** overwrite `supervised_pilot.pt`, and is not the final V0.
You can rerun the command; it reuses the sharded dataset if the settings match.

## Research limitations

- This is still a sample from the **beginning** of September, not a uniform
  sample of the entire month. Final training should sample more broadly.
- September validation uses a game-level split. October will be reserved for
  independent evaluation.
- Human moves are behavior-cloning targets, not optimal actions.
- Accuracy is evaluated on *unmasked* policy logits to match the initial
  supervised objective. The uniform-random-legal baseline is a reference only.
- The 4x64 model is a throughput/learning pilot; the proposal's 10x128 model
  will require a separate compute feasibility test.
- The scaffold and this update were prepared with AI assistance. Review and
  understand the code and disclose assistance per your course requirements.
