# EE641 supervised V0 training

This is the **final supervised warm-start stage** for the search-free chess project.
It does not implement PPO yet. It uses no search or engine-generated labels.

## Install and run on Windows

1. Extract this update ZIP and **copy its contents into your existing** `B:\EE641_Chess_RL_Starter\` folder. Replace `run.cmd` when asked. Do not delete `.venv`, previous checkpoints, or the downloaded PGN.
2. Make sure `B:\EE641_Chess_RL_Starter\data\lichess\lichess_db_standard_rated_2026-09.pgn.zst` is present.
3. Open CMD in the project folder and run:

    run.cmd train-v0

This command runs the tests and starts the full 20-million-position extraction and three-epoch training. You can run the same command again after an interruption.

## Settings

- 20,000,000 positions from selected rated 1800-2200 Lichess blitz/rapid games in September 2026.
- 12 positions sampled across the early/middle/late **thirds of each game** (not formal opening/middlegame/endgame labels).
- Whole-game deterministic 90%/10% training/validation split. October remains reserved for a later independent test.
- 10 residual blocks × 128 channels; policy outputs 4,672 logits; value predicts game outcome from side-to-move perspective.
- AdamW with cosine annealing, three epochs, batch size 128, cross entropy + 0.25 × MSE.
- Compact bit-packed board shards (lossless for the existing `encode_board` representation), streamed from the compressed PGN.

**Storage:** Keep about 12 GiB free on B: for processed data and checkpoints. The PGN archive stays compressed. Extraction can take considerably longer than a small pilot. Before starting, consider keeping Windows from sleeping automatically.

**Resume behavior:** Extraction writes `extraction_progress.json` periodically and safely overwrites uncommitted shards after an interruption. Training writes `checkpoints\v0_training_progress.pt` after every completed epoch. A restart resumes at the last completed extraction/training checkpoint, but work done since the last save may have to be repeated. Do not change run settings during a resume.

## Results

- `data\processed\v0_20000000\stats.json`: source and filtering statistics, split, per-third counts.
- `outputs\v0_supervised_metrics.csv`: epoch accuracy/loss, train speed and per-third validation accuracy.
- `checkpoints\v0_supervised.pt`: **permanent V0**, saved only after all epochs finish, using the best validation epoch.
- `checkpoints\v0_training_progress.pt`: temporary state for resumption, removed when V0 finishes.

Do not overwrite `v0_supervised.pt` once PPO starts. Use it as the archived V0 reference. Every ten PPO versions will be archived in the separate PPO stage.

## Limitations and research notes

- This is human-move **behavior cloning**, not a proof of strong chess or RL improvement.
- Human moves are supervision, not ground-truth optimal moves. Game outcomes provide noisy value targets.
- All examples come from a selected early section of September's archive, not an independent uniformly randomized sample of the whole month. Duplicate board positions can appear in different games.
- Loss is calculated on raw policy logits during supervised training, to learn both the move and implicit legality. Inference will mask illegal moves.
- The balanced early/middle/late thirds are relative to game length; actual opening/middlegame/endgame could differ.
- Timing excludes end-to-end PPO self-play. Training and extraction times from previous pilots should not be treated as guarantees.
- Training code is AI-assisted and should be reviewed, tested, and appropriately disclosed under course policy.
