# EE 641 chess RL: 1-million-position update

This update is for the existing `B:\EE641_Chess_RL_Starter` folder.
It **does not replace** the September Lichess data, Python virtual environment,
100k checkpoint, or other previous results.

## Install and run

1. Extract the update ZIP.
2. Copy its **contents** into `B:\EE641_Chess_RL_Starter\`.
3. Choose **Replace** for `run.cmd` (the only existing file changed).
4. Open CMD in the project folder and run:

```cmd
run.cmd train-1m
```

The command runs existing and new tests, checks the RTX 3090 CUDA installation,
benchmarks the proposed 10-block x 128-channel ResNet with synthetic data,
extracts 1,000,000 human positions from September PGN, trains a 6-block x
96-channel policy/value model for 3 epochs, and runs small local diagnostic
matches versus a random legal mover and the previous supervised 100k model.

No new dependencies are required. The existing `requirements.txt` is unchanged.
The Lichess `.pgn.zst` file stays compressed. No chess engine is used for
training or move selection.

## Full-game sampling and experiment design

The earlier 100k loader ALREADY sampled randomly throughout games. This update
ensures coverage of **early, middle, and late thirds** of each game's timeline:
up to 4 random eligible moves per third when using 12 moves per game.
These are **relative game thirds**; a long game can have a middle third that
contains what a chess player would call an endgame. The data summary and epoch
CSV report counts and accuracy for each third, separately.

We filter standard rated blitz/rapid games with both ratings in 1800-2200,
exclude registered bot games, time forfeits and abandonments, and skip moves
with recorded post-move clock below 30 seconds. We ignore `%eval` annotations.
A deterministic game-ID split keeps September validation separate from training.
The **October archive will remain an independent final test**.

This experiment samples from the beginning of the September archive, not a
uniform sample over the whole month. The held-out human move is not necessarily
the optimal chess move. More games and a distinct-month test will be necessary
before interpreting any improvement as general chess strength.

## Output files

- `data\processed\supervised_1000000_balanced\stats.json`
- `data\processed\supervised_1000000_balanced\train_*.npz` / `val_*.npz`
- `outputs\supervised_1000000_balanced_metrics.csv`
- `outputs\supervised_1000000_balanced_matches.json`
- `outputs\supervised_1000000_target_arch_benchmark.json`
- `checkpoints\supervised_1000000_balanced.pt`

The saved model is **still a pilot**, not V0. It does not overwrite the 3k or
100k checkpoints. Re-running the command skips finished training if the 1M
checkpoint exists; the dataset is reused if extracted and the settings match.

The small post-training matches include 4 starting positions played from both
colors against each opponent. Outcomes, including uncompleted 300-ply episodes,
are printed separately. These games are **diagnostics**, not statistically
reliable Elo measurements. Actual V0 will require a larger frozen benchmark
suite and a larger dataset.

## Course reproducibility

Data filtering, whole-game splitting, data proportions, training settings,
GPU throughput, model architecture, baseline accuracies, and pilot match
outcomes are recorded. Review and understand the AI-assisted code and report
assistance according to EE 641's course rules.
