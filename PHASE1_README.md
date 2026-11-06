# Phase 1 update: Lichess supervised pilot

This update goes **inside your existing** `B:\EE641_Chess_RL_Starter` folder.
Do not delete your existing folder, `.venv`, or the 29 GB September archive.

## Run (one command)

Open CMD in `B:\EE641_Chess_RL_Starter` and type:

```cmd
run.cmd train
```

The program will:

1. Install the new `zstandard` dependency into the **existing** virtual environment.
   Your CUDA-enabled PyTorch is preserved.
2. Read `data\lichess\lichess_db_standard_rated_2026-09.pgn.zst`
   **without decompressing the whole archive**.
3. Filter rated standard blitz/rapid games with both players rated 1800-2200.
   Exclude BOT accounts, missing results, time forfeits and abandoned games.
4. Sample up to 3,000 positions and split by whole game into 90% training
   and 10% validation (deterministic game-ID hash). Only the September archive
   is used; October will provide the final *independent* test set later.
5. Train a **small 2-block x 32-channel network for 2 epochs** as a quick
   correctness pilot. This is not the final 10-block x 128-channel V0.
6. Save these outputs:

   - `data\processed\pilot_train.npz`
   - `data\processed\pilot_val.npz`
   - `data\processed\pilot_stats.json`
   - `outputs\supervised_pilot_metrics.csv`
   - `checkpoints\supervised_pilot.pt`

## Commands

- `run.cmd train` runs the pilot; subsequent runs reuse extracted pilot data.
- `run.cmd tests` runs the starter + new dataset unit tests.
- `run.cmd` runs tests and checks the neural network shape/device.
- `run.cmd demo` runs the original random-move demonstration.

## Important limitations of the pilot

- The model predicts **human moves**, not optimal moves. This is supervised
  initialization only, not PPO, and the tiny pilot should not be rated as a
  chess-playing engine.
- We do **not** use Stockfish `%eval` annotations as labels.
- `[%clk]` gives remaining time **after** the move. We exclude positions with
  less than 30 seconds remaining after the move as a conservative proxy.
- Time forfeits and abandoned games are excluded because board value does not
  necessarily correspond to their recorded result.
- The actual proposed dataset is around 20 million positions. Loading these
  all into memory, as this *small pilot* does, would be wasteful. We will add
  disk sharding and benchmark throughput before the real V0 training.
- The pilot is always saved as `supervised_pilot.pt`. **V0 remains reserved**
  for the final supervised model; V10, V20, etc., for PPO archives.
- The synthetic unit tests and initial scaffold were written with assistance
  from ChatGPT. Review and understand the code and disclose assistance as
  required by EE 641.
