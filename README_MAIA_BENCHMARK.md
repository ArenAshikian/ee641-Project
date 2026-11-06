# EE 641: Independent Maia benchmark (one node, no search)

This is an **evaluation-only** update. None of the Maia weights or Maia moves are used to train your chess bot. The bot still selects each move through one neural-network forward pass, legal-move masking, and greedy argmax. Maia is an **independent opponent** accessed through the official Lc0 UCI engine with `go nodes 1` (the original Maia authors' specified setting).

## Install this update

Extract `EE641_Maia_Benchmark_Update.zip` into `B:\EE641_Chess_RL_Starter\` and replace `run.cmd`. It contains only `chess_rl/maia_benchmark.py`, `chess_rl/maia_stats.py`, a new test file, this README, and the updated `run.cmd`. It **does not overwrite checkpoints or PPO files**.

You need your four existing checkpoints:

- `checkpoints\v0_supervised.pt`
- `checkpoints\ppo_v0100.pt`
- `checkpoints\ppo_scale_pool\s0049.pt`
- `checkpoints\ppo_scale_latest\s0049.pt`

### 1. Get the Maia neural networks

From the root of your project:

```cmd
run.cmd maia-download
```

This downloads three fixed original Maia 1 networks (targeted at 1100, 1500 and 1900 human ratings) **from the official CSSLab release**, placing them under `benchmarks\maia\`. If GitHub fails, manually download and place files with these exact names:

- https://github.com/CSSLab/maia-chess/releases/download/v1.0/maia-1100.pb.gz
- https://github.com/CSSLab/maia-chess/releases/download/v1.0/maia-1500.pb.gz
- https://github.com/CSSLab/maia-chess/releases/download/v1.0/maia-1900.pb.gz

These rating numbers are **training target bands, not measured playing ratings**. Maia is normally stronger than its target-rated humans.

### 2. Install Lc0 for Windows

Download the official Windows Lc0 package:

https://lczero.org/play/download/

Your RTX 3090 supports the **NVIDIA CUDA 12** Windows package. Extract **all of its contents including `lc0.exe` and the required DLL files** into:

`B:\EE641_Chess_RL_Starter\benchmarks\lc0\`

Check that `benchmarks\lc0\lc0.exe` exists. Alternatively, set the `LC0_PATH` environment variable to a full executable path. **Lc0 itself is not included in our ZIP**.

The benchmark runs Lc0 using the selected Maia weights with `go nodes 1` per move. It refuses any engine reply reporting more than one visited node. This is the **original Maia one-node method**, not a multi-depth chess search. Our bot uses no engine, game tree, opening book or search.

If your Lc0 GPU build cannot load the older Maia `.pb.gz` weights, try the official **CPU DNNL BLAS** Windows package and put those files in the same `benchmarks\lc0\` location (replacing the GPU binary and its DLLs). This will be slower but still one-node.

### 3. Freeze new human-game evaluation openings

```cmd
run.cmd maia-prepare
```

This reads your existing 29 GB September Lichess `.pgn.zst` archive, accepts only the *V0 validation-game partition*, and selects 256 **distinct new human-game openings that were not in the PPO-scale evaluation set**. It saves them at `data\processed\maia_openings.json` and never silently replaces them. The test openings are **not PPO training starts**.

**Scientific limitation:** V0 used the September validation partition for model-selection/monitoring. These newly selected games are fresh to PPO evaluation, but this is **not a fully untouched independent dataset**. For a stronger final test, obtain a recent external `.pgn` or `.pgn.zst` of human standard games, and *before your first prepare* run:

```cmd
run.cmd maia-prepare --source "B:\path\to\unseen_human_games.pgn"
```

The benchmark supports ordinary PGN and Lichess `.pgn.zst`. If you use September defaults, label it accordingly in the report. **Never replace the opening file after an evaluation has begun**, since that would invalidate the comparison.

### 4. Smoke test (small real games)

```cmd
run.cmd maia-smoke
```

Runs 16 real games: all four checkpoint models face Maia 1100 from two human starting positions, both colors. It checks the Lc0 integration and produces a resumable diagnostic result. This does not run the large tournament.

### 5. Standard evaluation

```cmd
run.cmd maia-eval
```

4 models × 3 Maia networks × 64 opening pairs × 2 colors = **1,536 games**. Expect longer than the smoke test, depending on Maia/Lc0 inference speed. Restarting the same command resumes every completed game, not whole matchups. All neural checkpoints are read-only.

For an extended independent confirmation after inspecting the standard test:

```cmd
run.cmd maia-eval-full
```

This uses 128 pairs, so it takes **3,072 games** total. It reuses the locked starting-position list, and saves its results separately from the 64-pair experiment. It is optional rather than a prerequisite for Lichess deployment.

## Results

- `outputs\maia_p064_games.jsonl`: every game with outcome, starting position, color, move list, time, termination, Maia node count, append-flushed **after each game**.
- `outputs\maia_p064_manifest.json`: locked hashes of all checkpoints, weights, Lc0, settings and opening positions. Changes cause resume to **fail safely** instead of mixing experiments.
- `outputs\maia_p064_summary.csv` and `_summary.json`: W/D/L/truncations, mean score, paired-bootstrap confidence intervals and differences versus frozen V0/V100.
- `maia_p002_*` is the separate smoke run. `maia_p128_*` is the extended evaluation.

Artificial move limits are **truncations**, not draws. Paired-bootstrap score intervals exclude opening pairs with incomplete games. Statistical comparison uses the **same opening and both colors** for all four candidates.

Do not interpret this as a public Elo rating. These are controlled match scores against fixed Maia 1 models. A public rating requires the later Lichess Bot API phase. The experiment's central comparison is whether PPO improves a greedy, single-pass chess policy against opponents that were never part of self-play training.

## Original references

- Maia project: https://github.com/CSSLab/maia-chess
- Maia one-node instructions: https://github.com/CSSLab/maia-chess#how-to-run-maia
- Lc0 Windows downloads: https://lczero.org/play/download/
- Lc0 policyhead search documentation: https://lczero.org/dev/overview/
