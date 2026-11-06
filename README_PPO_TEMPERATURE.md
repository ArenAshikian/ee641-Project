# EE 641: Search-free PPO temperature / entropy comparison

This is a **read-only evaluation**. It does not change or retrain V0, V100, the optimizer, or any saved data. No Stockfish, search, engine labels, or board evaluation are used. **Each move is chosen using one neural forward pass, followed by legal-move filtering.**

## Install

1. Extract `EE641_PPO_Temperature_Update.zip` into `B:\EE641_Chess_RL_Starter\`.
2. Keep the `chess_rl` and `tests` subfolders. Replace `run.cmd` when prompted.
3. Do **not** delete `.venv`, `checkpoints`, `data`, or `outputs`.
4. Open CMD in the project directory and run:

```cmd
run.cmd ppo-temperature
```

The command runs the existing test suite plus the new regression tests, then launches the experiment. **Rerun the same command to resume** if interrupted.

## Design

The previous greedy-V0 comparison found **V0 sampled = 20.0%** and **V100 sampled = 38.1%** over 400 games each, while greedy V0 and greedy V100 were near 50%. We now test whether this sampling gain mainly results from PPO concentrating its move probabilities, rather than learning better move preferences.

First, the script collects up to 400 independent board positions from no-search greedy V0-vs-V0 games. On those *calibration positions only*, it measures legal-action entropy for both V0 and V100. It finds the V0 softmax temperature that most closely matches V100's mean entropy at temperature 1.0. **Calibration positions are separate from tournament starting positions.** If target entropy is outside the supported range (T=0.1 to 2.0), the script reports a boundary match instead of claiming an exact match.

Then every candidate condition plays **400 games (200 different starting positions from both colors)** against an unchanged **greedy V0 opponent**:

| Candidate | Sampling temperature | Games |
| --- | ---: | ---: |
| V0 | 1.0 | 400 |
| V0 | 0.7 | 400 |
| V0 | 0.5 | 400 |
| V100 | 1.0 | 400 |
| V100 | 0.7 | 400 |
| V100 | 0.5 | 400 |
| V0 | Matched to V100 T=1.0 entropy | 400 |
| **Total** | | **2,800** |

**Temperature below 1.0** makes sampling more concentrated on the highest-probability legal actions. It never evaluates future positions. The opponent always uses greedy V0 in each game. Color-swapped pairs use the same starting moves, and stochastic candidate games use the same random seeds per starting position and color.

The reported confidence intervals bootstrap whole *opening pairs* and exclude truncated games from the paired comparisons; 240-ply cutoffs are **not** called draws. New openings avoid known previous evaluation and training **starting positions**, but this cannot guarantee that none appeared as intermediate positions in September Lichess games or other matches.

## Primary result to inspect

Look for this line near the end:

```text
v100_t1_minus_v0_matched_entropy: +...; CI=[..., ...]; pairs=...
```

If the difference is near zero after matching legal-move entropy, that supports the **probability sharpening** explanation. If V100 still reliably outperforms the matched-entropy V0, that suggests PPO changed the policy in ways beyond simply making it more concentrated. This result by itself does **not** prove better move rankings or generalization to other opponents.

Other lines compare the models at T=1.0, T=0.7 and T=0.5 and show how each model responds to reduced sampling randomness.

## Files created

```text
outputs\ppo_temperature_manifest.json
outputs\ppo_temperature_calibration.json
outputs\ppo_temperature_games.csv
outputs\ppo_temperature_summary.csv
outputs\ppo_temperature_summary.json
```

The manifest records model SHA-256 hashes, opening positions, calibration settings, and experiment settings. The script saves each game immediately; it will refuse to silently combine data from different settings or different checkpoint files.

Original checkpoints and all previous results remain unchanged. The only two model files needed are:

```text
checkpoints\v0_supervised.pt
checkpoints\ppo_v0100.pt
```

No new packages are required beyond your existing Python environment.
