# EE 641: Fresh independent Maia confirmation

Goal: test whether the original supervised V0 and the PPO-trained Latest S49 differ in chess strength **on a new dataset**. This update only evaluates. It does not train or change any checkpoint, the Lichess bot, the PPO configuration, or the existing Maia tournament output.

## Design frozen before the new matches

- **Pre-specified primary comparison:** `latest_s49 - v0`, measured by chess score against **Maia 1900**. Two-sided paired opening/paired color comparison, 95% paired bootstrap confidence interval, and paired sign-flip permutation p-value. This is the primary test, not whichever comparison looks best afterward.
- **Secondary comparisons:** Latest S49 vs V100 on Maia 1900, and comparisons against Maia 1500. Treat these as supporting/exploratory.
- **Opponents:** Maia 1500 and Maia 1900, running on the **Lc0 policyhead** algorithm, not MCTS or minimax.
- **Our models:** frozen V0, V100, Latest S49. Each chooses its highest-scoring legal move with one neural-network forward pass; no search.
- **Starting positions:** 256 distinct human-game positions, drawn from the **October 2026 Lichess rated standard archive** (not September 2026 training or prior evaluation games), both colors in each matchup. Only games with both players rated 1800–2200, rated blitz/rapid, and no bots are accepted. Exact FENs used in the earlier PPO opening lists and previous Maia tests are excluded. Common chess positions could still have occurred in the 20M-position supervised dataset, so this is an independent game source, not a mathematical guarantee that every position is novel.
- **Total:** 3 models × 2 Maia levels × 256 opening pairs × 2 colors = **3,072 games**. A separate **12-game smoke test** uses 2 different opening pairs that will never be counted in the main experiment.
- **Outcome:** wins = 1, draws = 0.5, losses = 0. Artificial move-limit truncations are **excluded**, not scored as draws. Matched confidence intervals use only complete pairs.

## Install

1. Stop `run.cmd lichess-play` if it is active, so the GPU is available and your online chess clock isn't affected.
2. Unzip `EE641_Maia_October_Confirmation_Update.zip` into `B:\EE641_Chess_RL_Starter\`, allowing `run.cmd` to be replaced. This ZIP adds `chess_rl\maia_confirm.py`, `tests\test_maia_confirm.py`, `README_MAIA_CONFIRM.md`, and updates only the launcher. It does not overwrite your Maia scripts, existing experiments or model files.
3. Keep your existing Lc0 and Maia networks. No Lichess API token is required.

## Commands

From Command Prompt in your project folder:

```
run.cmd maia-confirm-prepare
run.cmd maia-confirm-smoke
run.cmd maia-confirm-eval
```

The first command **streams and decompresses just the beginning of the 31 GB October archive over HTTPS until it finds 258 usable positions**. It does not download or save the full archive. It may take time depending on connection speed and filtering. It freezes the opening list in `data\processed\maia_confirmation_oct2026.json` for reproducibility. If the internet stream is blocked, download October PGN data separately and use `run.cmd maia-confirm-prepare "B:\path\october_games.pgn"` instead, before creating a manifest. A partial PGN containing enough rated human games works; you don't need the full monthly file.

The second command performs six isolated statistical tests followed by 12 live Maia smoke games. **Only start the third command if this succeeds.**

The third command runs 3,072 new games, potentially a lengthy run. It saves results after **every game**. If stopped cleanly or interrupted, rerun exactly the same command to resume. You can read progress at any time with:

```
run.cmd maia-confirm-report
```

## Outputs

- `outputs\maia_confirm_oct2026_manifest.json`: fixed hashes of checkpoints, weights, Lc0, exact test openings and primary hypothesis.
- `outputs\maia_confirm_oct2026_games.jsonl`: each completed game, moves, termination, outcome and elapsed time. Saves after each game.
- `outputs\maia_confirm_oct2026_summary.json`: win/draw/loss scores, paired comparisons, confidence intervals and the pre-specified primary test (p-value only after all 3,072 games finish).
- `outputs\maia_confirm_oct2026_smoke_*`: separate smoke results.

## How to interpret

If the primary difference is positive with a 95% paired confidence interval excluding zero and the two-sided p-value under 0.05, there is evidence that this PPO-trained checkpoint is stronger than V0 **against Maia 1900 on new human openings**. It does not prove improvement against all chess opponents or that every PPO update was beneficial. Repeated PPO training with multiple seeds would strengthen causal claims. Do not choose a different primary hypothesis based on the new results.

The source archive is officially published by Lichess:
https://database.lichess.org/standard/

This evaluation never accesses Stockfish labels or engine evaluations from source PGN (even when they are present in comments): it extracts only actual human moves to construct starting positions. Maia plays through Lc0 policyhead, strictly as a frozen independent test opponent. Neither Maia nor Lc0 participates in training.
