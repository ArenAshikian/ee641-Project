# EE 641: PPO scaling + checkpoint-pool ablation

**Install:** unzip the update over `B:\EE641_Chess_RL_Starter\`, keeping your existing `data`, `.venv`, `checkpoints`, and `outputs`. Replace `run.cmd` when prompted. The update contains only new Python modules/tests and `run.cmd`, plus this README. It leaves V0–V100, old training code, all previous evaluation CSVs, and your large PGN archive untouched.

## Run

```cmd
run.cmd ppo-scale-prepare
run.cmd ppo-scale-smoke
run.cmd ppo-scale
```

`ppo-scale-prepare` streams your **existing** `data\lichess\lichess_db_standard_rated_2026-09.pgn.zst` and saves 8,192 real human-game starting positions from V0's *training-game* split and 256 from the separate *validation-game* split. It uses the same filters and whole-game split as `train-v0`. No extra data download is needed.

`ppo-scale-smoke` runs four real PPO games per branch using V100, but saves its checkpoints and metrics in a disposable temporary folder. **Run this once before the long experiment.**

`ppo-scale` then runs **two independent branches from exactly the same saved V100 weights and optimizer state**:

| | Pool branch | Latest-only branch |
|---|---|---|
| Training command | `run.cmd ppo-scale-pool` | `run.cmd ppo-scale-latest` |
| Initial checkpoint | `checkpoints\ppo_v0100.pt` | same |
| Games/iteration | 512 (256 paired openings) | 512 |
| Iterations | 49 | 49 |
| Added games | 25,088 | 25,088 |
| Opponents | 50% current frozen policy, 50% uniformly selected archived policy | 100% current frozen policy |
| PPO | 2 passes, batch 64, lr 1e-5, epsilon .1 | identical |
| Self-play moves | sampled, legal-masked | sampled, legal-masked |
| Evaluation moves | greedy, legal-masked | greedy, legal-masked |

The historical pool includes **V0, V10, V20, ..., V100** and, after step 10, 20, etc., that branch's own archived models. No search, chess engine, MCTS, minimax, or move labels enter PPO.

The two branches take substantial training time. For a **single branch**, use `run.cmd ppo-scale-pool` first and inspect its results before starting `run.cmd ppo-scale-latest`. `ppo-scale` runs both sequentially. The script saves **after every 512-game iteration**. If interrupted, repeat the same command; only the unfinished iteration is repeated. The step-10/20/30/40/49 snapshots and existing results are never overwritten.

After training both branches, run:

```cmd
run.cmd ppo-scale-eval
```

This completes any missing evaluations and runs direct **pool vs latest** games at matching saved training milestones. It also calculates paired differences in score against the same fixed opponents on the same held-out human openings. Running it again safely skips completed games.

## Evaluation

- Greedy games against frozen V0, V30, V100, and the older 1M supervised checkpoint if available.
- Every ten iterations plus the final step 49, 128 held-out human openings **from both colors** per matchup.
- Paired-bootstrap 95% confidence intervals; move-limit truncations are **not** draws.
- True head-to-head pool vs latest checkpoints at matching steps.
- A fixed V100 baseline can be evaluated on the same human starts.
- A once-off confirmation on **all 256 held-out starts** is possible with `.venv\Scripts\python.exe -m chess_rl.ppo_scale --mode eval --eval-pairs 256`. This creates separate p256 files; it never changes the p128 experiment.

**Important:** September validation games are held out from V0's gradient training but were used to monitor V0 model accuracy, so these are development/validation benchmarks, not a final untouched external test. The proposal's independent Maia benchmark and live Lichess bot study still require an actual external Maia model/interface and Lichess bot games. This update does **not** claim to implement either. A future October PGN set can serve as a stronger independent final opening evaluation.

## Output files

- `data\processed\ppo_scale_human_openings.json`: locked train/eval PGN positions and source identity.
- `checkpoints\ppo_scale_pool\latest.pt` / `...\s0010.pt`, ..., `...\s0049.pt`: isolated pool weights and optimizer state.
- `checkpoints\ppo_scale_latest\latest.pt` / `...\s0010.pt`, ..., `...\s0049.pt`: isolated latest-only weights.
- `outputs\ppo_scale_pool_metrics.csv`, `...latest_metrics.csv`: **sample-weighted averages over all PPO minibatches**, rollout value MSE, value explained variance, KL estimator, entropy, clip fraction, rewards, games, optimizer steps, game lengths.
- `outputs\ppo_scale_pool_opponents.csv`, `...latest_opponents.csv`: per-step counts and W/D/L/truncated **for every archived opponent**.
- `outputs\ppo_scale_eval_<branch>_sNNNN_p128.csv`, `..._summary.json`: resumable fixed-opponent games.
- `outputs\ppo_scale_pool_vs_latest_direct_sNNNN_p128.csv` and summary JSON: true branch head-to-head.
- `outputs\ppo_scale_report_p128.json`, `ppo_scale_pool_vs_latest_p128.csv`, `ppo_scale_direct_match_summary_p128.csv`: experiment summaries.

For performance, the **positive** approximate-KL estimator is `mean[(ratio-1)-log(ratio)]`, which cannot be negative even for individual samples. The signed sample KL is still logged separately. Explained variance from rollout GAE targets is diagnostic, **not** proof of value generalization to unseen games.

## Guardrails

1. The source V100 and all old archives are checked by SHA256 before continuation. Branches refuse to resume if the frozen starting checkpoint, source model archives, PPO settings, random seed, or human opening pool changes.
2. The historical branch always samples with replacement from the entire available historical set, not just the last checkpoint.
3. BatchNorm running statistics remain frozen during PPO, so old and new action log probabilities are comparable.
4. Result files are written after each evaluation game; training checkpoints commit after each iteration.
5. Training uses only game outcomes (+1/0/-1), plus critic bootstrap at artificial move cutoffs. It does not use human move labels or engine scores for PPO.

**Suggested first step:** run `run.cmd ppo-scale-prepare` and check the saved counts. Then begin `run.cmd ppo-scale` or just the pool branch.
