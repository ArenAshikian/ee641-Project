# EE 641: Search-Free Chess Reinforcement Learning

A chess-playing neural network trained first on human games and then improved through self-play reinforcement learning. This is a course research project for **USC EE 641, A Computational Introduction to Deep Learning (Fall 2026)**.

The main experiment compares two ways to train a chess agent with Proximal Policy Optimization (PPO):

1. **Current-only self-play:** play against a frozen snapshot of the latest model.
2. **Historical-opponent self-play:** play against the current snapshot, recent archived models, and older archived models.

**Core constraint:** Our agent chooses moves using a single neural-network forward pass followed by a legal-move mask. It does **not** use minimax, alpha-beta pruning, Monte Carlo tree search, Stockfish, an opening book, or engine evaluations to select moves during training or deployment. `python-chess` handles the chess rules. External chess engines/models are used only as evaluation opponents or baselines.

> **Project status (November 2026):** Supervised pretraining, initial PPO, current-only PPO continuation, historical-opponent PPO, Maia evaluation, and a Lichess bot interface are implemented. The longer historical run is an ongoing experiment. Not every training artifact is included in Git because datasets and checkpoints are large.

## Contents

- [Research question](#research-question)
- [How the model works](#how-the-model-works)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Quick start: code and tests](#quick-start-code-and-tests)
- [Datasets and pretrained checkpoints](#datasets-and-pretrained-checkpoints)
- [Training experiments](#training-experiments)
- [Evaluation and saved results](#evaluation-and-saved-results)
- [Playing on Lichess](#playing-on-lichess)
- [Reproducibility and known limitations](#reproducibility-and-known-limitations)
- [Common problems](#common-problems)

## Research question

Can a chess policy improve through **self-play PPO without search at move time**, and does training against a pool of historical checkpoints improve its general playing strength compared with training against only the latest model?

We investigate this using:

- An initial supervised model trained to predict moves from filtered human Lichess games.
- Sparse terminal rewards in PPO: **+1** for a win, **0** for a draw, **-1** for a loss.
- Frozen self-play opponents, so the opponent network does not change during a collected rollout.
- Matched training-game budgets and paired-color evaluations.
- Fixed external opponents, including Maia, to measure strength separately from the changing training opponents.

Training win rate against the historical pool is **not** a direct Elo rating. The pool changes as training advances. External evaluations are necessary to tell whether the model is actually improving.

## How the model works

### Inputs and outputs

| Component | Implementation |
| --- | --- |
| Board representation | **61 × 8 × 8** planes, including position history, repetition information, castling rights, and half-move clock |
| Network | Convolutional residual network with **10 residual blocks**, **128 channels** for the main model |
| Policy output | **4,672 actions** (64 source squares × 73 move types) |
| Value output | One scalar in **[-1, 1]** |
| Legal moves | Enumerated and masked using `python-chess` |
| Deployment/evaluation move choice | Highest-scoring legal move from one forward pass |
| Training | PyTorch PPO, using sampled legal moves and a value estimate |

The model cannot choose illegal moves because actions outside the legal-move mask are excluded. Chess rules, checkmate, stalemate, repetition, and other terminal conditions are handled by the environment.

### Learning stages

**Phase 1: supervised V0.** The main supervised run uses up to 20 million board/move examples from September 2026 Lichess rated blitz/rapid games, filtered to human opponents rated 1800–2200. Training uses human moves, not Stockfish labels or PGN engine evaluations. The saved network is `checkpoints/v0_supervised.pt`.

**Phase 2: initial PPO.** V0 is fine-tuned using full-game self-play and frozen historical opponents. The original V100 checkpoint represents **11,840 cumulative PPO games** and is the common starting point for our main comparison.

**Phase 3: scaling comparison.** One experiment continues with the latest frozen model as its opponent. Another continues from the *same original V100* with a mixture of current, recent, and older opponent checkpoints. These branches are independent and save into different directories.

## Repository structure

```text
.
├── chess_rl/                  # Game encoding, models, training, evaluations, Lichess bot
│   ├── encoding.py            # 61-plane input, legal 4,672-action representation
│   ├── environment.py         # Chess environment and game termination
│   ├── network.py             # Residual policy/value network
│   ├── train_v0.py            # Full supervised V0 training
│   ├── ppo_train.py           # Initial PPO
│   ├── ppo_long.py            # Original V10–V100 training
│   ├── ppo_scale.py           # Original pool versus latest-only experiment
│   ├── ppo_batch_continue.py  # Batched latest-only continuation
│   ├── ppo_historical_continue.py # Historical pool continuation from V100
│   ├── maia_benchmark.py      # Maia/Lc0 evaluation
│   ├── maia_confirm.py        # Separate held-out October evaluation
│   └── lichess_live.py        # Live Lichess BOT API interface
├── tests/                     # Unit and integration tests
├── outputs/                   # Tracked CSV/JSON summaries, other generated results
├── data/                      # Local datasets and opening manifests (not in Git)
├── checkpoints/               # Model/optimizer states (not in Git)
├── benchmarks/                # Local Maia weights and Lc0 executable (not in Git)
├── requirements.txt
├── run.cmd                    # Windows command launcher
└── README.md
```

Additional modules support training diagnostics, milestone evaluation, model tournaments, datasets, and other experiments. `run.cmd` is the command entry point for Windows.

## Requirements

- **Windows 10/11** for the supplied `run.cmd` launcher. The Python modules may be usable on other systems, but a macOS/Linux setup script is not supplied yet.
- **Python 3.11 (64-bit) recommended**, plus Git and an internet connection for initial installation.
- Dependencies in `requirements.txt`, including PyTorch, NumPy, `python-chess`, and `zstandard`.
- A **CUDA-capable NVIDIA GPU** strongly recommended for training. The full V0 training script explicitly requires CUDA. Long PPO experiments were developed and run on an **RTX 3090 (24 GB VRAM)**; performance and memory use may differ on another GPU.
- Enough free storage for datasets and archived checkpoints. The local Lichess archive is approximately **28 GB** in our current setup and is intentionally excluded from Git.

A CPU can run some basic tests and inference, but it is not a practical substitute for the GPU when reproducing large training runs. The first-run dependency installer does not guarantee that the best CUDA-enabled PyTorch build is selected for every machine; check PyTorch's official installation instructions if CUDA is unavailable.

## Quick start: code and tests

From **Windows Command Prompt**:

```cmd
git clone https://github.com/ArenAshikian/ee641-Project.git
cd ee641-Project
run.cmd tests
```

`run.cmd` automatically creates `.venv` if needed and installs `requirements.txt` on its first setup. The initial installation requires internet access.

You can also run the starter model check:

```cmd
run.cmd
```

With no argument, the launcher runs the tests and a small untrained model shape/legality check. This does **not** start the large training job.

Check CUDA using the Python inside the project environment:

```cmd
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

**Clean-clone boundary:** Basic source-code checks should not require the 28 GB PGN archive, but training, checkpoint-based smoke tests, and external benchmarks require extra files described below. We have **not yet independently validated a clean clone on a second PC**.

## Datasets and pretrained checkpoints

### September Lichess data

The default PGN path expected by the scripts is:

```text
data/lichess/lichess_db_standard_rated_2026-09.pgn.zst
```

Obtain the September 2026 standard rated-game archive from the [Lichess public database](https://database.lichess.org/), and place the `.pgn.zst` file at that path. Do not extract it unless a script asks for extraction; the training code reads the compressed archive directly.

To generate the supervised V0 from scratch (a substantial GPU and disk task):

```cmd
run.cmd train-v0
```

The V0 trainer streams up to 20 million examples, produces compact processed shards under `data/processed/`, and saves `checkpoints/v0_supervised.pt`. It is **not necessary** to repeat this run if your goal is to evaluate or continue a supplied checkpoint.

### Saved human opening positions

Later PPO experiments depend on the exact saved opening manifest:

```text
data/processed/ppo_scale_human_openings.json
```

It contains **8,192 training starts** and **256 held-out evaluation starts**, obtained from separate filtered human-game partitions. The saved move sequences preserve history for the input planes.

To prepare it from the original dataset when starting a **new** experiment:

```cmd
run.cmd ppo-scale-prepare
```

**Do not regenerate or edit this manifest when resuming an existing experiment.** Checkpoints validate its content hash. For a teammate continuing this project's existing experiments, obtain an **unchanged copy of the original manifest** together with the checkpoints. Rebuilding it on another PC can produce a different source identity or hash, even with the same PGN data.

Other evaluation workflows may use separate frozen manifests under `data/processed/`, including the October 2026 Maia confirmation openings. These are also excluded from Git and should be preserved alongside the experiments they belong to.

### Checkpoint distribution

Large `.pt` checkpoints are ignored by Git. We plan to distribute selected reproducibility assets through **GitHub Releases or another shared artifact location**. **No checkpoint download URL is provided here until those assets have actually been uploaded.**

Keep the original relative filenames and directory structure when copying/extracting them:

| Checkpoint or asset | Why it matters |
| --- | --- |
| `checkpoints/v0_supervised.pt` | Supervised starting model |
| `checkpoints/ppo_v0010.pt` through `ppo_v0100.pt` (every ten) | Original historical opponents and V100 starting point |
| `checkpoints/ppo_scale_latest/s0049.pt` | 36,928-game latest-only model, also used by the current Lichess bot |
| `checkpoints/ppo_batch_continue/s0026.pt`, `s0075.pt`, `s0124.pt` | Latest-only 50K, 75K, and 100K milestones |
| `checkpoints/ppo_historical_v100/sXXXX.pt` | Historical-pool milestones and archived ancestors |
| `checkpoints/ppo_historical_v100/latest.pt` | Resumable historical progress, including optimizer state |
| `data/processed/ppo_scale_human_openings.json` | Required fixed training openings for the comparison |

If a teammate is **resuming historical PPO**, they must receive the original V0/V10/.../V100 files, the original opening manifest, and the historical branch's `latest.pt` **plus the archived historical checkpoints needed by that step**. Sending only `latest.pt` is insufficient because the resumed learner samples earlier models as opponents.

Checkpoint verification intentionally rejects changes to source checkpoints, model configuration, opening manifests, or archived history. Keep copies of checkpoints you intend to use for comparisons. Do not mix files from different experimental branches.

## Training experiments

All commands below run from the repository root using Windows CMD. Training can be interrupted and restarted from its last successfully saved iteration; verify the saved state in the logs before changing anything.

### Historical-opponent PPO (main experiment)

```cmd
run.cmd ppo-historical-smoke
run.cmd ppo-historical-600k
```

The smoke test checks the original source checkpoint, runs focused tests, collects eight short games, and updates model weights **in memory only**. It does not save smoke-test checkpoints or output metrics.

The full run starts at original V100 (**11,840 cumulative PPO games**) and uses **512 games per new iteration**:

| Opponent category | Games per iteration | Share |
| --- | ---: | ---: |
| Frozen current learner | 128 | 25% |
| Five most recent archived ancestors | 256 | 50% |
| Sampled older archived ancestors | 128 | 25% |

Each human opening is played with both learner colors against the same selected opponent. Only the learner's experience updates the policy. PPO uses a 240-ply cutoff, two optimization epochs per iteration, minibatches of 64, and a learning rate of `1e-5` in this branch.

**Important:** The historical branch starts from **original V100**, *not* from the later 100K latest-only model. It never samples future latest-only branch checkpoints as historical opponents.

| Historical step | Cumulative PPO games | Command to reach it |
| ---: | ---: | --- |
| 49 | 36,928 | `run.cmd ppo-historical-37k` |
| 75 | 50,240 | `run.cmd ppo-historical-50k` |
| 124 | 75,328 | `run.cmd ppo-historical-75k` |
| 173 | 100,416 | `run.cmd ppo-historical-100k` |
| 367 | 199,744 | `run.cmd ppo-historical-200k` |
| 563 | 300,096 | `run.cmd ppo-historical-300k` |
| 758 | 399,936 | `run.cmd ppo-historical-400k` |
| 953 | 499,776 | `run.cmd ppo-historical-500k` |
| 1,149 | 600,128 | `run.cmd ppo-historical-600k` |

The trainer saves:

- `checkpoints/ppo_historical_v100/latest.pt` after every completed iteration.
- Permanent branch archives every ten iterations **plus** selected matched-budget milestones.
- `outputs/ppo_historical_v100_metrics.csv` with iteration-level metrics.
- `outputs/ppo_historical_v100_opponents.csv` with opponent-level results.

Report the most recently saved iteration:

```cmd
run.cmd ppo-historical-report
```

Re-running the same training target resumes when its checkpoint/configuration is consistent. For a longer target there is also `run.cmd ppo-historical-1m`. **Avoid running two GPU training jobs at the same time on the same GPU.**

### Latest-only PPO (comparison branch)

The reference latest-only model is:

```text
checkpoints/ppo_scale_latest/s0049.pt
```

It also starts from original V100 and reaches **36,928 cumulative PPO games**, but trains against the latest frozen model rather than the historical pool. Its batched continuation begins from S49.

```cmd
run.cmd ppo-continue-smoke
run.cmd ppo-continue-50k
run.cmd ppo-continue-75k
run.cmd ppo-continue-100k
run.cmd ppo-continue-600k
run.cmd ppo-continue-report
```

The branches use separate directories, so running either command does not intentionally overwrite the other branch's checkpoints. The latest-only branch and historical branch do not have perfectly matched rollout implementations in all early iterations; see the reproducibility notes below.

### Other research commands

| Command | Purpose |
| --- | --- |
| `run.cmd ppo-batch-smoke` | Test batched PPO rollout behavior |
| `run.cmd ppo-batch-bench` | Benchmark batched versus serial rollout |
| `run.cmd ppo-scale-report` | Report original pool/latest PPO scaling comparison |
| `run.cmd ppo-diagnose` | Analyze self-play PPO behavior |
| `run.cmd ppo-policy-test` | Compare policy behavior under greedy versus sampled moves |
| `run.cmd ppo-temperature` | Explore sampling temperature and policy entropy |
| `run.cmd championship` | Frozen neural-checkpoint tournament without search |
| `run.cmd research-snapshot` | Create a local research snapshot |

Many of these commands require particular already-trained checkpoints. The basic `run.cmd tests` command is the first step after cloning.

## Evaluation and saved results

### Maia benchmarks

Maia is an external neural chess opponent. We use Maia through **Lc0 in policyhead mode**, which chooses moves from its policy without tree search. It is an evaluation opponent, **not** a reward source or a teacher during PPO.

1. Download the Maia weights:

   ```cmd
   run.cmd maia-download
   ```

2. Install a compatible Windows `lc0.exe` and its required DLLs under `benchmarks/lc0/`, or set `LC0_PATH` to the installed executable. The evaluation code creates/uses `lc0-policyhead.exe` so that Maia is run without MCTS.
3. Prepare benchmark starts (requires the September archive):

   ```cmd
   run.cmd maia-prepare
   ```

4. Run a small validation, then an evaluation:

   ```cmd
   run.cmd maia-smoke
   run.cmd maia-eval
   ```

The independent October 2026 confirmation uses separate held-out data:

```cmd
run.cmd maia-confirm-prepare
run.cmd maia-confirm-smoke
run.cmd maia-confirm-eval
run.cmd maia-confirm-report
```

The milestone scripts compare additional latest-only checkpoints on the saved evaluation setup:

```cmd
run.cmd ppo-milestone-smoke
run.cmd ppo-milestone-eval
run.cmd ppo-milestone-report
```

### Current findings

In a prior **512-game per-model** evaluation against Maia1900, the observed game scores were:

| Model | Wins | Draws | Losses | Score |
| --- | ---: | ---: | ---: | ---: |
| Supervised V0 | 133 | 153 | 226 | 40.92% |
| Latest-only S49 (36,928 games) | 166 | 137 | 209 | 45.80% |
| Latest-only ~50K | 181 | 149 | 181 | 50.00%* |
| Latest-only ~75K | 189 | 139 | 184 | 50.49% |
| Latest-only ~100K | 200 | 137 | 175 | 52.44% |

\* The ~50K evaluation also contained one truncated game. The reported 50.00% is the score across completed games. These results describe the **latest-only** learning curve and are not results for the ongoing historical branch.

These results indicate improvement relative to V0 in this evaluation, but differences between adjacent milestones are not necessarily statistically significant. Evaluation positions, paired games, confidence intervals, and checkpoint provenance matter when interpreting these numbers.

### Tracked output files

The repository includes selected `outputs/*.csv` and `outputs/*.json` metrics and summaries. The current `.gitignore` excludes datasets, model weights, and raw `.jsonl` game traces by default. This keeps the normal Git checkout small but means raw game-level replay files may need to be shared separately for some analyses.

For the historical branch, start with:

```text
outputs/ppo_historical_v100_metrics.csv
outputs/ppo_historical_v100_opponents.csv
```

Compare checkpoints at **the same cumulative game count** and evaluate both against the **same fixed opponent/openings**, not just against their respective training opponents.

## Playing on Lichess

`lichess_live.py` currently serves the frozen **latest-only S49** model. **It does not load the newest historical checkpoint automatically.**

First check the local move selection with:

```cmd
run.cmd lichess-selftest
```

For online play, create a dedicated Lichess bot account and a valid personal API token with the required BOT/play permissions. **Do not upgrade a normal personal account:** the bot upgrade is irreversible and intended for a new, eligible zero-game account.

In one Command Prompt window:

```cmd
set "LICHESS_BOT_TOKEN=REPLACE_WITH_YOUR_TOKEN"
run.cmd lichess-check
run.cmd lichess-play
```

The bot accepts **standard casual 3+2** challenges by default and handles one game at a time. Optionally restrict challengers:

```cmd
set "LICHESS_ALLOWED_CHALLENGER=lichess_username"
```

Do not commit or paste an actual API token into the repository, logs, or support messages. An HTTP 401 with `No such token` indicates Lichess did not accept the credential.

If using a separate training window at the same time, account for shared GPU memory and utilization. The Lichess integration runs inference; it does not train or update the selected checkpoint.

## Reproducibility and known limitations

- **No automatic checkpoint installer yet.** A source-only clone does not include training state or the opening manifest. Selected checkpoints and matching artifacts must be published/shared separately before full cross-machine reproduction is possible.
- **Preserve original experiment assets.** The historical trainer checks checkpoint hashes, the V100 optimizer/history, archived opponents, and the saved opening-manifest checksum. Copy the *exact* manifest, not a newly generated approximation.
- **Machine-specific dataset identity.** The opening-preparation script stores the source PGN path, file size, and modification time in its manifest. If a teammate regenerates a manifest on a new machine, it may differ even if the source games are similar. For continuing the same experiment, transfer the original manifest unchanged.
- **Training requires substantial compute.** CUDA performance, available VRAM, disk space, and opponent cache overhead can affect memory requirements and runtime. We have not verified the long run on a second PC.
- **Not a perfectly controlled ablation yet.** The original latest-only experiment used serial rollouts for its first 49 steps; the new V100 historical branch uses batched rollouts immediately. Although both start from the same V100 model and optimizer state, implementation/RNG differences can affect outcomes. A stronger study would repeat both methods with matched rollout code and multiple random seeds.
- **Sparse and truncated rewards.** Win/draw/loss is the learning signal. A 240-ply cap uses bootstrapping for unfinished games; increasing draw or truncation rates should be investigated.
- **Performance must be measured externally.** A higher win rate against changing archived opponents does not prove higher chess strength. Maia results, direct branch matches, and fixed test openings are the meaningful comparisons.
- **Windows-first tooling.** `run.cmd` targets Windows. A teammate using Linux/macOS will need to run the Python modules directly and adapt paths/environment setup.
- **Training outputs change.** Git commits capture metrics at the time of staging. New rows written during ongoing training are not automatically uploaded.

## Common problems

| Error / symptom | Likely cause / next action |
| --- | --- |
| `ModuleNotFoundError` | Activate/use `.venv`, or rerun the launcher's dependency installation |
| CUDA is unavailable | Confirm an NVIDIA GPU, driver, and CUDA-enabled PyTorch installation |
| `Missing original checkpoints` | Obtain the matching V0 and V10–V100 checkpoint files |
| `Human opening manifest missing` | Obtain the exact original manifest, or prepare a **new** one only when beginning an independent experiment |
| Manifest / SHA256 / configuration mismatch | Do not bypass the check; verify that all checkpoints and manifests belong to the same run |
| Maia weights missing | Run `run.cmd maia-download` |
| Lc0 executable missing | Install the Lc0 executable/DLLs, or configure `LC0_PATH` |
| Lichess HTTP 401 `No such token` | Supply a fresh, valid `LICHESS_BOT_TOKEN` in the **same CMD window** |
| Training appears to have stopped | Look for the latest `Saved ... step` line; restart the same resumable command after resolving the error |
| GPU out-of-memory | Close other GPU workloads and check model/cache usage before changing batch sizes (changing settings mid-run may invalidate resumability) |

## Repository and project notes

Repository: [ArenAshikian/ee641-Project](https://github.com/ArenAshikian/ee641-Project)

This repository is intended to make the implementation and experimental evidence available to project teammates. Large research assets are kept out of normal Git history, while source code, tests, configurations, and compact metrics are versioned.

*The README documents the implementation snapshot as of November 6, 2026. It does not claim that the 600K historical run has finished or that the code has already been validated on every teammate's machine.*
