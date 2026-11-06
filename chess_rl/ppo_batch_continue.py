"""Isolated, resumable, search-free PPO continuation from frozen Latest S49.

Only the latest-opponent branch is continued. The opponent is a frozen model
snapshot each iteration. Batching changes rollout throughput and RNG ordering,
not rewards, the PPO objective, architecture, or the one-inference-per-move rule.
"""
import argparse
import copy
import json
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone

import chess
import numpy as np
import torch

from chess_rl.human_openings import MANIFEST, training_openings, validate_manifest
from chess_rl.ppo_batch import rollout_latest_batched
from chess_rl.ppo_scale import load_model_and_optimizer
from chess_rl.ppo_scale_metrics import detailed_ppo_update
from chess_rl.ppo_train import CHECKPOINTS, OUTPUTS, atomic_save, sha256
from chess_rl.ppo_long import safe_csv

SOURCE = CHECKPOINTS / 'ppo_scale_latest' / 's0049.pt'
SOURCE_SHA256 = '75bec742766a4c9709211ade12719064bc0eb6405a4931314cd7d2178f04a748'
DIR = CHECKPOINTS / 'ppo_batch_continue'
PROGRESS = DIR / 'latest.pt'
METRICS = OUTPUTS / 'ppo_batch_continue_metrics.csv'
GAMES = 512
START_GAMES = 11840 + 49 * 512
MAX_PLIES = 240
ROLLOUT_BATCH = 32
PPO_EPOCHS = 2
PPO_BATCH = 64
LR = 1e-5
SEED = 6412028  # Independent deterministic continuation seed.
# Permanent, read-only archives at each research milestone.
# Original 50K/75K/100K archives remain unchanged.
MILESTONES = {
    26: 50240,
    75: 75328,
    124: 100416,
    221: 150080,
    318: 199744,
    514: 300096,
    709: 399936,
    904: 499776,
    1100: 600128,
    1393: 750144,
    1881: 1000000,
}


def archive_path(step):
    return DIR / f's{step:04d}.pt'


def total_games(step):
    return START_GAMES + GAMES * step


def base_checkpoint():
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Frozen S49 checkpoint missing: {SOURCE}')
    current_hash = sha256(SOURCE)
    if current_hash != SOURCE_SHA256:
        raise RuntimeError(f'Frozen S49 SHA256 mismatch: {current_hash}')
    saved = torch.load(SOURCE, map_location='cpu', weights_only=True)
    if int(saved.get('step', -1)) != 49 or int(saved.get('iteration', -1)) != 149:
        raise RuntimeError('Source must be Latest S49 trained from V100.')
    if not {'model_state_dict', 'optimizer_state_dict', 'model_channels', 'model_blocks'} <= saved.keys():
        raise RuntimeError('Source lacks original model or optimizer data.')
    old = saved.get('config', {})
    expected = {'branch': 'latest', 'games_per_iteration': GAMES, 'max_plies': MAX_PLIES,
                'ppo_epochs': PPO_EPOCHS, 'batch_size': PPO_BATCH, 'learning_rate': LR}
    for name, value in expected.items():
        if old.get(name) != value:
            raise RuntimeError(f'S49 source setting {name} differs from expected: {old.get(name)}')
    return saved


def load_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f'September PPO training opening manifest missing: {MANIFEST}')
    doc = json.loads(MANIFEST.read_text(encoding='utf-8'))
    return validate_manifest(doc)


def settings(doc):
    return {
        'schema': 1,
        'frozen_source': 'checkpoints/ppo_scale_latest/s0049.pt',
        'source_sha256': SOURCE_SHA256,
        'parent_cumulative_games': START_GAMES,
        'training_opening_manifest_sha256': doc['sha256'],
        'games_per_iteration': GAMES,
        'max_plies': MAX_PLIES,
        'rollout_batch_size': ROLLOUT_BATCH,
        'ppo_epochs': PPO_EPOCHS,
        'ppo_minibatch_size': PPO_BATCH,
        'learning_rate': LR,
        'seed': SEED,
        'opponent': 'frozen latest snapshot from start of each iteration',
        'reward': '+1 win / 0 draw / -1 loss',
        'gamma': 1.0, 'gae_lambda': 0.95, 'clip': 0.1,
        'entropy_weight': 0.01, 'value_weight': 0.5,
        'search': False,
        'note': 'New independent PPO continuation, not bit-identical to sequential trajectory RNG.'
    }


def summarize_game(totals, samples, result):
    totals['games'] += 1
    totals['learner_decisions'] += len(samples)
    if result['truncated']:
        totals['truncated'] += 1
    else:
        outcome = result['outcome_for_learner']
        totals['wins' if outcome > 0 else 'losses' if outcome < 0 else 'draws'] += 1


def one_iteration(model, optimizer, manifest, step, device, games=GAMES,
                  max_plies=MAX_PLIES, rollout_batch=ROLLOUT_BATCH,
                  epochs=PPO_EPOCHS, minibatch=PPO_BATCH):
    if games < 2 or games % 2:
        raise ValueError('Self-play games must be even and at least 2.')
    t0 = time.perf_counter()
    torch.manual_seed(SEED + step * 167)
    random.seed(SEED + step * 9127)
    snapshot = copy.deepcopy(model).to(device).eval()
    openings = training_openings(manifest, games, SEED + step * 17)
    colors = [chess.WHITE if i % 2 == 0 else chess.BLACK for i in range(games)]
    records = []
    totals = Counter()
    game_lengths = []
    try:
        for offset in range(0, games, rollout_batch):
            outcome = rollout_latest_batched(snapshot, openings[offset:offset+rollout_batch],
                                             colors[offset:offset+rollout_batch], device,
                                             max_plies=max_plies)
            for samples, result in outcome:
                records.extend(samples)
                summarize_game(totals, samples, result)
                game_lengths.append(result['plies'])
            if (offset + len(outcome)) % 128 == 0 or offset + len(outcome) == games:
                print(f'  step {step}: {offset + len(outcome)}/{games} games collected', flush=True)
    finally:
        del snapshot
    metrics = detailed_ppo_update(model, optimizer, records, device, epochs=epochs,
                                  batch_size=minibatch, seed=SEED + step)
    row = {
        'step': step,
        'cumulative_games': total_games(step) if games == GAMES else START_GAMES + games,
        'games': totals['games'],
        'wins': totals['wins'], 'draws': totals['draws'], 'losses': totals['losses'],
        'truncated': totals['truncated'],
        'learner_decisions': totals['learner_decisions'],
        'mean_plies': float(np.mean(game_lengths)),
        'distinct_start_positions': games // 2,
        'recent_opponent_games': games, 'historical_opponent_games': 0,
        **metrics,
        'elapsed_seconds': round(time.perf_counter() - t0, 2),
    }
    print(f'  step {step} complete: W={row["wins"]} D={row["draws"]} '
          f'L={row["losses"]} T={row["truncated"]}, '
          f'KL+={row["approx_kl_positive"]:.5f}, '
          f'clip={row["clip_fraction"]:.3f}, '
          f'{row["elapsed_seconds"]:.1f}s', flush=True)
    return row


def save_run(model, optimizer, source, cfg, history, step):
    DIR.mkdir(parents=True, exist_ok=True)
    if step in MILESTONES and archive_path(step).exists():
        raise FileExistsError('Refusing to overwrite a saved milestone checkpoint: '
                              + str(archive_path(step)))
    saved = {
        'schema': 1, 'step': step, 'iteration': 149 + step,
        'cumulative_games': total_games(step),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_channels': int(source['model_channels']),
        'model_blocks': int(source['model_blocks']),
        'config': cfg, 'history': history,
        'description': 'Search-free batched latest-only PPO continuation from frozen S49',
        'saved_utc': datetime.now(timezone.utc).isoformat(),
    }
    # Save a resumable progress state after EVERY completed 512-game iteration.
    # Archive the specified milestones separately and never overwrite them.
    atomic_save(saved, PROGRESS)
    if step in MILESTONES:
        atomic_save(saved, archive_path(step))
    safe_csv(history, METRICS)


def run(target):
    if target not in MILESTONES:
        raise ValueError(f'Choose a saved target from {sorted(MILESTONES)}.')
    source = base_checkpoint()
    manifest = load_manifest()
    cfg = settings(manifest)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if PROGRESS.is_file():
        saved = torch.load(PROGRESS, map_location='cpu', weights_only=True)
        if saved.get('config') != cfg:
            raise RuntimeError('Existing continuation config differs. Refusing to mix experiments.')
        done = int(saved['step'])
        if done < 1 or done > max(MILESTONES) or saved.get('cumulative_games') != total_games(done):
            raise RuntimeError('Saved progress metadata is inconsistent.')
        model, optimizer = load_model_and_optimizer(saved, device, LR)
        history = saved['history']
        if len(history) != done:
            raise RuntimeError('Saved history does not contain all completed iterations.')
    else:
        if DIR.exists() and any(DIR.glob('s*.pt')):
            raise RuntimeError('Milestone archives exist but progress file is missing. '
                               'Refusing to restart and overwrite experiment.')
        model, optimizer = load_model_and_optimizer(source, device, LR)
        done = 0
        history = []
    model.eval()
    if done >= target:
        print(f'Already completed {done} new iterations ({total_games(done):,} cumulative games).', flush=True)
        return
    print(f'Device: {device}; frozen parent SHA256: {SOURCE_SHA256}', flush=True)
    print(f'Resuming from new step {done+1}. Target step {target}, '
          f'{total_games(target):,} cumulative PPO games.', flush=True)
    print('Training ONLY the isolated batched continuation. '
          'Original checkpoints and October confirmation remain read-only.', flush=True)
    for step in range(done+1, target+1):
        row = one_iteration(model, optimizer, manifest, step, device)
        history.append(row)
        save_run(model, optimizer, source, cfg, history, step)
        print(f'Saved step {step}: {total_games(step):,} cumulative PPO games.', flush=True)
    print('Finished. Resumable model:', PROGRESS, flush=True)
    if target in MILESTONES:
        print('Permanent milestone:', archive_path(target), flush=True)


def smoke():
    source = base_checkpoint()
    manifest = load_manifest()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, optimizer = load_model_and_optimizer(source, device, LR)
    model.eval()
    before = [p.detach().clone() for p in model.parameters()]
    print(f'GPU training smoke: device={device}, 4 games, max_plies=16, one PPO epoch.', flush=True)
    row = one_iteration(model, optimizer, manifest, step=1, device=device,
                        games=4, max_plies=16, rollout_batch=4, epochs=1, minibatch=8)
    if not all(math.isfinite(float(row[name])) for name in ('approx_kl_positive', 'clip_fraction', 'value_loss')):
        raise RuntimeError('Smoke PPO numerical diagnostics were not finite.')
    changed = sum(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    if changed < 1 or row['optimizer_steps'] < 1:
        raise RuntimeError('PPO training produced no optimizer update.')
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError('Frozen S49 source changed during smoke test!')
    print(f'Smoke successful: {changed} parameter tensors updated in memory.', flush=True)
    print('No checkpoints, optimizer state files, or experiment outputs were saved.', flush=True)


def report():
    source = base_checkpoint()
    if not PROGRESS.is_file():
        print('No batched continuation has started. Frozen S49 is still the latest model.')
        return
    saved = torch.load(PROGRESS, map_location='cpu', weights_only=True)
    done = int(saved['step'])
    print(f'Batched PPO: new iterations {done}; cumulative games {saved["cumulative_games"]:,}')
    print('Frozen parent SHA256:', SOURCE_SHA256)
    print('Resumable checkpoint:', PROGRESS)
    for step, milestone in MILESTONES.items():
        if archive_path(step).exists():
            print(f'Archived near {milestone:,} games:', archive_path(step))
    if saved.get('history'):
        print('Latest update metrics:', saved['history'][-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke', 'train', 'report'])
    parser.add_argument('--steps', type=int, choices=sorted(MILESTONES))
    args = parser.parse_args()
    if args.mode == 'smoke':
        smoke()
    elif args.mode == 'report':
        report()
    else:
        if args.steps is None:
            parser.error(f'train requires --steps from {sorted(MILESTONES)}')
        run(args.steps)


if __name__ == '__main__':
    main()
