"""Independent, resumable V100 (11,840 games) -> 600K historical PPO experiment.

Starts from the original permanent V100 model and AdamW optimizer, NOT the
later 100K latest-only continuation. All other branches remain read-only.
All training uses one neural forward pass per played move, with no search.

Opponent mix for each 512-game iteration:
  25% fresh frozen snapshot of learner (current)
  50% recent archived ancestors, including newly archived branch versions
  25% diverse older ancestors, sampled from the archived history
Each starting position is played once with each learner color against the
same opponent. Only learner moves are included in PPO optimization.
"""
import argparse
import copy
import json
import math
import random
import time
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone

import chess
import numpy as np
import torch

from chess_rl.human_openings import MANIFEST, training_openings, validate_manifest
from chess_rl.ppo_batch_mixed import rollout_mixed_batched
from chess_rl.ppo_scale import load_model_and_optimizer
from chess_rl.ppo_scale_metrics import detailed_ppo_update
from chess_rl.ppo_long import safe_csv
from chess_rl.ppo_train import (CHECKPOINTS, OUTPUTS, V0, atomic_save,
                                read_historical, sha256)

PARENT = CHECKPOINTS / 'ppo_v0100.pt'
BRANCH = CHECKPOINTS / 'ppo_historical_v100'
PROGRESS = BRANCH / 'latest.pt'
METRICS = OUTPUTS / 'ppo_historical_v100_metrics.csv'
OPPONENTS = OUTPUTS / 'ppo_historical_v100_opponents.csv'
START_GAMES = 11840
GAMES = 512
MAX_PLIES = 240
BATCH = 32
EPOCHS = 2
MINIBATCH = 64
LR = 1e-5
SEED = 641528   # Original V100 current-only branch (steps 1-49).
LATER_SEED = 6412028  # Latest-only continuation (steps 50+), matched openings.
# Only original V0..V100 are eligible. Later current-only checkpoints would
# leak future training into this branch, invalidating the comparison.
EARLIER = [('original_v0', V0)] + [
    (f'original_v{v}', CHECKPOINTS / f'ppo_v{v:04d}.pt')
    for v in range(10, 101, 10)
]
# Matched cumulative totals with the existing latest-only experiment.
# Targets in this map are both permanently archived and CLI-eligible.
MILESTONES = {
    49: 36928, 75: 50240, 124: 75328, 173: 100416,
    270: 150080, 367: 199744, 563: 300096, 758: 399936,
    953: 499776, 1149: 600128, 1930: 1000000,
}
# Even when the user requests 600K in one run, save these additional
# milestones so comparison models cannot be lost.

CACHE_LIMIT = 16  # Cap frozen GPU models. Never hold the whole 100-checkpoint pool.
MODEL_CACHE = OrderedDict()


def total_games(step):
    return START_GAMES + step * GAMES


def archive_path(step):
    return BRANCH / f's{step:04d}.pt'


def should_archive(step):
    return step % 10 == 0 or step in MILESTONES


def fixed_hashes():
    missing = [str(path) for _, path in EARLIER if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing original checkpoints: ' + ', '.join(missing))
    return {name: sha256(path) for name, path in EARLIER}


def read_parent():
    if not PARENT.is_file():
        raise FileNotFoundError('Original V100 checkpoint missing: ' + str(PARENT))
    saved = torch.load(PARENT, map_location='cpu', weights_only=True)
    if int(saved.get('iteration', -1)) != 100:
        raise RuntimeError('Historical training requires original PPO V100 archive.')
    if int(saved.get('model_channels', -1)) != 128 or int(saved.get('model_blocks', -1)) != 10:
        raise RuntimeError('V100 model architecture differs from original 10-block, 128-channel network.')
    prior = saved.get('history', [])
    # V10 was initially trained for 320 games; V11..V100 contributed 90*128.
    if len(prior) != 90 or sum(int(entry.get('games', -1)) for entry in prior) != 90 * 128:
        raise RuntimeError('V100 PPO history does not match the expected 11,840-game parent.')
    if not {'model_state_dict', 'optimizer_state_dict', 'model_channels', 'model_blocks'} <= saved.keys():
        raise RuntimeError('Parent checkpoint lacks model or optimizer state.')
    config = saved.get('config', {})
    if config.get('start_iteration') != 10 or config.get('opponent_mix') != '50% recent frozen, 50% uniform archived V0/V10/...':
        raise RuntimeError('Unexpected V100 training provenance. This is not the original historical archive.')
    if config.get('v0_sha256') != sha256(V0):
        raise RuntimeError('V100 was trained from a different V0 supervised checkpoint.')
    return saved


def load_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError('Human opening manifest missing: ' + str(MANIFEST))
    return validate_manifest(json.loads(MANIFEST.read_text(encoding='utf-8')))


def configuration(parent_hash, hashes, manifest):
    return {
        'schema': 2, 'branch': 'historical_from_v100',
        'parent_file': 'checkpoints/ppo_v0100.pt',
        'parent_sha256': parent_hash,
        'fixed_opponent_sha256': hashes,
        'training_opening_manifest_sha256': manifest['sha256'],
        'start_games': START_GAMES, 'games_per_iteration': GAMES,
        'max_plies': MAX_PLIES, 'rollout_batch': BATCH,
        'ppo_epochs': EPOCHS, 'minibatch': MINIBATCH, 'learning_rate': LR,
        'seed_initial': SEED, 'seed_later': LATER_SEED, 'gamma': 1.0, 'gae_lambda': 0.95,
        'clip': 0.1, 'entropy_weight': 0.01, 'value_weight': 0.5,
        'mix': '25% current / 50% recent frozen ancestors / 25% old ancestors',
        'paired_colors': True, 'search': False,
        'recent_count': 5, 'older_sample_count': 5,
    }


def all_ancestors(step):
    """Archived predecessors only, never a checkpoint at or after this step."""
    ancestors = dict(EARLIER)
    for i in range(10, step, 10):
        ancestors[f'hist_s{i:04d}'] = archive_path(i)
    # Milestone archives not exactly divisible by ten are also ancestors.
    for i in MILESTONES:
        if i < step:
            ancestors[f'hist_s{i:04d}'] = archive_path(i)
    return ancestors


def opponent_pools(step, seed=SEED):
    """Five closest archived ancestors, plus five sampled older alternatives."""
    ancestors = all_ancestors(step)
    # Chronological V0..V100, followed by newly archived branch ancestors.
    line = [name for name, _ in EARLIER]
    line += [f'hist_s{i:04d}' for i in sorted(
        set([i for i in range(10, step, 10)] + [i for i in MILESTONES if i < step]))]
    recent = list(reversed(line[-5:]))
    older = [name for name in ancestors if name not in recent]
    if not recent or not older:
        raise RuntimeError('Historical pool has no recent or old checkpoint candidates.')
    rng = random.Random(seed + step * 1907)
    older_selected = rng.sample(older, min(5, len(older)))
    # Validate only the opponent files actually loaded this iteration.
    selected = {name: ancestors[name] for name in recent + older_selected}
    return recent, older_selected, selected


def choose_opponents(games, recent, older, seed):
    """Balanced 2-color pairs, with exact 25/50/25 mix for 512 games."""
    if games < 8 or games % 8:
        raise ValueError('Games must be divisible by eight and >= 8.')
    if not recent or not older:
        raise ValueError('Need both recent and older opponent choices.')
    pairs = games // 2
    rng = random.Random(seed)
    bands = ['current'] * (pairs // 4)
    bands += ['recent'] * (pairs // 2)
    bands += ['old'] * (pairs // 4)
    rng.shuffle(bands)
    rivals = []
    for label in bands:
        if label == 'current':
            chosen = 'current'
        elif label == 'recent':
            chosen = rng.choice(recent)
        else:
            chosen = rng.choice(older)
        rivals.extend((chosen, chosen))
    assert len(rivals) == games
    return rivals


def opponent_model(name, path, device):
    """Small read-only LRU cache; archived model weights never receive updates."""
    cache_key = (str(path), str(device))
    if cache_key in MODEL_CACHE:
        MODEL_CACHE.move_to_end(cache_key)
        return MODEL_CACHE[cache_key]
    if not path.is_file():
        raise FileNotFoundError('Required historical checkpoint missing: ' + str(path))
    model = read_historical(path, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    MODEL_CACHE[cache_key] = model
    while len(MODEL_CACHE) > CACHE_LIMIT:
        MODEL_CACHE.popitem(last=False)
    return model


def schedule_seed(step):
    # Preserve the original latest-only run's sampled opening schedule,
    # even though rollout batching and changing opponents alter the games.
    return (SEED, step) if step <= 49 else (LATER_SEED, step - 49)


def one_iteration(model, optimizer, manifest, step, device, games=GAMES,
                  max_plies=MAX_PLIES, epochs=EPOCHS, minibatch=MINIBATCH):
    if games < 8 or games % 8:
        raise ValueError('Training games must be divisible by 8.')
    start = time.perf_counter()
    seed, phase_step = schedule_seed(step)
    torch.manual_seed(seed + phase_step * 167)
    random.seed(seed + phase_step * 9127)
    frozen_current = copy.deepcopy(model).to(device).eval()
    for parameter in frozen_current.parameters():
        parameter.requires_grad_(False)
    recent, older, selected = opponent_pools(step)
    print(f'  historical step {step}: recent={recent}; older={older}', flush=True)
    rival_names = choose_opponents(games, recent, older, seed + phase_step * 9127)
    frozen = {name: opponent_model(name, selected[name], device)
              for name in set(rival_names) if name != 'current'}
    openings = training_openings(manifest, games, seed + phase_step * 17)
    colors = [chess.WHITE if i % 2 == 0 else chess.BLACK for i in range(games)]
    experiences = []
    totals = Counter()
    breakdown = defaultdict(Counter)
    lengths = []
    for offset in range(0, games, BATCH):
        count = min(BATCH, games - offset)
        outcomes = rollout_mixed_batched(
            frozen_current, frozen, openings[offset:offset+count],
            colors[offset:offset+count], rival_names[offset:offset+count],
            device, max_plies=max_plies)
        for name, (samples, result) in zip(rival_names[offset:offset+count], outcomes):
            experiences.extend(samples)
            totals['games'] += 1
            totals['learner_decisions'] += len(samples)
            lengths.append(result['plies'])
            field = ('truncated' if result['truncated'] else
                     'wins' if result['outcome_for_learner'] > 0 else
                     'losses' if result['outcome_for_learner'] < 0 else 'draws')
            totals[field] += 1
            breakdown[name][field] += 1
            breakdown[name]['games'] += 1
        if (offset + count) % 128 == 0 or offset + count == games:
            print(f'  historical step {step}: {offset+count}/{games} games collected', flush=True)
    del frozen_current, frozen
    metrics = detailed_ppo_update(model, optimizer, experiences, device,
                                  epochs=epochs, batch_size=minibatch, seed=seed + phase_step)
    row = {
        'step': step, 'cumulative_games': total_games(step) if games == GAMES else None,
        'games': games, 'wins': totals['wins'], 'draws': totals['draws'],
        'losses': totals['losses'], 'truncated': totals['truncated'],
        'learner_decisions': totals['learner_decisions'],
        'mean_plies': float(np.mean(lengths)), 'distinct_start_positions': games // 2,
        'current_opponent_games': breakdown['current']['games'],
        'historical_opponent_games': games - breakdown['current']['games'],
        'recent_historical_games': sum(breakdown[name]['games'] for name in recent),
        'older_historical_games': sum(breakdown[name]['games'] for name in older),
        **metrics,
        'elapsed_seconds': round(time.perf_counter()-start, 2),
    }
    opponent_rows = [dict(step=step, opponent=name,
                          games=counters['games'], wins=counters['wins'],
                          draws=counters['draws'], losses=counters['losses'],
                          truncated=counters['truncated'])
                     for name, counters in sorted(breakdown.items())]
    print(f'  historical step {step}: W={row["wins"]} D={row["draws"]} '
          f'L={row["losses"]} T={row["truncated"]}; '
          f'current/recent/old={row["current_opponent_games"]}/'
          f'{row["recent_historical_games"]}/{row["older_historical_games"]}; '
          f'KL+={row["approx_kl_positive"]:.5f}, clip={row["clip_fraction"]:.3f}; '
          f'{row["elapsed_seconds"]:.1f}s', flush=True)
    return row, opponent_rows


def save_iteration(model, optimizer, parent, config, history, opponent_history, step):
    target = archive_path(step)
    if should_archive(step) and target.exists():
        raise FileExistsError('Refusing to overwrite archived historical checkpoint: ' + str(target))
    saved = {
        'schema': 2, 'step': step, 'iteration': 100 + step,
        'cumulative_games': total_games(step),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_channels': int(parent['model_channels']),
        'model_blocks': int(parent['model_blocks']),
        'config': config, 'history': history, 'opponent_history': opponent_history,
        'description': 'Search-free batched historical-opponent continuation from original V100',
        'saved_utc': datetime.now(timezone.utc).isoformat(),
    }
    atomic_save(saved, PROGRESS)
    if should_archive(step):
        atomic_save(saved, target)
    safe_csv(history, METRICS)
    safe_csv(opponent_history, OPPONENTS)


def run(target):
    if target not in MILESTONES:
        raise ValueError('Target must be one of ' + str(sorted(MILESTONES)))
    parent = read_parent()
    parent_hash = sha256(PARENT)
    manifest = load_manifest()
    hashes = fixed_hashes()
    cfg = configuration(parent_hash, hashes, manifest)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if PROGRESS.exists():
        saved = torch.load(PROGRESS, map_location='cpu', weights_only=True)
        if saved.get('config') != cfg:
            raise RuntimeError('Historical branch settings/parent hashes changed. Refusing to mix runs.')
        done = int(saved['step'])
        if (done < 1 or done > max(MILESTONES) or saved.get('cumulative_games') != total_games(done)
                or len(saved['history']) != done):
            raise RuntimeError('Historical progress is inconsistent.')
        for i in sorted(set(range(10, done+1, 10)) |
                        {m for m in MILESTONES if m <= done}):
            if not archive_path(i).exists():
                if i == done:
                    # Crash between atomic progress save and archival save.
                    atomic_save(saved, archive_path(i))
                else:
                    raise RuntimeError(f'Missing prior frozen archive: {archive_path(i)}')
        model, optimizer = load_model_and_optimizer(saved, device, LR)
        history = saved['history']
        opponent_history = saved['opponent_history']
    else:
        if BRANCH.exists() and any(BRANCH.iterdir()):
            raise RuntimeError('Historical branch folder already contains files but no progress. '
                               'Refusing to overwrite or restart it.')
        model, optimizer = load_model_and_optimizer(parent, device, LR)
        done, history, opponent_history = 0, [], []
    model.eval()
    if done >= target:
        print(f'Historical branch already at step {done}: {total_games(done):,} total games.')
        return
    print(f'Historical-pool PPO, device={device}, frozen V100 parent SHA256={parent_hash}', flush=True)
    print(f'New step {done+1}; target step {target} ({total_games(target):,} cumulative games).', flush=True)
    print('Opponent sampling per 512 games: 128 current, 256 recent archived, '
          '128 older archived. Colors paired; no search.', flush=True)
    for step in range(done + 1, target + 1):
        row, opponents = one_iteration(model, optimizer, manifest, step, device)
        history.append(row)
        opponent_history.extend(opponents)
        save_iteration(model, optimizer, parent, cfg, history, opponent_history, step)
        print(f'Saved historical step {step}: {total_games(step):,} cumulative games.', flush=True)
    print('Historical branch finished. Resumable checkpoint:', PROGRESS, flush=True)
    if should_archive(target):
        print('Permanent milestone:', archive_path(target), flush=True)


def smoke():
    """Tiny real GPU/CPU test; no training files, checkpoint, or metrics written."""
    parent = read_parent()
    manifest = load_manifest()
    fixed_hashes()  # Validate all source opponents before beginning.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, optimizer = load_model_and_optimizer(parent, device, LR)
    before = [p.detach().clone() for p in model.parameters()]
    row, rows = one_iteration(model, optimizer, manifest, 1, device,
                              games=8, max_plies=16, epochs=1, minibatch=8)
    assert row['current_opponent_games'] == 2
    assert row['recent_historical_games'] == 4
    assert row['older_historical_games'] == 2
    assert len(rows) > 1
    if not math.isfinite(row['approx_kl_positive']) or not math.isfinite(row['clip_fraction']):
        raise RuntimeError('Nonfinite PPO diagnostics during smoke.')
    changed = sum(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    if changed < 1 or row['optimizer_steps'] < 1:
        raise RuntimeError('No PPO model parameters updated in smoke.')
    print(f'Historical smoke passed: {changed} parameter tensors changed in memory. '
          'No checkpoints or experiment outputs saved.', flush=True)


def report():
    if not PROGRESS.exists():
        print('No V100 historical-branch checkpoint yet. Original self-play branch is unchanged.')
        return
    saved = torch.load(PROGRESS, map_location='cpu', weights_only=True)
    step = saved['step']
    print(f'Historical training: step={step}; cumulative games={total_games(step):,}')
    print(f'Progress={PROGRESS}; original V100 parent={PARENT}')
    print('Last iteration:', saved['history'][-1])
    print('Opponent mix:', saved['opponent_history'][-12:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['smoke', 'train', 'report'])
    p.add_argument('--steps', type=int, choices=sorted(MILESTONES))
    args = p.parse_args()
    if args.mode == 'smoke':
        smoke()
    elif args.mode == 'report':
        report()
    elif args.steps is None:
        p.error('train requires --steps from ' + str(sorted(MILESTONES)))
    else:
        run(args.steps)


if __name__ == '__main__':
    main()
