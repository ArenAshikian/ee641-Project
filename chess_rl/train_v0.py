"""EE641 full supervised V0 training for a search-free chess policy.

Reads September Lichess PGN.ZST directly. Streams 20M board/move examples,
stores compact shards, learns a 10x128 policy-value network, and saves a
permanent V0. Self-play PPO is a separate future phase.

No PGN [%eval] annotations, Stockfish, tree search, or engine labels used.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
import zstandard

from chess_rl.compact_dataset import load_shard, write_shard
from chess_rl.lichess_data import (
    DEFAULT_PGN, PROJECT_ROOT, read_raw_games, read_headers,
    selection_reason, validation_game,
)
from chess_rl.network import ChessNetwork
from chess_rl.train_1m import examples_from_game, PHASE_NAMES


DATA_VERSION = 2
TRAINING_VERSION = 1
DATA_ROOT = PROJECT_ROOT / 'data' / 'processed'


def atomic_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temp, path)


def config_for(args):
    return {
        'data_version': DATA_VERSION,
        'source': args.source.name,
        'source_size_bytes': args.source.stat().st_size,
        'positions': args.positions,
        'positions_per_game': args.positions_per_game,
        'seed': args.seed,
        'checkpoint_games': args.checkpoint_games,
    }


def initial_progress(args):
    return {
        'config': config_for(args),
        'scanned': 0, 'accepted': 0,
        'counts': {'train': 0, 'val': 0},
        'parts': {'train': 0, 'val': 0},
        'rejections': {},
        'thirds': {'train': {}, 'val': {}},
        'sum_legal': {'train': {}, 'val': {}},
        'results': {},
        'time_controls': {},
        'rating_bands': {},
        'sum_game_plies': 0,
        'min_game_plies': None,
        'max_game_plies': None,
    }


def extract(args, folder):
    folder.mkdir(parents=True, exist_ok=True)
    done_path = folder / 'stats.json'
    progress_path = folder / 'extraction_progress.json'
    expected = config_for(args)
    if done_path.is_file():
        stats = json.loads(done_path.read_text(encoding='utf-8'))
        if stats.get('config') != expected:
            raise RuntimeError('An existing V0 dataset has different settings. Use a different dataset folder.')
        shard_paths = list(folder.glob('train_*.npz')) + list(folder.glob('val_*.npz'))
        if (len(shard_paths) != sum(stats['parts'].values()) or
                not all(f.is_file() for f in shard_paths)):
            raise RuntimeError('Saved dataset shards are missing. Do not continue training.')
        print(f'Reusing {args.positions:,} previously extracted positions.', flush=True)
        return stats

    if progress_path.is_file():
        state = json.loads(progress_path.read_text(encoding='utf-8'))
        if state['config'] != expected:
            raise RuntimeError('Interrupted extraction settings differ from this run.')
        print(f'Resuming extraction after {state["scanned"]:,} scanned games.', flush=True)
    else:
        state = initial_progress(args)
        if list(folder.glob('train_*.npz')) or list(folder.glob('val_*.npz')):
            raise RuntimeError('Found shards without extraction_progress.json; move this dataset folder aside.')

    # Drop incomplete/uncommitted files from a crash between shard write and state commit.
    for split in ('train', 'val'):
        for file in folder.glob(f'{split}_*.npz'):
            if file.name.endswith('.partial.npz') or int(file.stem.rsplit('_', 1)[1]) >= state['parts'][split]:
                file.unlink()
        for i in range(state['parts'][split]):
            if not (folder / f'{split}_{i:05d}.npz').is_file():
                raise RuntimeError(f'Missing committed {split} shard {i}.')

    rejections = Counter(state['rejections'])
    thirds = {s: Counter(state['thirds'][s]) for s in ('train', 'val')}
    legal = {s: Counter(state['sum_legal'][s]) for s in ('train', 'val')}
    results = Counter(state['results'])
    time_controls = Counter(state['time_controls'])
    rating_bands = Counter(state['rating_bands'])
    buffers = {'train': [], 'val': []}
    start_time = time.perf_counter()
    already_scanned = state['scanned']
    print(f'Streaming {args.source.name}; target {args.positions:,} positions.', flush=True)
    print('Board representation is bit-packed; source archive stays compressed.', flush=True)

    def commit():
        for split in ('train', 'val'):
            state['parts'][split] = write_shard(folder, split, state['parts'][split], buffers[split])
        state['rejections'] = dict(rejections)
        state['thirds'] = {s: dict(thirds[s]) for s in ('train', 'val')}
        state['sum_legal'] = {s: dict(legal[s]) for s in ('train', 'val')}
        state['results'] = dict(results)
        state['time_controls'] = dict(time_controls)
        state['rating_bands'] = dict(rating_bands)
        atomic_json(progress_path, state)

    with zstandard.open(args.source, 'rt', encoding='utf-8', errors='replace') as stream:
        for game_i, raw in enumerate(read_raw_games(stream), start=1):
            if game_i <= already_scanned:
                continue
            state['scanned'] = game_i
            headers = read_headers(raw)
            reason = selection_reason(headers)
            rejections[reason] += 1
            if reason != 'accepted':
                if game_i >= args.max_scanned_games:
                    break
                continue
            game = chess.pgn.read_game(io.StringIO(raw))
            if game is None or game.errors:
                rejections['parse_error'] += 1
                continue
            state['accepted'] += 1
            plies = game.end().ply()
            state['sum_game_plies'] += plies
            state['min_game_plies'] = min(state['min_game_plies'] or plies, plies)
            state['max_game_plies'] = max(state['max_game_plies'] or plies, plies)
            results[headers['Result']] += 1
            event = headers.get('Event', '').lower()
            time_controls['blitz' if 'blitz' in event else 'rapid'] += 1
            for color in ('White', 'Black'):
                rating = int(headers[f'{color}Elo'])
                rating_bands[f'{(rating // 100) * 100}-{(rating // 100) * 100 + 99}'] += 1
            split = 'val' if validation_game(headers) else 'train'
            site = headers.get('Site') or str(game_i)
            game_seed = args.seed + int.from_bytes(hashlib.sha256(site.encode()).digest()[:4], 'big')
            for example in examples_from_game(game, args.positions_per_game, game_seed):
                if state['counts']['train'] + state['counts']['val'] >= args.positions:
                    break
                buffers[split].append(example)
                state['counts'][split] += 1
                third = PHASE_NAMES[example[4]]
                thirds[split][third] += 1
                legal[split][third] += example[3]
            if state['accepted'] % args.checkpoint_games == 0:
                commit()
                n = state['counts']['train'] + state['counts']['val']
                print(f'  Scanned {game_i:,}; accepted {state["accepted"]:,}; '
                      f'saved {n:,}/{args.positions:,} positions; '
                      f'elapsed {time.perf_counter()-start_time:.0f}s', flush=True)
            if state['counts']['train'] + state['counts']['val'] >= args.positions or game_i >= args.max_scanned_games:
                break

    commit()
    total = sum(state['counts'].values())
    if total < args.positions:
        raise RuntimeError(f'Only extracted {total:,}/{args.positions:,} positions. '
                           'Increase --max-scanned-games if the archive has enough data.')
    if min(state['counts'].values()) < 1000:
        raise RuntimeError('Validation or training split has fewer than 1,000 positions.')
    stats = {**state, 'total_positions': total,
             'game_split': 'Hash of entire game ID, 90% train / 10% validation',
             'sample_description': 'Up to 12 moves from early, middle, late thirds per selected full game',
             'source_sampling_limit': 'The first eligible games encountered in the September archive',
             'engine_evaluations': 'Never read PGN engine [%eval] annotations',
             'October_archive': 'Reserved for independent final test',
             'post_move_clock_filter': 'Skip moves with recorded remaining time below 30 seconds',
             'total_extraction_seconds_in_this_run': round(time.perf_counter() - start_time, 1)}
    atomic_json(done_path, stats)
    progress_path.unlink(missing_ok=True)
    print(f'Extracted {state["counts"]["train"]:,} training and '
          f'{state["counts"]["val"]:,} validation positions.', flush=True)
    print('Dataset stats:', done_path, flush=True)
    return stats


def shard_files(folder, split):
    return sorted(folder.glob(f'{split}_*.npz'))


def baselines(train_shards, val_shards):
    # Uniform legal-move accuracy: average 1 / number of legal moves in each state.
    # Most-common action is fitted on training only (no validation leakage).
    frequencies = Counter()
    for file in train_shards:
        with np.load(file) as data:
            frequencies.update(data['actions'].tolist())
    most_common = frequencies.most_common(1)[0][0]
    totals = Counter()
    for file in val_shards:
        with np.load(file) as data:
            actions = data['actions']
            counts = data['legal_counts'].astype(np.float64)
            totals['n'] += len(actions)
            totals['correct'] += int(np.count_nonzero(actions == most_common))
            totals['random_sum'] += float((1.0 / counts).sum())
    return {'most_common_action': int(most_common),
            'most_common_action_accuracy': totals['correct'] / totals['n'],
            'random_legal_accuracy': totals['random_sum'] / totals['n']}


def one_epoch(model, shards, device, batch_size, optimizer=None, seed=7):
    is_train = optimizer is not None
    model.train(is_train)
    paths = list(shards)
    if is_train:
        random.Random(seed).shuffle(paths)
    totals = Counter()
    third_total = np.zeros(3, dtype=np.int64)
    third_correct = np.zeros(3, dtype=np.int64)
    start = time.perf_counter()
    for file in paths:
        boards, actions, outcomes, thirds, _ = load_shard(file)
        order = np.arange(len(actions))
        if is_train:
            np.random.default_rng(seed ^ int(hashlib.md5(file.name.encode()).hexdigest()[:8], 16)).shuffle(order)
        # The complete loaded shard stays CPU-side, one minibatch goes to GPU.
        with torch.set_grad_enabled(is_train):
            for i in range(0, len(actions), batch_size):
                ix = order[i:i+batch_size]
                x = torch.from_numpy(boards[ix].astype(np.float32)).to(device)
                y = torch.from_numpy(actions[ix]).to(device)
                target = torch.from_numpy(outcomes[ix]).to(device)
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                logits, values = model(x)
                policy = F.cross_entropy(logits, y)
                value = F.mse_loss(values, target)
                loss = policy + 0.25 * value
                if is_train:
                    loss.backward()
                    optimizer.step()
                n = len(ix)
                totals['n'] += n
                totals['correct'] += int((logits.argmax(1) == y).sum().item())
                totals['loss'] += float(loss.item()) * n
                totals['policy'] += float(policy.item()) * n
                totals['value'] += float(value.item()) * n
                matches = (logits.argmax(1) == y).detach().cpu().numpy()
                third_total += np.bincount(thirds[ix], minlength=3)
                third_correct += np.bincount(thirds[ix], weights=matches.astype(np.int64), minlength=3).astype(np.int64)
        del boards, actions, outcomes, thirds
    elapsed = time.perf_counter()-start
    result = {'positions': totals['n'],
              'accuracy': totals['correct']/totals['n'],
              'loss': totals['loss']/totals['n'],
              'policy_loss': totals['policy']/totals['n'],
              'value_mse': totals['value']/totals['n'],
              'seconds': round(elapsed, 2),
              'positions_per_second': round(totals['n']/max(elapsed, 1e-3), 1)}
    for i, label in enumerate(PHASE_NAMES):
        result[f'{label}_accuracy'] = float(third_correct[i]/third_total[i]) if third_total[i] else None
    return result


def write_metrics(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_progress(path, model, optimizer, scheduler, epoch, rows, best_accuracy, best_state, args, stats):
    temporary = path.with_suffix('.tmp')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_model_state_dict': best_state,
        'best_validation_accuracy': best_accuracy,
        'last_epoch': epoch,
        'rows': rows,
        'config': {'positions': args.positions, 'blocks': args.blocks,
                   'channels': args.channels, 'seed': args.seed, 'epochs': args.epochs},
        'data_config': stats['config'],
    }, temporary)
    os.replace(temporary, path)


def train(args, folder, stats):
    device = torch.device('cuda')
    output_dir = PROJECT_ROOT / 'outputs'
    checkpoint_dir = PROJECT_ROOT / 'checkpoints'
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    train_files = shard_files(folder, 'train')
    val_files = shard_files(folder, 'val')
    print(f'{len(train_files):,} train shards, {len(val_files):,} validation shards.', flush=True)
    control = baselines(train_files, val_files)
    print(f'Baselines: random legal {control["random_legal_accuracy"]:.1%}; '
          f'most-common action {control["most_common_action_accuracy"]:.1%}', flush=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = ChessNetwork(channels=args.channels, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate/10)
    progress_file = checkpoint_dir / 'v0_training_progress.pt'
    final_file = checkpoint_dir / 'v0_supervised.pt'
    metrics_file = output_dir / 'v0_supervised_metrics.csv'
    if final_file.exists():
        print(f'Permanent V0 already exists: {final_file}. Nothing overwritten.', flush=True)
        return
    first_epoch, rows, best_accuracy, best_state = 1, [], -1.0, None
    settings = {'positions': args.positions, 'blocks': args.blocks,
                'channels': args.channels, 'seed': args.seed, 'epochs': args.epochs}
    if progress_file.exists():
        saved = torch.load(progress_file, map_location='cpu', weights_only=True)
        if saved['config'] != settings or saved['data_config'] != stats['config']:
            raise RuntimeError('Progress checkpoint settings differ from this run.')
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        rows = saved['rows']
        first_epoch = saved['last_epoch'] + 1
        best_accuracy = saved['best_validation_accuracy']
        best_state = saved['best_model_state_dict']
        print(f'Resuming V0 training at epoch {first_epoch}.', flush=True)
    for epoch in range(first_epoch, args.epochs+1):
        lr = optimizer.param_groups[0]['lr']
        print(f'V0 epoch {epoch}/{args.epochs}, LR={lr:.2e}', flush=True)
        train_result = one_epoch(model, train_files, device, args.batch_size, optimizer, args.seed + epoch)
        val_result = one_epoch(model, val_files, device, args.batch_size)
        if val_result['accuracy'] > best_accuracy:
            best_accuracy = val_result['accuracy']
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        scheduler.step()
        row = {'epoch': epoch, 'lr': lr,
               **{f'train_{k}': v for k,v in train_result.items()},
               **{f'val_{k}': v for k,v in val_result.items()}}
        rows.append(row)
        write_metrics(metrics_file, rows)
        save_progress(progress_file, model, optimizer, scheduler, epoch, rows,
                      best_accuracy, best_state, args, stats)
        print(f'  Train accuracy {train_result["accuracy"]:.1%}, '
              f'validation {val_result["accuracy"]:.1%}; '
              f'early {val_result["early_accuracy"]:.1%}, '
              f'middle {val_result["middle_accuracy"]:.1%}, '
              f'late {val_result["late_accuracy"]:.1%}; '
              f'val value MSE {val_result["value_mse"]:.4f}', flush=True)
        print(f'  Speed {train_result["positions_per_second"]:,.0f} positions/sec. '
              'Saved resumable training checkpoint.', flush=True)

    # The best validation checkpoint becomes V0, not necessarily the last epoch.
    model.load_state_dict(best_state)
    final = {
        'model_state_dict': model.state_dict(),
        'model_channels': args.channels, 'model_blocks': args.blocks,
        'positions': args.positions, 'epochs': args.epochs,
        'seed': args.seed, 'best_validation_accuracy': best_accuracy,
        'selected_epoch': max(rows, key=lambda row: row['val_accuracy'])['epoch'],
        'best_validation_metrics': {k[4:]:v for k,v in max(rows, key=lambda row: row['val_accuracy']).items() if k.startswith('val_')},
        'baseline_accuracy': control,
        'data_stats': stats,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'description': 'Permanent supervised V0 baseline. No PPO training performed.',
    }
    torch.save(final, final_file)
    print('Saved permanent V0:', final_file, flush=True)
    print('Saved metric history:', metrics_file, flush=True)
    progress_file.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='EE641 full supervised V0 training')
    parser.add_argument('--source', type=Path, default=DEFAULT_PGN)
    parser.add_argument('--positions', type=int, default=20_000_000)
    parser.add_argument('--positions-per-game', type=int, default=12)
    parser.add_argument('--checkpoint-games', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--blocks', type=int, default=10)
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-scanned-games', type=int, default=50_000_000)
    args = parser.parse_args()
    if (min(args.positions, args.positions_per_game, args.checkpoint_games, args.epochs,
            args.blocks, args.channels, args.batch_size, args.max_scanned_games) <= 0 or
            args.positions < 10_000):
        parser.error('All counts must be positive, and at least 10,000 positions are required.')
    if not args.source.is_file():
        parser.error(f'September archive not found: {args.source}')
    if not torch.cuda.is_available():
        parser.error('CUDA is required for 20M V0 training. Check GPU setup.')
    print('GPU:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__, flush=True)
    if (PROJECT_ROOT / 'checkpoints' / 'v0_supervised.pt').exists():
        print('Permanent V0 already exists. Training will not overwrite it.', flush=True)
        return
    folder = DATA_ROOT / f'v0_{args.positions}'
    if not (folder/'stats.json').exists():
        free = shutil.disk_usage(DATA_ROOT).free / (1024**3)
        if free < 12:
            parser.error(f'Only {free:.1f} GiB free in {DATA_ROOT}; '
                         'at least 12 GiB recommended for compact 20M-position data.')
    stats = extract(args, folder)
    train(args, folder, stats)


if __name__ == '__main__':
    main()
