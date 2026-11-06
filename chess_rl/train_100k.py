"""100k-position supervised experiment for EE641.

Streams the Lichess archive into small disk shards rather than retaining
100,000 encoded boards in RAM. This is still a pilot, not final V0 training.
"""

import argparse
import csv
import hashlib
import io
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chess.pgn
import numpy as np
import torch

from chess_rl.lichess_data import (DEFAULT_PGN, PROJECT_ROOT, positions_from_game,
                                  read_headers, read_raw_games, selection_reason,
                                  validation_game)
from chess_rl.network import ChessNetwork
from chess_rl.train_supervised import make_loader, run_epoch

import zstandard


def flush_shard(out_dir, split, pending, part):
    if not pending:
        return part
    filename = out_dir / f'{split}_{part:04d}.npz'
    np.savez_compressed(
        filename,
        boards=np.asarray([row[0] for row in pending], dtype=np.float16),
        actions=np.asarray([row[1] for row in pending], dtype=np.int32),
        outcomes=np.asarray([row[2] for row in pending], dtype=np.float32),
        legal_counts=np.asarray([row[3] for row in pending], dtype=np.int16),
    )
    pending.clear()
    return part + 1


def extract_shards(args, folder):
    stats_file = folder / 'stats.json'
    if stats_file.exists() and not args.rebuild:
        stats = json.loads(stats_file.read_text(encoding='utf-8'))
        if (stats.get('source') == args.source.name
                and stats.get('seed') == args.seed
                and stats.get('requested_positions') == args.positions
                and stats.get('max_positions_per_game') == args.positions_per_game
                and stats.get('shard_size') == args.shard_size
                and stats.get('total_positions') == args.positions
                and all((folder / file).is_file() for file in stats.get('shards', []))):
            print('Reusing previously extracted 100k dataset.', flush=True)
            return stats

    if not args.source.is_file():
        raise FileNotFoundError(f'Cannot find September archive: {args.source}')

    folder.mkdir(parents=True, exist_ok=True)
    for previous in list(folder.glob('train_*.npz')) + list(folder.glob('val_*.npz')):
        previous.unlink()
    if stats_file.exists():
        stats_file.unlink()

    counts = {'train': 0, 'val': 0}
    parts = {'train': 0, 'val': 0}
    buffers = {'train': [], 'val': []}
    rejections = Counter()
    accepted = scanned = 0
    outcomes = Counter()
    start = time.perf_counter()
    print('Reading:', args.source, flush=True)
    print('Extracting', f'{args.positions:,}', 'positions into', args.shard_size,
          'position shards. Original archive remains compressed.', flush=True)

    with zstandard.open(args.source, 'rt', encoding='utf-8', errors='replace') as handle:
        for raw in read_raw_games(handle):
            scanned += 1
            headers = read_headers(raw)
            reason = selection_reason(headers)
            rejections[reason] += 1
            if reason != 'accepted':
                if scanned >= args.max_scanned_games:
                    break
                continue

            game = chess.pgn.read_game(io.StringIO(raw))
            if game is None or game.errors:
                rejections['parse_error'] += 1
                continue
            accepted += 1
            site = headers.get('Site', str(scanned))
            site_seed = int.from_bytes(hashlib.sha256(site.encode('utf-8')).digest()[:4], 'big')
            split = 'val' if validation_game(headers) else 'train'
            for example in positions_from_game(game, args.positions_per_game,
                                               args.seed + site_seed,
                                               include_legal_count=True):
                if counts['train'] + counts['val'] >= args.positions:
                    break
                buffers[split].append(example)
                counts[split] += 1
                outcomes[('win' if example[2] == 1 else
                          'loss' if example[2] == -1 else 'draw')] += 1
                if len(buffers[split]) >= args.shard_size:
                    parts[split] = flush_shard(folder, split, buffers[split], parts[split])

            if accepted % 1000 == 0:
                n = counts['train'] + counts['val']
                print(f'  Scanned {scanned:,} games; accepted {accepted:,}; '
                      f'positions {n:,}/{args.positions:,}', flush=True)
            if counts['train'] + counts['val'] >= args.positions or scanned >= args.max_scanned_games:
                break

    for split in ('train', 'val'):
        parts[split] = flush_shard(folder, split, buffers[split], parts[split])

    total = counts['train'] + counts['val']
    if total != args.positions or min(counts.values()) < 100:
        raise RuntimeError(f'Extraction incomplete: {total:,}/{args.positions:,} positions '
                           f'(train={counts["train"]}, val={counts["val"]}).')

    shard_names = [f'{split}_{i:04d}.npz' for split in ('train', 'val')
                   for i in range(parts[split])]
    stats = {
        'source': args.source.name,
        'seed': args.seed,
        'requested_positions': args.positions,
        'total_positions': total,
        'train_positions': counts['train'],
        'validation_positions': counts['val'],
        'games_scanned': scanned,
        'games_accepted': accepted,
        'max_positions_per_game': args.positions_per_game,
        'shard_size': args.shard_size,
        'shards': shard_names,
        'outcome_counts': dict(outcomes),
        'game_selection_counts': dict(rejections),
        'extraction_seconds': round(time.perf_counter() - start, 1),
        'notes': [
            'Small-scale training experiment, not final V0.',
            'September games split deterministically by game ID, not position.',
            'October archive reserved for independent testing.',
            'Moves are labeled from human games; Stockfish evaluations are unused.',
            'Clock exclusion uses clock AFTER the move as a proxy.',
            'Time-forfeit and abandoned games excluded.',
        ],
    }
    stats_file.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(f'Extracted {counts["train"]:,} training and '
          f'{counts["val"]:,} validation positions, '
          f'from {accepted:,} accepted games in {stats["extraction_seconds"]}s.', flush=True)
    return stats


def shard_files(folder, split):
    return sorted(folder.glob(f'{split}_*.npz'))


def run_split(model, files, batch_size, device, optimizer=None, seed=0):
    files = list(files)
    if optimizer is not None:
        random.Random(seed).shuffle(files)
    totals = Counter()
    count = 0
    start = time.perf_counter()
    for file in files:
        loader = make_loader(file, batch_size, shuffle=optimizer is not None)
        # No whole-dataset allocation: only the current shard is loaded.
        metrics = run_epoch(model, loader, optimizer, device)
        n = len(loader.dataset)
        for name, value in metrics.items():
            totals[name] += value * n
        count += n
        del loader
    if not count:
        raise RuntimeError('No positions found in the dataset shards.')
    return {**{name: float(value / count) for name, value in totals.items()},
            'positions': count,
            'seconds': round(time.perf_counter() - start, 2),
            'positions_per_second': round(count / max(time.perf_counter() - start, 1e-8), 1)}


def calculate_baselines(train_files, val_files):
    frequency = Counter()
    for path in train_files:
        with np.load(path) as data:
            frequency.update(data['actions'].tolist())
    majority_action = frequency.most_common(1)[0][0]
    majority_correct = 0
    total = 0
    chance_sum = 0.0
    for path in val_files:
        with np.load(path) as data:
            actions = data['actions']
            legal_counts = data['legal_counts']
            majority_correct += int((actions == majority_action).sum())
            chance_sum += float(np.sum(1.0 / legal_counts))
            total += len(actions)
    return {'uniform_random_legal_accuracy': chance_sum / total,
            'majority_action_accuracy': majority_correct / total,
            'majority_action': majority_action}


def main():
    parser = argparse.ArgumentParser(description='EE641 sharded 100k supervised pilot')
    parser.add_argument('--source', type=Path, default=DEFAULT_PGN)
    parser.add_argument('--positions', type=int, default=100000)
    parser.add_argument('--positions-per-game', type=int, default=8)
    parser.add_argument('--shard-size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-scanned-games', type=int, default=2000000)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    if min(args.positions, args.epochs, args.batch_size, args.channels,
           args.blocks, args.shard_size, args.positions_per_game) < 1:
        parser.error('All size/count arguments must be positive.')
    if args.positions < 1000:
        parser.error('Use at least 1,000 positions for this experiment.')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('PyTorch:', torch.__version__, '| Device:', device, flush=True)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0), flush=True)
    else:
        print('WARNING: CUDA not detected; this experiment will run on CPU.', flush=True)

    folder = PROJECT_ROOT / 'data' / 'processed' / f'supervised_{args.positions}'
    folder.mkdir(parents=True, exist_ok=True)
    stats = extract_shards(args, folder)
    train_files = shard_files(folder, 'train')
    val_files = shard_files(folder, 'val')
    baselines = calculate_baselines(train_files, val_files)
    print('Validation baselines: uniform random legal move '
          f'{baselines["uniform_random_legal_accuracy"]:.1%}; '
          f'most-common action {baselines["majority_action_accuracy"]:.1%}', flush=True)

    model = ChessNetwork(channels=args.channels, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    out_dir = PROJECT_ROOT / 'outputs'
    checkpoint_dir = PROJECT_ROOT / 'checkpoints'
    out_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    rows = []
    print(f'Supervised experiment: {args.positions:,} positions, '
          f'{args.blocks} residual blocks x {args.channels} channels, '
          f'{args.epochs} epochs, batch size {args.batch_size}', flush=True)
    for epoch in range(1, args.epochs + 1):
        train = run_split(model, train_files, args.batch_size, device, optimizer, args.seed + epoch)
        val = run_split(model, val_files, args.batch_size, device)
        row = {'epoch': epoch,
               **{'train_' + k: v for k, v in train.items()},
               **{'val_' + k: v for k, v in val.items()},
               **baselines}
        rows.append(row)
        print(f'Epoch {epoch}: train acc {train["human_move_accuracy"]:.1%}, '
              f'val acc {val["human_move_accuracy"]:.1%}; '
              f'train loss {train["loss"]:.4f}, val loss {val["loss"]:.4f}; '
              f'train speed {train["positions_per_second"]:.0f} positions/sec', flush=True)

    csv_path = out_dir / f'supervised_{args.positions}_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint_path = checkpoint_dir / f'supervised_{args.positions}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_channels': args.channels,
        'model_blocks': args.blocks,
        'seed': args.seed,
        'epochs': args.epochs,
        'positions': args.positions,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'note': 'Sharded supervised experiment; NOT final V0 PPO baseline.',
        'last_validation_metrics': val,
        'data_stats': stats,
        'baselines': baselines,
    }, checkpoint_path)
    print('Saved checkpoint:', checkpoint_path, flush=True)
    print('Saved metrics:', csv_path, flush=True)
    print('Your earlier supervised_pilot.pt remains untouched.', flush=True)


if __name__ == '__main__':
    main()
