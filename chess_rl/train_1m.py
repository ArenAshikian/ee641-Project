"""EE641 supervised 1-million-position experiment.

Read compressed Lichess games, sample across each game's early/middle/late
thirds, train a larger policy-value network, and run small match diagnostics.
No engine labels, tree search, or engine-assisted move selection are used here.
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

import chess
import chess.pgn
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import zstandard

from chess_rl.encoding import action_for_move, encode_board
from chess_rl.environment import ChessEnvironment
from chess_rl.lichess_data import (
    DEFAULT_PGN, PROJECT_ROOT, read_raw_games, read_headers,
    selection_reason, validation_game,
)
from chess_rl.network import ChessNetwork


PHASE_NAMES = ('early', 'middle', 'late')
DATA_VERSION = 1


def pick_moves_by_game_third(game, limit=12, seed=7):
    """Return (move_index, third) pairs, sampled throughout a complete game.

    Thirds are relative to GAME LENGTH, not formal chess opening/endgame labels.
    Missing or short-game thirds are filled from other eligible moves.
    """
    nodes = list(game.mainline())
    if not nodes:
        return []
    groups = [[], [], []]
    for i, node in enumerate(nodes):
        remaining_clock = node.clock()  # PGN [%clk] records clock AFTER the move.
        if remaining_clock is not None and remaining_clock < 30:
            continue
        third = min(2, (i * 3) // len(nodes))
        groups[third].append(i)

    rng = random.Random(seed)
    selected = []
    quota, extra = divmod(limit, 3)
    for phase in range(3):
        take = min(len(groups[phase]), quota + (phase < extra))
        selected.extend((i, phase) for i in rng.sample(groups[phase], take))

    selected_indices = {i for i, phase in selected}
    leftovers = [(i, phase) for phase, group in enumerate(groups)
                 for i in group if i not in selected_indices]
    if len(selected) < limit and leftovers:
        selected.extend(rng.sample(leftovers, min(limit - len(selected), len(leftovers))))
    return sorted(selected)


def examples_from_game(game, limit=12, seed=7):
    """Yield board, human action, player-relative outcome, legal count, game third."""
    chosen = dict(pick_moves_by_game_third(game, limit, seed))
    result = game.headers.get('Result', '*')
    winner = chess.WHITE if result == '1-0' else chess.BLACK if result == '0-1' else None
    board = game.board()
    for i, node in enumerate(game.mainline()):
        if i in chosen:
            outcome = 0.0 if winner is None else (1.0 if winner == board.turn else -1.0)
            yield (encode_board(board), action_for_move(board, node.move),
                   outcome, board.legal_moves.count(), chosen[i])
        board.push(node.move)


def save_shard(folder, split, part, buffer):
    """Flush at most one small shard to disk; do not hold the full dataset in RAM."""
    if not buffer:
        return part
    file = folder / f'{split}_{part:04d}.npz'
    np.savez_compressed(
        file,
        boards=np.asarray([row[0] for row in buffer], dtype=np.float16),
        actions=np.asarray([row[1] for row in buffer], dtype=np.int32),
        outcomes=np.asarray([row[2] for row in buffer], dtype=np.float32),
        legal_counts=np.asarray([row[3] for row in buffer], dtype=np.int16),
        thirds=np.asarray([row[4] for row in buffer], dtype=np.int8),
    )
    buffer.clear()
    return part + 1


def files_for(folder, split):
    return sorted(folder.glob(f'{split}_*.npz'))


def extract(args, folder):
    metadata = folder / 'stats.json'
    expected = {
        'data_version': DATA_VERSION,
        'source': args.source.name,
        'source_bytes': args.source.stat().st_size if args.source.exists() else 0,
        'requested_positions': args.positions,
        'positions_per_game': args.positions_per_game,
        'shard_size': args.shard_size,
        'seed': args.seed,
    }
    if metadata.exists() and not args.rebuild:
        old = json.loads(metadata.read_text(encoding='utf-8'))
        if (all(old.get(k) == value for k, value in expected.items())
                and old.get('total_positions') == args.positions
                and all((folder / name).is_file() for name in old.get('shards', []))):
            print('Reusing the previously extracted 1M dataset.', flush=True)
            return old

    folder.mkdir(parents=True, exist_ok=True)
    for path in list(files_for(folder, 'train')) + list(files_for(folder, 'val')):
        path.unlink()
    metadata.unlink(missing_ok=True)

    counts = Counter()
    thirds = {'train': Counter(), 'val': Counter()}
    rejections = Counter()
    game_results = Counter()
    time_controls = Counter()
    game_lengths = []
    legal_sums = Counter()
    buffers = {'train': [], 'val': []}
    parts = {'train': 0, 'val': 0}
    scanned = accepted = 0
    start = time.perf_counter()
    print('Reading:', args.source, flush=True)
    print(f'Extracting {args.positions:,} positions; sampled throughout complete games.', flush=True)
    print('The original .pgn.zst archive remains compressed.', flush=True)

    with zstandard.open(args.source, 'rt', encoding='utf-8', errors='replace') as handle:
        for raw in read_raw_games(handle):
            scanned += 1
            headers = read_headers(raw)
            reason = selection_reason(headers)
            rejections[reason] += 1
            if reason == 'accepted':
                game = chess.pgn.read_game(io.StringIO(raw))
                if game is None or game.errors:
                    rejections['parse_error'] += 1
                else:
                    accepted += 1
                    game_results[headers.get('Result', '*')] += 1
                    time_controls[('blitz' if 'blitz' in headers.get('Event', '').lower()
                                   else 'rapid')] += 1
                    game_lengths.append(game.end().ply())
                    site = headers.get('Site', str(scanned))
                    digest = hashlib.sha256(site.encode('utf-8')).digest()
                    game_seed = args.seed + int.from_bytes(digest[:4], 'big')
                    split = 'val' if validation_game(headers) else 'train'
                    for example in examples_from_game(game, args.positions_per_game, game_seed):
                        if counts['total'] >= args.positions:
                            break
                        buffers[split].append(example)
                        counts[split] += 1
                        counts['total'] += 1
                        thirds[split][PHASE_NAMES[example[4]]] += 1
                        legal_sums[(split, PHASE_NAMES[example[4]])] += example[3]
                        if len(buffers[split]) >= args.shard_size:
                            parts[split] = save_shard(folder, split, parts[split], buffers[split])
                    if accepted % 5000 == 0:
                        print(f'  Scanned {scanned:,}, accepted {accepted:,}, '
                              f'positions {counts["total"]:,}/{args.positions:,}', flush=True)

            if counts['total'] >= args.positions or scanned >= args.max_scanned_games:
                break

    for split in ('train', 'val'):
        parts[split] = save_shard(folder, split, parts[split], buffers[split])
    if counts['total'] != args.positions or min(counts['train'], counts['val']) < 1000:
        raise RuntimeError(f'Could not extract enough positions: {counts["total"]:,} '
                           f'(train {counts["train"]:,}, val {counts["val"]:,}).')

    shard_names = [f'{split}_{i:04d}.npz' for split in ('train', 'val')
                   for i in range(parts[split])]
    stats = {
        **expected,
        'total_positions': counts['total'],
        'train_positions': counts['train'],
        'validation_positions': counts['val'],
        'games_scanned': scanned,
        'games_accepted': accepted,
        'game_selection_counts': dict(rejections),
        'positions_by_game_third': {k: dict(v) for k, v in thirds.items()},
        'average_legal_moves_by_game_third': {
            split: {phase: round(legal_sums[(split, phase)] / thirds[split][phase], 2)
                    if thirds[split][phase] else None for phase in PHASE_NAMES}
            for split in ('train', 'val')
        },
        'game_results': dict(game_results),
        'time_control_categories': dict(time_controls),
        'game_length_plies': {
            'mean': round(float(np.mean(game_lengths)), 1),
            'median': round(float(np.median(game_lengths)), 1),
            'min': min(game_lengths), 'max': max(game_lengths),
        },
        'shards': shard_names,
        'extraction_seconds': round(time.perf_counter() - start, 1),
        'notes': [
            'Game thirds are relative to full game length; not chess-phase annotations.',
            'Split is deterministic by whole game ID (10% validation); repeated opening positions can still appear across games.',
            'Source is an initial section of September, not a random month-wide sample.',
            'Stockfish PGN evaluations ignored; labels are human moves and game results.',
            'Moves with recorded post-move clock below 30 seconds excluded.',
            'Time-forfeit and abandoned games excluded by game metadata.',
            'October dataset reserved for independent testing.',
        ],
    }
    metadata.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(f'Extracted {counts["train"]:,} train / {counts["val"]:,} validation '
          f'positions in {stats["extraction_seconds"]}s.', flush=True)
    for split in ('train', 'val'):
        print(f'  {split} thirds: ' + ', '.join(f'{p}={thirds[split][p]:,}'
                                               for p in PHASE_NAMES), flush=True)
    return stats


def make_loader(shard, batch_size, shuffle):
    with np.load(shard) as data:
        boards = torch.from_numpy(data['boards'].astype(np.float32))
        actions = torch.from_numpy(data['actions'].astype(np.int64))
        outcomes = torch.from_numpy(data['outcomes'].astype(np.float32))
        thirds = torch.from_numpy(data['thirds'].astype(np.int64))
    return DataLoader(TensorDataset(boards, actions, outcomes, thirds),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_epoch(model, shards, batch_size, device, optimizer=None, seed=0):
    training = optimizer is not None
    model.train(training)
    files = list(shards)
    if training:
        random.Random(seed).shuffle(files)
    totals = Counter()
    correct_thirds = np.zeros(3, dtype=np.int64)
    count_thirds = np.zeros(3, dtype=np.int64)
    start = time.perf_counter()
    for shard in files:
        loader = make_loader(shard, batch_size, shuffle=training)
        for boards, actions, outcomes, phases in loader:
            boards = boards.to(device)
            actions = actions.to(device)
            outcomes = outcomes.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                logits, values = model(boards)
                policy_loss = nn.functional.cross_entropy(logits, actions)
                value_loss = nn.functional.mse_loss(values, outcomes)
                loss = policy_loss + 0.25 * value_loss
                if training:
                    loss.backward()
                    optimizer.step()
            n = len(actions)
            totals['positions'] += n
            totals['loss'] += loss.item() * n
            totals['policy_loss'] += policy_loss.item() * n
            totals['value_mse'] += value_loss.item() * n
            matches = (logits.argmax(dim=1) == actions).detach().cpu().numpy()
            totals['correct'] += int(matches.sum())
            phase_np = phases.numpy()
            count_thirds += np.bincount(phase_np, minlength=3)
            correct_thirds += np.bincount(phase_np, weights=matches, minlength=3).astype(np.int64)
        del loader
    seconds = time.perf_counter() - start
    n = totals['positions']
    if not n:
        raise RuntimeError('Empty training or validation dataset.')
    result = {
        'loss': totals['loss'] / n,
        'policy_loss': totals['policy_loss'] / n,
        'value_mse': totals['value_mse'] / n,
        'accuracy': totals['correct'] / n,
        'positions': n,
        'seconds': round(seconds, 2),
        'positions_per_second': round(n / max(seconds, 1e-6), 1),
    }
    for i, label in enumerate(PHASE_NAMES):
        result[f'{label}_accuracy'] = (float(correct_thirds[i] / count_thirds[i])
                                      if count_thirds[i] else float('nan'))
        result[f'{label}_count'] = int(count_thirds[i])
    return result


def measure_baselines(train_shards, val_shards):
    frequencies = Counter()
    for shard in train_shards:
        with np.load(shard) as data:
            frequencies.update(data['actions'].tolist())
    common_action = frequencies.most_common(1)[0][0]
    matched = count = 0
    chance = 0.0
    by_third = [Counter() for _ in range(3)]
    for shard in val_shards:
        with np.load(shard) as data:
            actions = data['actions']
            legal_counts = data['legal_counts']
            phases = data['thirds']
            matches = actions == common_action
            random_chance = 1.0 / legal_counts
            matched += int(matches.sum())
            chance += float(random_chance.sum())
            count += len(actions)
            for i in range(3):
                take = phases == i
                by_third[i]['n'] += int(take.sum())
                by_third[i]['common_correct'] += int(matches[take].sum())
                by_third[i]['random_sum'] += float(random_chance[take].sum())
    out = {
        'random_legal_accuracy': chance / count,
        'most_common_action_accuracy': matched / count,
        'most_common_action': int(common_action),
    }
    for i, label in enumerate(PHASE_NAMES):
        n = by_third[i]['n']
        out[f'{label}_random_accuracy'] = by_third[i]['random_sum'] / n if n else 0.
        out[f'{label}_most_common_accuracy'] = by_third[i]['common_correct'] / n if n else 0.
    return out


def benchmark_target_network(device, batch_size=64, steps=6):
    """Measure 10x128 training throughput on synthetic inputs, before dataset training."""
    print('Benchmarking proposed 10-block x 128-channel network...', flush=True)
    if device.type != 'cuda':
        print('  CUDA unavailable, skipping large-model GPU benchmark.', flush=True)
        return {'skipped': True, 'reason': 'No CUDA'}
    model = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = ChessNetwork(channels=128, blocks=10).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        boards = torch.rand(batch_size, 61, 8, 8, device=device)
        actions = torch.randint(4672, (batch_size,), device=device)
        outcomes = torch.rand(batch_size, device=device) * 2 - 1
        start = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            logits, values = model(boards)
            loss = nn.functional.cross_entropy(logits, actions) + 0.25 * nn.functional.mse_loss(values, outcomes)
            loss.backward()
            opt.step()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        speed = round(steps * batch_size / seconds, 1)
        print(f'  10x128: {speed:,.0f} positions/sec, peak allocated GPU memory {peak_gb:.2f} GiB '
              f'(synthetic, batch {batch_size}).', flush=True)
        return {'channels': 128, 'blocks': 10, 'batch_size': batch_size,
                'steps': steps, 'positions_per_second': speed,
                'peak_allocated_gpu_gib': round(peak_gb, 2), 'synthetic': True}
    except torch.cuda.OutOfMemoryError:
        print('  10x128 benchmark ran out of GPU memory. Main 6x96 training can still run.', flush=True)
        return {'error': 'out_of_memory', 'batch_size': batch_size}
    finally:
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def load_previous_model(device):
    path = PROJECT_ROOT / 'checkpoints' / 'supervised_100000.pt'
    if not path.is_file():
        print('  100k checkpoint not found: skipping previous-model matches.', flush=True)
        return None
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = ChessNetwork(saved['model_channels'], saved['model_blocks']).to(device)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()
    return model


def fixed_openings(seed=7, count=4):
    rng = random.Random(seed)
    result = []
    for _ in range(count):
        board = chess.Board()
        for _ in range(4):
            board.push(rng.choice(list(board.legal_moves)))
        result.append(board.fen())
    return result


def play_matches(model, device, openings=4, max_plies=300):
    """Small diagnostic only: too few games for reliable Elo conclusions."""
    print('Running small diagnostic matches against random and 100k model...', flush=True)
    rivals = [('random_legal', None)]
    previous = load_previous_model(device)
    if previous is not None:
        rivals.append(('supervised_100000', previous))
    results = {}
    for name, rival in rivals:
        results[name] = Counter()
        for opening_i, fen in enumerate(fixed_openings(count=openings)):
            for agent_color in (chess.WHITE, chess.BLACK):
                env = ChessEnvironment(max_plies=max_plies, claim_draw=True)
                env.reset(fen)
                rng = random.Random(9173 + opening_i)
                while True:
                    status = env.status()
                    if status.terminated or status.truncated:
                        break
                    our_turn = env.board.turn == agent_color
                    if our_turn:
                        move = model.choose_move(env.board, device)
                    elif rival is not None:
                        move = rival.choose_move(env.board, device)
                    else:
                        move = rng.choice(list(env.board.legal_moves))
                    env.step(move)
                if status.truncated:
                    results[name]['truncated'] += 1
                elif status.winner is None:
                    results[name]['draw'] += 1
                elif status.winner == agent_color:
                    results[name]['win'] += 1
                else:
                    results[name]['loss'] += 1
        r = results[name]
        print(f'  vs {name}: W={r["win"]} D={r["draw"]} L={r["loss"]} '
              f'truncated={r["truncated"]}', flush=True)
    return {name: dict(counts) for name, counts in results.items()}


def main():
    parser = argparse.ArgumentParser(description='EE641 1M-position supervised experiment')
    parser.add_argument('--source', type=Path, default=DEFAULT_PGN)
    parser.add_argument('--positions', type=int, default=1_000_000)
    parser.add_argument('--positions-per-game', type=int, default=12)
    parser.add_argument('--shard-size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--channels', type=int, default=96)
    parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-scanned-games', type=int, default=4_000_000)
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--skip-matches', action='store_true')
    args = parser.parse_args()
    if min(args.positions, args.positions_per_game, args.shard_size, args.epochs,
           args.channels, args.blocks, args.batch_size, args.max_scanned_games) < 1:
        parser.error('All sizes and counts must be positive.')
    if args.positions < 1000:
        parser.error('Need at least 1,000 positions for this experiment.')
    if not args.source.is_file():
        parser.error(f'Missing September PGN archive at {args.source}')

    out = PROJECT_ROOT / 'outputs'
    checkpoints = PROJECT_ROOT / 'checkpoints'
    folder = PROJECT_ROOT / 'data' / 'processed' / f'supervised_{args.positions}_balanced'
    out.mkdir(exist_ok=True)
    checkpoints.mkdir(exist_ok=True)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints / f'supervised_{args.positions}_balanced.pt'
    if checkpoint_path.is_file() and not args.retrain and not args.rebuild:
        print(f'1M experiment already completed: {checkpoint_path}', flush=True)
        print('Use --retrain to rerun training or --rebuild to regenerate data.', flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'PyTorch: {torch.__version__} | Device: {device}', flush=True)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0), flush=True)
    else:
        raise RuntimeError('CUDA is required for the 1-million-position experiment.')

    architecture_benchmark = benchmark_target_network(device)
    # Keep network initialization repeatable even when the benchmark uses random tensors.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    stats = extract(args, folder)
    train_shards, val_shards = files_for(folder, 'train'), files_for(folder, 'val')
    baselines = measure_baselines(train_shards, val_shards)
    print('Validation baselines: random legal move '
          f'{baselines["random_legal_accuracy"]:.1%}, '
          f'common-action {baselines["most_common_action_accuracy"]:.1%}', flush=True)

    model = ChessNetwork(channels=args.channels, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    csv_path = out / f'supervised_{args.positions}_balanced_metrics.csv'
    progress_path = checkpoints / f'supervised_{args.positions}_balanced_progress.pt'
    rows = []
    first_epoch = 1
    if progress_path.is_file() and not args.retrain and not args.rebuild:
        saved = torch.load(progress_path, map_location='cpu', weights_only=True)
        if (saved['model_channels'] == args.channels
                and saved['model_blocks'] == args.blocks
                and saved['positions'] == args.positions
                and saved['seed'] == args.seed):
            model.load_state_dict(saved['model_state_dict'])
            optimizer.load_state_dict(saved['optimizer_state_dict'])
            # Optimizer states saved on the CPU need to follow model to CUDA.
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
            rows = saved['rows']
            first_epoch = saved['last_epoch'] + 1
            print(f'Resuming supervised training from epoch {first_epoch}.', flush=True)
    print(f'Training {args.blocks} blocks x {args.channels} channels, '
          f'{args.epochs} epochs, batch size {args.batch_size}.', flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        train = run_epoch(model, train_shards, args.batch_size, device,
                          optimizer, args.seed + epoch)
        val = run_epoch(model, val_shards, args.batch_size, device)
        row = {'epoch': epoch,
               **{'train_' + k: v for k, v in train.items()},
               **{'val_' + k: v for k, v in val.items()},
               **baselines}
        rows.append(row)
        print(f'Epoch {epoch}: train accuracy {train["accuracy"]:.1%}, '
              f'validation accuracy {val["accuracy"]:.1%}, '
              f'val loss {val["loss"]:.4f}, '
              f'train speed {train["positions_per_second"]:,.0f} positions/sec', flush=True)
        print('  Validation thirds: ' + ', '.join(
            f'{name} {val[name + "_accuracy"]:.1%}' for name in PHASE_NAMES), flush=True)
        with csv_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_channels': args.channels,
            'model_blocks': args.blocks,
            'positions': args.positions,
            'seed': args.seed,
            'last_epoch': epoch,
            'rows': rows,
        }, progress_path)
        print('  Saved resumable epoch checkpoint.', flush=True)

    # Save the trained weights BEFORE evaluation, so a diagnostic match issue
    # cannot discard an otherwise completed training run.
    final_val = {k[4:]: v for k, v in rows[-1].items() if k.startswith('val_')}
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_channels': args.channels,
        'model_blocks': args.blocks,
        'positions': args.positions,
        'epochs': args.epochs,
        'seed': args.seed,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'last_validation_metrics': final_val,
        'data_stats': stats,
        'baseline_accuracy': baselines,
        'note': '1M-position research pilot; NOT final V0 or PPO.',
    }, checkpoint_path)

    model.eval()
    matches = {} if args.skip_matches else play_matches(model, device)
    match_path = out / f'supervised_{args.positions}_balanced_matches.json'
    match_path.write_text(json.dumps(matches, indent=2), encoding='utf-8')
    benchmark_path = out / f'supervised_{args.positions}_target_arch_benchmark.json'
    benchmark_path.write_text(json.dumps(architecture_benchmark, indent=2), encoding='utf-8')
    progress_path.unlink(missing_ok=True)
    print('Saved model:', checkpoint_path, flush=True)
    print('Saved metrics:', csv_path, flush=True)
    print('Saved match diagnostics:', match_path, flush=True)
    print('Saved architecture benchmark:', benchmark_path, flush=True)
    print('Your 3k and 100k checkpoints are unchanged.', flush=True)


if __name__ == '__main__':
    main()
