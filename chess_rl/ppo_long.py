"""Long, search-free PPO experiment continuing from the archived V10.

The original ppo_train_latest.pt and all earlier models stay untouched.
No engine, search, chess analysis, or best-move labels enter training or play.
"""

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone

import chess
import numpy as np
import torch

from chess_rl.network import ChessNetwork
from chess_rl.ppo_train import (CHECKPOINTS, OUTPUTS, V0, V1M,
                                atomic_save, board_from_opening, load_supervised,
                                one_move, opening_sequences, ppo_update,
                                read_pool, read_historical, rollout_game, sha256)

START = CHECKPOINTS / 'ppo_v0010.pt'
PROGRESS = CHECKPOINTS / 'ppo_long_latest.pt'
METRICS = OUTPUTS / 'ppo_long_metrics.csv'
EVAL_SEED = 42021
EVAL_OPENINGS = 50


def settings(start_hash, v0_hash, games, seed, max_plies, ppo_epochs, batch_size, lr):
    return {
        'start_iteration': 10,
        'start_v10_sha256': start_hash,
        'v0_sha256': v0_hash,
        'games_per_iteration': games,
        'seed': seed,
        'max_plies': max_plies,
        'ppo_epochs': ppo_epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'clip': 0.1,
        'gamma': 1.0,
        'gae_lambda': 0.95,
        'value_weight': 0.5,
        'entropy_weight': 0.01,
        'opponent_mix': '50% recent frozen, 50% uniform archived V0/V10/...',
        'evaluation_seed': EVAL_SEED,
        'evaluation_openings': EVAL_OPENINGS,
    }


def check_start(saved):
    """Reject a wrong archive rather than silently training the wrong model."""
    if int(saved.get('iteration', -1)) != 10:
        raise RuntimeError('ppo_v0010.pt does not contain PPO iteration 10.')
    if int(saved.get('model_blocks', -1)) != 10 or int(saved.get('model_channels', -1)) != 128:
        raise RuntimeError('V10 model architecture differs from expected 10x128.')
    if 'model_state_dict' not in saved or 'optimizer_state_dict' not in saved:
        raise RuntimeError('V10 archive is missing model or optimizer state.')


def restore_optimizer(model, saved, device, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    opt.load_state_dict(saved['optimizer_state_dict'])
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    for group in opt.param_groups:
        group['lr'] = lr
    return opt


def safe_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def checkpoint_for(version):
    if version == 10:
        return START
    return CHECKPOINTS / f'ppo_v{version:04d}.pt'


def result_score(winner, learner_color):
    if winner is None:
        return 0.5
    return 1.0 if winner == learner_color else 0.0


def play_eval_game(candidate, opponent, opening, color, device, max_plies, seed):
    """Single network pass per decision. No search and no artificial draw."""
    board = board_from_opening(opening)
    rng = random.Random(seed)
    plies = 0
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return {
                'score': result_score(outcome.winner, color),
                'winner': ('white' if outcome.winner == chess.WHITE else
                           'black' if outcome.winner == chess.BLACK else 'draw'),
                'termination': outcome.termination.name,
                'plies': plies,
                'truncated': False,
            }
        if plies >= max_plies:
            return {'score': None, 'winner': None,
                    'termination': 'MAX_PLIES', 'plies': plies, 'truncated': True}
        if board.turn == color:
            move, _ = one_move(candidate, board, device, sample=False)
        elif opponent is None:
            move = rng.choice(list(board.legal_moves))
        else:
            move, _ = one_move(opponent, board, device, sample=False)
        board.push(move)
        plies += 1


def paired_interval(pairs, seed):
    """Bootstrap paired opening scores. Remove pairs with any truncated game."""
    values = np.array([v for v in pairs if len(v) == 2 and None not in v], dtype=np.float64)
    if len(values) < 5:
        return None
    per_opening = values.mean(axis=1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(per_opening), size=(2000, len(per_opening)))
    means = per_opening[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [round(float(lo), 4), round(float(hi), 4)]



def refresh_evaluation_summary():
    """One small CSV makes V10, V20, ..., V100 easy to compare."""
    rows = []
    for path in sorted(OUTPUTS.glob('ppo_long_eval_v????.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        for opponent, result in report['opponents'].items():
            ci = result['paired_bootstrap_95pct_score_interval']
            rows.append({
                'version': report['version'], 'opponent': opponent,
                'wins': result['wins'], 'draws': result['draws'],
                'losses': result['losses'], 'truncated': result['truncated'],
                'score_completed': result['score_on_completed_games'],
                'ci95_low': ci[0] if ci else '',
                'ci95_high': ci[1] if ci else '',
            })
    safe_csv(rows, OUTPUTS / 'ppo_long_evaluation_summary.csv')

def evaluate_version(version, device, max_plies, model=None):
    """Evaluate against identical frozen opponents and color-swapped openings.

    A single set of openings is used at EVERY version. Evaluation is not PPO
    training data. All 4 opponents are search-free.
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    archive = checkpoint_for(version)
    if not archive.is_file():
        raise FileNotFoundError(f'Missing model archive to evaluate: {archive}')
    out_json = OUTPUTS / f'ppo_long_eval_v{version:04d}.json'
    out_csv = OUTPUTS / f'ppo_long_games_v{version:04d}.csv'
    if out_json.exists() and out_csv.exists():
        previous = json.loads(out_json.read_text(encoding='utf-8'))
        if (previous.get('checkpoint_sha256') != sha256(archive) or
                previous.get('max_plies') != max_plies or
                previous.get('eval_seed') != EVAL_SEED):
            raise RuntimeError(f'V{version} archived evaluation differs from current settings or weights.')
        print(f'V{version} evaluation already saved; skipping.', flush=True)
        refresh_evaluation_summary()
        return previous
    if model is None:
        candidate = read_historical(archive, device)
    else:
        candidate = model
    candidate.eval()
    opponents = [('v0', V0), ('v10', START), ('random_legal', None)]
    if V1M.exists():
        opponents.append(('supervised_1m', V1M))
    openings = opening_sequences(EVAL_OPENINGS, EVAL_SEED)
    details = []
    summary = {
        'version': version,
        'checkpoint': archive.name,
        'checkpoint_sha256': sha256(archive),
        'eval_seed': EVAL_SEED,
        'opening_pairs': EVAL_OPENINGS,
        'games_per_opponent': 2 * EVAL_OPENINGS,
        'max_plies': max_plies,
        'search_used': False,
        'opponents': {},
    }
    for label, path in opponents:
        opp = read_historical(path, device) if path is not None else None
        tally = Counter()
        pairs = []
        reasons = Counter()
        for j, opening in enumerate(openings):
            scores = []
            for color in (chess.WHITE, chess.BLACK):
                result = play_eval_game(candidate, opp, opening, color, device,
                                        max_plies, EVAL_SEED + j * 97 + int(color))
                scores.append(result['score'])
                tally['games'] += 1
                reasons[result['termination']] += 1
                if result['truncated']:
                    tally['truncated'] += 1
                elif result['score'] == 1:
                    tally['wins'] += 1
                elif result['score'] == 0:
                    tally['losses'] += 1
                else:
                    tally['draws'] += 1
                details.append({
                    'version': version, 'opponent': label,
                    'opening_pair': j, 'opening_uci': ' '.join(opening),
                    'learner_color': 'white' if color == chess.WHITE else 'black',
                    'score': result['score'] if result['score'] is not None else '',
                    'winner': result['winner'] or '', 'termination': result['termination'],
                    'truncated': int(result['truncated']), 'plies': result['plies'],
                })
            pairs.append(scores)
        complete = tally['wins'] + tally['draws'] + tally['losses']
        score = ((tally['wins'] + 0.5 * tally['draws']) / complete) if complete else None
        summary['opponents'][label] = {
            'wins': tally['wins'], 'draws': tally['draws'], 'losses': tally['losses'],
            'truncated': tally['truncated'], 'games': tally['games'],
            'score_on_completed_games': round(score, 4) if score is not None else None,
            'paired_bootstrap_95pct_score_interval': paired_interval(pairs, EVAL_SEED + version),
            'termination_reasons': dict(reasons),
        }
        print(f'  V{version} vs {label}: W={tally["wins"]} D={tally["draws"]} '
              f'L={tally["losses"]} truncated={tally["truncated"]}; '
              f'score={score:.3f}' if score is not None else
              f'  V{version} vs {label}: no completed games', flush=True)
        del opp
    safe_csv(details, out_csv)
    temp = out_json.with_suffix('.tmp')
    temp.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    os.replace(temp, out_json)
    refresh_evaluation_summary()
    return summary


def missing_evaluations(device, max_plies, through):
    """Retry pending evaluations after an interruption, never rerun completed ones."""
    for v in range(10, through + 1, 10):
        archive = checkpoint_for(v)
        if archive.is_file():
            evaluate_version(v, device, max_plies)



def recover_v10():
    """Old PPO runs usually archive V10; recover it if only latest remains."""
    if START.is_file():
        return
    old_latest = CHECKPOINTS / 'ppo_train_latest.pt'
    if old_latest.is_file():
        saved = torch.load(old_latest, map_location='cpu', weights_only=True)
        if int(saved.get('iteration', -1)) == 10:
            check_start(saved)
            atomic_save(saved, START)
            print('Recovered permanent V10 from original PPO latest checkpoint.', flush=True)

def extend_training(target, games_per_iteration, seed, max_plies,
                    ppo_epochs, batch_size, lr, device):
    if target < 20 or target % 10:
        raise ValueError('Target must be at least 20 and a multiple of 10 (e.g., 100).')
    recover_v10()
    if not START.is_file():
        raise FileNotFoundError(f'Missing V10 archive: {START}. '
                                'Check whether ppo_train_latest.pt is at iteration 10.')
    if not V0.is_file():
        raise FileNotFoundError(f'Missing original V0 checkpoint: {V0}')
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    base = torch.load(START, map_location='cpu', weights_only=True)
    check_start(base)
    current_v0_hash = sha256(V0)
    if base.get('config', {}).get('v0_sha256') != current_v0_hash:
        raise RuntimeError('The V0 checkpoint differs from the one used for V10.')
    cfg = settings(sha256(START), current_v0_hash, games_per_iteration, seed,
                   max_plies, ppo_epochs, batch_size, lr)
    if PROGRESS.exists():
        saved = torch.load(PROGRESS, map_location='cpu', weights_only=True)
        if saved['config'] != cfg:
            raise RuntimeError('Long-run settings changed. Existing progress is preserved.')
        if int(saved['iteration']) < 11 or int(saved['iteration']) > target:
            raise RuntimeError('Unexpected long-run checkpoint iteration.')
        model = ChessNetwork(channels=int(saved['model_channels']),
                             blocks=int(saved['model_blocks'])).to(device)
        model.load_state_dict(saved['model_state_dict'])
        optimizer = restore_optimizer(model, saved, device, lr)
        first = int(saved['iteration']) + 1
        history = saved['history']
        if saved['iteration'] % 10 == 0:
            archive = checkpoint_for(saved['iteration'])
            if not archive.is_file():
                atomic_save(saved, archive)
        print(f'Resuming long PPO at iteration {first}.', flush=True)
    else:
        model = ChessNetwork(channels=128, blocks=10).to(device)
        model.load_state_dict(base['model_state_dict'])
        optimizer = restore_optimizer(model, base, device, lr)
        first, history = 11, []
        print('Starting long PPO from permanent V10, keeping V0 and V10 untouched.', flush=True)
    safe_csv(history, METRICS)
    # Evaluation never contributes to PPO rewards. Verify baseline at V10.
    missing_evaluations(device, max_plies, min(target, first - 1))
    if first > target:
        print(f'Already reached V{target}; nothing to retrain.', flush=True)
        return
    for i in range(first, target + 1):
        t0 = time.perf_counter()
        rng = random.Random(seed + i * 9127)
        torch.manual_seed(seed + i * 167)
        snapshot, historical = read_pool(model, 'train', i, device)
        historical_models = {name: read_historical(file, device) for name, file in historical}
        openings = opening_sequences(games_per_iteration, seed + i * 17)
        experiences = []
        counts = Counter()
        lengths = []
        for game_idx, opening in enumerate(openings):
            if rng.random() < 0.5:
                name, rival = 'recent_frozen', snapshot
            else:
                name = rng.choice(list(historical_models))
                rival = historical_models[name]
            color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
            transitions, result = rollout_game(snapshot, rival, opening, color,
                                                device, max_plies)
            experiences.extend(transitions)
            counts[name] += 1
            counts['games'] += 1
            if result['truncated']:
                counts['truncated'] += 1
            else:
                outcome = result['outcome_for_learner']
                counts['wins' if outcome > 0 else 'losses' if outcome < 0 else 'draws'] += 1
            lengths.append(result['plies'])
        del historical_models, snapshot
        metrics = ppo_update(model, optimizer, experiences, device, ppo_epochs,
                             batch_size, clip=0.1, value_weight=0.5,
                             entropy_weight=0.01, seed=seed + i)
        row = {
            'iteration': i, 'games': counts['games'],
            'wins': counts['wins'], 'draws': counts['draws'], 'losses': counts['losses'],
            'truncated': counts['truncated'], 'transitions': len(experiences),
            'mean_plies': round(float(np.mean(lengths)), 1),
            'recent_opponent_games': counts['recent_frozen'],
            'historical_opponent_games': counts['games'] - counts['recent_frozen'],
            **{key: round(val, 5) if isinstance(val, float) else val
               for key, val in metrics.items()},
            'elapsed_seconds': round(time.perf_counter() - t0, 1),
        }
        history.append(row)
        entry = {
            'iteration': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_blocks': 10, 'model_channels': 128,
            'config': cfg, 'history': history,
            'description': 'Extended search-free PPO from archived V10.',
            'saved_utc': datetime.now(timezone.utc).isoformat(),
        }
        if i % 10 == 0 and checkpoint_for(i).exists():
            raise FileExistsError(f'Refusing to overwrite permanent checkpoint V{i}.')
        atomic_save(entry, PROGRESS)
        if i % 10 == 0:
            atomic_save(entry, checkpoint_for(i))
        safe_csv(history, METRICS)
        print(f'PPO long {i}/{target}: W={counts["wins"]} D={counts["draws"]} '
              f'L={counts["losses"]} truncated={counts["truncated"]}; '
              f'{len(experiences)} learner moves; '
              f'policy_loss={metrics["policy_loss"]:.4f}, '
              f'value_loss={metrics["value_loss"]:.4f}, '
              f'KL={metrics["approx_kl"]:.4f}; '
              f'{row["elapsed_seconds"]:.0f}s', flush=True)
        if i % 10 == 0:
            print(f'Evaluating permanent V{i} against frozen opponents...', flush=True)
            evaluate_version(i, device, max_plies, model=model)
    print('Long PPO training complete. Last permanent archive:', checkpoint_for(target), flush=True)
    print('Long-run metrics:', METRICS, flush=True)


def main():
    p = argparse.ArgumentParser(description='Search-free extended PPO from V10')
    p.add_argument('--target', type=int, default=100)
    p.add_argument('--games', type=int, default=128)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--max-plies', type=int, default=240)
    p.add_argument('--ppo-epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--eval-only', action='store_true')
    args = p.parse_args()
    if args.games < 2 or args.games % 2 or args.max_plies < 4 or args.ppo_epochs < 1 or args.batch_size < 2:
        p.error('Games must be even and >=2; max plies >=4; PPO epochs >=1; batch size >=2.')
    if args.seed < 0 or args.lr <= 0:
        p.error('Seed must be nonnegative and learning rate positive.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, '| one-pass neural chess, NO SEARCH or engine.', flush=True)
    if args.eval_only:
        recover_v10()
        if not START.exists():
            p.error(f'Missing {START}')
        if PROGRESS.exists():
            saved = torch.load(PROGRESS, map_location='cpu', weights_only=True)
            latest = int(saved['iteration'])
        else:
            latest = 10
        latest_archive = latest - latest % 10
        print('Evaluating archived checkpoints through V' + str(latest_archive), flush=True)
        missing_evaluations(device, args.max_plies, latest_archive)
    else:
        extend_training(args.target, args.games, args.seed, args.max_plies,
                        args.ppo_epochs, args.batch_size, args.lr, device)


if __name__ == '__main__':
    main()
