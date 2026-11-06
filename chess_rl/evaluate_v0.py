"""Reproducible matches for the September supervised chess checkpoints.

The neural policies each choose a move with ONE forward pass, no search.
An optional shallow Stockfish opponent is used for evaluation only.
No engine move or evaluation is ever used as a training target.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch

from chess_rl.environment import ChessEnvironment
from chess_rl.network import ChessNetwork

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'outputs'
CKPT_DIR = ROOT / 'checkpoints'
CANDIDATE = CKPT_DIR / 'supervised_1000000_balanced.pt'
PREVIOUS = CKPT_DIR / 'supervised_100000.pt'
FIELDNAMES = ['opponent', 'opponent_sha256', 'candidate_sha256', 'opening_index',
              'opening_fen', 'candidate_color', 'random_seed', 'plies_played',
              'game_result', 'termination', 'terminated', 'truncated', 'candidate_score']


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def load_policy(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    blocks = int(saved['model_blocks'])
    channels = int(saved['model_channels'])
    model = ChessNetwork(channels=channels, blocks=blocks).to(device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval()
    return model, {'blocks': blocks, 'channels': channels,
                   'sha256': checksum(path), 'path': str(path)}


def opening_positions(count, seed):
    """Deterministic, unique shallow openings; no engine or opening book used."""
    rng = random.Random(seed)
    first_moves = ['e2e4', 'd2d4', 'g1f3', 'c2c4']
    positions = []
    seen = set()
    attempts = 0
    while len(positions) < count and attempts < count * 80:
        attempts += 1
        board = chess.Board()
        board.push_uci(first_moves[(attempts - 1) % len(first_moves)])
        for _ in range(3 + 2 * (len(positions) % 3)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over(claim_draw=True):
            continue
        fen = board.fen()
        if fen not in seen:
            seen.add(fen)
            positions.append(fen)
    if len(positions) != count:
        raise RuntimeError(f'Only generated {len(positions)} unique openings.')
    return positions


def match_seed(seed, opponent, opening_i, candidate_color):
    value = f'{seed}:{opponent}:{opening_i}:{"white" if candidate_color else "black"}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'big')


def play_game(model, opponent, name, fen, color, seed, device, max_plies=300):
    env = ChessEnvironment(max_plies=max_plies, claim_draw=True)
    env.reset(fen)
    rng = random.Random(seed)
    while True:
        status = env.status()
        if status.terminated or status.truncated:
            break
        if env.board.turn == color:
            move = model.choose_move(env.board, device)
        elif name == 'random_legal':
            move = rng.choice(list(env.board.legal_moves))
        elif name == 'stockfish_depth1':
            move = opponent.play(env.board, chess.engine.Limit(depth=1)).move
        else:
            move = opponent.choose_move(env.board, device)
        env.step(move)
    if status.truncated:
        score = None   # Truncation is NOT a real draw.
    elif status.winner is None:
        score = 0.5
    else:
        score = 1.0 if status.winner == color else 0.0
    return status, env.plies, score


def paired_bootstrap_ci(rows, seed=17, draws=3000):
    """Resample whole opening pairs to avoid pretending color-swaps are independent."""
    by_opening = defaultdict(list)
    for row in rows:
        if row['candidate_score'] != '':
            by_opening[int(row['opening_index'])].append(float(row['candidate_score']))
    pairs = [scores for scores in by_opening.values() if len(scores) == 2]
    if len(pairs) < 5:
        return None
    arr = np.asarray(pairs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(draws)
    for i in range(draws):
        samples = arr[rng.integers(0, len(arr), size=len(arr))]
        sampled_means[i] = samples.mean()
    low, high = np.quantile(sampled_means, [0.025, 0.975])
    return [round(float(low), 4), round(float(high), 4)]


def summary(rows, seed):
    counts = Counter()
    scored = []
    for row in rows:
        if row['truncated'] == 'True':
            counts['truncated'] += 1
        elif row['candidate_score'] != '':
            result = float(row['candidate_score'])
            scored.append(result)
            counts[{1.0: 'wins', 0.5: 'draws', 0.0: 'losses'}[result]] += 1
        else:
            counts['incomplete'] += 1
    completed = len(scored)
    return {
        'games': len(rows),
        'completed_games': completed,
        'wins': counts['wins'],
        'draws': counts['draws'],
        'losses': counts['losses'],
        'truncated': counts['truncated'],
        'mean_score_completed_only': round(sum(scored) / completed, 4) if completed else None,
        'paired_opening_bootstrap_95_percent_ci': paired_bootstrap_ci(rows, seed),
        'note': 'Score excludes truncated games; bootstrap CI uses only complete color-swapped opening pairs.',
    }


def find_stockfish(path=None):
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f'Stockfish path does not exist: {p}')
        return str(p)
    configured = os.environ.get('STOCKFISH_PATH')
    if configured and Path(configured).is_file():
        return configured
    return shutil.which('stockfish')


def run_evaluation(openings=50, seed=7, max_plies=300, stockfish=None, output_dir=OUTPUT_DIR):
    if openings < 1 or max_plies < 1:
        raise ValueError('Opening count and max plies must be positive.')
    if not CANDIDATE.is_file():
        raise FileNotFoundError(f'Missing model: {CANDIDATE}. Run train-1m first.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use the installed CUDA-enabled PyTorch.')
    device = torch.device('cuda')
    model, candidate_info = load_policy(CANDIDATE, device)
    print('Candidate:', CANDIDATE.name, candidate_info['sha256'][:12],
          f'({candidate_info["blocks"]}x{candidate_info["channels"]})', flush=True)
    rivals = [('random_legal', None, 'none')]
    if PREVIOUS.is_file():
        prev_model, prev_info = load_policy(PREVIOUS, device)
        rivals.append(('supervised_100000', prev_model, prev_info['sha256']))
        print('Opponent:', PREVIOUS.name, prev_info['sha256'][:12],
              f'({prev_info["blocks"]}x{prev_info["channels"]})', flush=True)
    else:
        print('100k checkpoint missing; only random and optional engine will be evaluated.', flush=True)

    engine_path = find_stockfish(stockfish)
    engine = None
    if engine_path:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        rivals.append(('stockfish_depth1', engine, 'engine:' + engine_path))
        print('Search baseline: Stockfish, depth=1, evaluation ONLY.', flush=True)
    else:
        print('Stockfish not found: skipping optional search baseline. No engine install required.', flush=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / 'v0_fixed_opponent_games.csv'
    output_json = output_dir / 'v0_fixed_opponent_summary.json'
    fens = opening_positions(openings, seed)
    expected = {(name, i, side) for name, _, _ in rivals
                for i in range(openings) for side in ('white', 'black')}
    prior = {}
    try:
        if output_csv.exists():
            with output_csv.open(newline='', encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    identity = (row['opponent'], int(row['opening_index']), row['candidate_color'])
                    if identity not in expected:
                        continue
                    name, i, side = identity
                    other_sha = next((digest for n, _, digest in rivals if n == name), None)
                    if (row['candidate_sha256'] != candidate_info['sha256']
                            or row['opponent_sha256'] != other_sha
                            or row['opening_fen'] != fens[i]
                            or int(row['random_seed']) != match_seed(seed, name, i, side == 'white')):
                        raise RuntimeError('Existing match CSV has incompatible settings or model weights. '
                                           'Rename it before a new evaluation, or restore the original settings.')
                    prior[identity] = row
            if prior:
                print(f'Resuming: {len(prior)} games already saved.', flush=True)

        with output_csv.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
            if output_csv.stat().st_size == 0:
                writer.writeheader()
            for name, rival, rival_hash in rivals:
                print(f'Evaluating against {name}, {openings * 2} paired-color games...', flush=True)
                for i, fen in enumerate(fens):
                    for color in (chess.WHITE, chess.BLACK):
                        side = 'white' if color else 'black'
                        if (name, i, side) in prior:
                            continue
                        rng_seed = match_seed(seed, name, i, color)
                        status, plies, score = play_game(model, rival, name, fen, color,
                                                         rng_seed, device, max_plies)
                        row = {
                            'opponent': name,
                            'opponent_sha256': rival_hash,
                            'candidate_sha256': candidate_info['sha256'],
                            'opening_index': i,
                            'opening_fen': fen,
                            'candidate_color': side,
                            'random_seed': rng_seed,
                            'plies_played': plies,
                            'game_result': status.result,
                            'termination': status.termination,
                            'terminated': status.terminated,
                            'truncated': status.truncated,
                            'candidate_score': '' if score is None else score,
                        }
                        writer.writerow(row)
                        stream.flush()  # interrupted evaluation can resume.
                        prior[(name, i, side)] = {k: str(v) for k, v in row.items()}
                    if (i + 1) % 10 == 0:
                        print(f'  {i + 1}/{openings} opening pairs complete', flush=True)
        summaries = {}
        for name, _, _ in rivals:
            rows = [row for (n, _, _), row in prior.items() if n == name]
            summaries[name] = summary(rows, seed)
            result = summaries[name]
            print(f'  {name}: W={result["wins"]}, D={result["draws"]}, '
                  f'L={result["losses"]}, truncated={result["truncated"]}; '
                  f'completed-game score={result["mean_score_completed_only"]}', flush=True)
        payload = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'candidate': candidate_info,
            'opening_count': openings,
            'games_per_opponent': 2 * openings,
            'seed': seed,
            'max_plies_after_opening': max_plies,
            'opening_method': 'deterministic shallow randomized openings, with both colors',
            'search_baseline_policy': 'Stockfish depth one, if available, evaluation only; no training labels.',
            'limitations': ['Not a tournament Elo rating.',
                            'Randomized opening positions are not a representative opening book.',
                            'Truncated games are excluded from completed-game scores.',
                            'Confidence intervals resample paired-color openings.'],
            'opponents': summaries,
        }
        output_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print('Saved games:', output_csv, flush=True)
        print('Saved summary:', output_json, flush=True)
        return payload
    finally:
        if engine is not None:
            engine.quit()


def main():
    parser = argparse.ArgumentParser(description='EE641 supervised checkpoint match evaluation')
    parser.add_argument('--openings', type=int, default=50)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-plies', type=int, default=300)
    parser.add_argument('--stockfish', type=Path, default=None,
                        help='Optional path to Stockfish executable for evaluation ONLY')
    args = parser.parse_args()
    run_evaluation(args.openings, args.seed, args.max_plies, args.stockfish)


if __name__ == '__main__':
    main()
