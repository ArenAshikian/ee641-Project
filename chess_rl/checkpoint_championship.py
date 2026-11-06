"""Search-free checkpoint championship for EE 641.

Five 400-game head-to-head matches on new, paired-color opening positions.
This evaluates frozen neural policies. It never trains, searches, or uses an engine.
"""

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import chess
import numpy as np
import torch

from chess_rl.ppo_train import (CHECKPOINTS, OUTPUTS, V0, board_from_opening,
                                opening_sequences, read_historical, sha256)
from chess_rl.ppo_long import play_eval_game

MATCHUPS = [(100, 0), (100, 10), (100, 30), (100, 50), (30, 50)]
SEED = 983257
OPENING_PAIRS = 200
MAX_PLIES = 240
RESULTS = OUTPUTS / 'championship_games.csv'
MANIFEST = OUTPUTS / 'championship_manifest.json'
SUMMARY_CSV = OUTPUTS / 'championship_summary.csv'
SUMMARY_JSON = OUTPUTS / 'championship_summary.json'
COLUMNS = ['matchup', 'candidate', 'opponent', 'opening_id', 'opening_uci',
           'candidate_color', 'score', 'winner', 'termination', 'truncated', 'plies']


def checkpoint(version):
    if version == 0:
        return V0
    return CHECKPOINTS / f'ppo_v{version:04d}.pt'


def match_name(candidate, opponent):
    return f'V{candidate}_vs_V{opponent}'


def position_signature(board):
    """Board position, side, castling and en passant, without move counters."""
    return " ".join(board.fen().split()[:4])


def past_opening_fens():
    """Exclude previous PPO training/evaluation starting boards, not whole games.

    These are the default seeds/counts from the completed V1-V100 experiment.
    This does not assert the positions were never encountered later in a game.
    """
    used = set()
    sources = [(32, 7 + i * 17) for i in range(1, 11)]
    sources += [(128, 7 + i * 17) for i in range(11, 101)]
    sources += [(50, 42021), (20, 42), (4, 42)]
    for count, seed in sources:
        for opening in opening_sequences(count, seed):
            used.add(position_signature(board_from_opening(opening)))
    return used


def fresh_openings(count, seed, forbidden):
    """New 10/12/14-ply random-legal opening boards, with reproducible seed.

    No Stockfish, heuristic, tree search, or expert book is involved.
    Each selected board is unique, not present among prior PPO starting boards.
    """
    rng = random.Random(seed)
    first = ['e2e4', 'd2d4', 'c2c4', 'g1f3']
    openings, seen = [], set(forbidden)
    attempts = 0
    while len(openings) < count:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError('Could not generate enough fresh legal opening positions.')
        board = chess.Board()
        board.push_uci(first[(attempts - 1) % len(first)])
        depth = (10, 12, 14)[(attempts - 1) % 3]
        for _ in range(depth - 1):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over(claim_draw=True):
            continue
        fen = board.fen()
        signature = position_signature(board)
        if signature in seen:
            continue
        seen.add(signature)
        openings.append({'uci': [move.uci() for move in board.move_stack], 'fen': fen})
    return openings


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temp, path)


def write_csv(path, data, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)
    os.replace(tmp, path)


def make_manifest(pairs, seed, max_plies):
    required_versions = sorted({v for pair in MATCHUPS for v in pair})
    hashes = {}
    for version in required_versions:
        file = checkpoint(version)
        if not file.is_file():
            raise FileNotFoundError(f'Missing checkpoint: {file}')
        hashes[str(version)] = sha256(file)
    if MANIFEST.exists():
        saved = json.loads(MANIFEST.read_text(encoding='utf-8'))
        if (saved.get('schema') != 1 or saved.get('pairs') != pairs or
            saved.get('seed') != seed or saved.get('max_plies') != max_plies or
            saved.get('matchups') != [list(m) for m in MATCHUPS] or
            saved.get('model_sha256') != hashes or
            len(saved.get('openings', [])) != pairs):
            raise RuntimeError('Existing championship uses different settings or model files. '
                               'Results are preserved; do not mix experiments.')
        return saved
    if RESULTS.exists():
        raise RuntimeError('Results CSV exists without its manifest. '
                           'Refusing to mix or overwrite game records.')
    print('Generating held-out paired opening positions...', flush=True)
    openings = fresh_openings(pairs, seed, past_opening_fens())
    manifest = {
        'schema': 1, 'seed': seed, 'pairs': pairs, 'max_plies': max_plies,
        'matchups': [list(x) for x in MATCHUPS], 'model_sha256': hashes,
        'openings': openings, 'search_used': False,
        'opening_note': ('New 10/12/14-ply random-legal starting positions, disjoint by '
                         'board/turn/castling/en-passant state from past PPO training and '
                         'fixed-evaluation starts (not necessarily all game states).'),
    }
    write_json(MANIFEST, manifest)
    return manifest


def read_games(manifest):
    existing = {}
    if not RESULTS.exists():
        return existing
    with RESULTS.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise RuntimeError('Championship CSV columns differ from expected format.')
        for row in reader:
            game_key = (row['matchup'], int(row['opening_id']), row['candidate_color'])
            if game_key in existing:
                raise RuntimeError(f'Duplicate saved game: {game_key}')
            candidate = int(row['candidate'])
            opponent = int(row['opponent'])
            oid = int(row['opening_id'])
            if ((candidate, opponent) not in MATCHUPS or
                    row['matchup'] != match_name(candidate, opponent) or
                    not 0 <= oid < manifest['pairs'] or
                    row['candidate_color'] not in ('white', 'black') or
                    row['opening_uci'] != ' '.join(manifest['openings'][oid]['uci']) or
                    row['truncated'] not in ('0', '1') or
                    (row['score'] not in ('0.0', '0.5', '1.0', '') or
                     (row['truncated'] == '1') != (row['score'] == ''))):
                raise RuntimeError(f'Corrupt or mismatched saved game: {game_key}')
            existing[game_key] = row
    return existing


def paired_ci(opening_scores, seed):
    """95% percentile bootstrap resampling complete two-color opening pairs."""
    scores = np.asarray(opening_scores, dtype=np.float64)
    if scores.size < 5:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(scores), size=(4000, len(scores)))
    lo, hi = np.quantile(scores[indices].mean(axis=1), [0.025, 0.975])
    return [round(float(lo), 4), round(float(hi), 4)]


def summarize_match(candidate, opponent, records, pairs, seed):
    label = match_name(candidate, opponent)
    tallies = Counter()
    reasons = Counter()
    scores = {}
    for (match, oid, color), row in records.items():
        if match != label:
            continue
        tallies['played'] += 1
        reasons[row['termination']] += 1
        if row['truncated'] == '1':
            tallies['truncated'] += 1
        else:
            s = float(row['score'])
            tallies['wins' if s == 1 else 'losses' if s == 0 else 'draws'] += 1
            scores.setdefault(oid, {})[color] = s
    complete = tallies['wins'] + tallies['draws'] + tallies['losses']
    fully_paired = [np.mean([d['white'], d['black']])
                    for _, d in sorted(scores.items())
                    if 'white' in d and 'black' in d]
    score = (tallies['wins'] + 0.5 * tallies['draws']) / complete if complete else None
    paired_score = float(np.mean(fully_paired)) if fully_paired else None
    ci = paired_ci(fully_paired, seed + candidate * 1009 + opponent)
    return {
        'matchup': label, 'candidate': candidate, 'opponent': opponent,
        'expected_games': pairs * 2, 'played': tallies['played'],
        'wins': tallies['wins'], 'draws': tallies['draws'],
        'losses': tallies['losses'], 'truncated': tallies['truncated'],
        'completed_game_score': round(score, 4) if score is not None else None,
        'fully_completed_pairs': len(fully_paired),
        'paired_score': round(paired_score, 4) if paired_score is not None else None,
        'paired_bootstrap_95pct': ci,
        'termination_reasons': dict(reasons),
    }


def save_summary(manifest, records):
    rows = [summarize_match(a, b, records, manifest['pairs'], manifest['seed'])
            for a, b in MATCHUPS]
    write_json(SUMMARY_JSON, {'method': 'search-free, paired colors, deterministic argmax',
                              'pairs_per_matchup': manifest['pairs'],
                              'max_plies': manifest['max_plies'],
                              'matchups': rows})
    output = []
    for row in rows:
        ci = row['paired_bootstrap_95pct']
        output.append({k: row[k] for k in (
            'matchup', 'played', 'expected_games', 'wins', 'draws', 'losses', 'truncated',
            'completed_game_score', 'fully_completed_pairs', 'paired_score')} |
            {'ci95_low': ci[0] if ci else '', 'ci95_high': ci[1] if ci else ''})
    write_csv(SUMMARY_CSV, output, list(output[0]))
    return rows


def run_championship(pairs=OPENING_PAIRS, seed=SEED, max_plies=MAX_PLIES):
    manifest = make_manifest(pairs, seed, max_plies)
    completed = read_games(manifest)
    total = len(MATCHUPS) * pairs * 2
    print(f'Resuming/starting championship: {len(completed)}/{total} games saved.', flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | 5 matchups, {pairs * 2} games each, NO SEARCH.', flush=True)
    # Persist a complete results record immediately after each game.
    for candidate, opponent in MATCHUPS:
        label = match_name(candidate, opponent)
        remaining = [(oid, color) for oid in range(pairs)
                     for color in ('white', 'black')
                     if (label, oid, color) not in completed]
        if not remaining:
            print(f'{label}: already complete.', flush=True)
            continue
        print(f'{label}: {len(remaining)} games remaining.', flush=True)
        model_a = read_historical(checkpoint(candidate), device)
        model_b = read_historical(checkpoint(opponent), device)
        model_a.eval()
        model_b.eval()
        with RESULTS.open('a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            if file.tell() == 0:
                writer.writeheader()
            for oid, color in remaining:
                opening = manifest['openings'][oid]['uci']
                chess_color = chess.WHITE if color == 'white' else chess.BLACK
                result = play_eval_game(model_a, model_b, opening, chess_color,
                                        device, max_plies, seed + oid * 97 + int(chess_color))
                row = {
                    'matchup': label, 'candidate': candidate, 'opponent': opponent,
                    'opening_id': oid, 'opening_uci': ' '.join(opening),
                    'candidate_color': color,
                    'score': '' if result['truncated'] else str(float(result['score'])),
                    'winner': result['winner'] or '', 'termination': result['termination'],
                    'truncated': int(result['truncated']), 'plies': result['plies'],
                }
                writer.writerow(row)
                file.flush()
                completed[(label, oid, color)] = {k: str(v) for k, v in row.items()}
                done = sum(k[0] == label for k in completed)
                if done % 40 == 0 or done == pairs * 2:
                    print(f'  {label}: {done}/{pairs * 2} games saved', flush=True)
        del model_a, model_b
        summary = save_summary(manifest, completed)
        s = next(row for row in summary if row['matchup'] == label)
        print(f'  {label}: W={s["wins"]} D={s["draws"]} L={s["losses"]} '
              f'truncated={s["truncated"]}; '
              f'score={s["completed_game_score"]}; '
              f'paired 95% CI={s["paired_bootstrap_95pct"]}', flush=True)
    save_summary(manifest, completed)
    print('Championship finished. Results:', SUMMARY_CSV, flush=True)
    print('All individual games:', RESULTS, flush=True)


def main():
    parser = argparse.ArgumentParser(description='No-search frozen checkpoint championship')
    parser.add_argument('--pairs', type=int, default=OPENING_PAIRS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--max-plies', type=int, default=MAX_PLIES)
    args = parser.parse_args()
    if args.pairs < 5 or args.seed < 0 or args.max_plies < 4:
        parser.error('Need at least 5 opening pairs, nonnegative seed and max-plies >= 4.')
    run_championship(args.pairs, args.seed, args.max_plies)


if __name__ == '__main__':
    main()
