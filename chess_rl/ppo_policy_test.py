"""Matched greedy-versus-sampled chess evaluation, without search or training.

Four candidates/selection modes each play the same paired-color openings
against an unchanged greedy V0. Only one neural forward pass selects each move.
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

from chess_rl.checkpoint_championship import (fresh_openings, past_opening_fens,
                                              position_signature)
from chess_rl.ppo_train import (OUTPUTS, V0, CHECKPOINTS, board_from_opening,
                                one_move, read_historical, sha256)

V100 = CHECKPOINTS / 'ppo_v0100.pt'
RESULTS = OUTPUTS / 'ppo_policy_comparison_games.csv'
SUMMARY_CSV = OUTPUTS / 'ppo_policy_comparison_summary.csv'
SUMMARY_JSON = OUTPUTS / 'ppo_policy_comparison_summary.json'
MANIFEST = OUTPUTS / 'ppo_policy_comparison_manifest.json'
CHAMPIONSHIP_MANIFEST = OUTPUTS / 'championship_manifest.json'

CONDITIONS = [('v0', 'greedy'), ('v100', 'greedy'),
              ('v0', 'sampled'), ('v100', 'sampled')]
DEFAULT_PAIRS = 200
DEFAULT_SEED = 615841
MAX_PLIES = 240
COLUMNS = ['condition', 'candidate', 'mode', 'opponent', 'opening_id',
           'opening_uci', 'candidate_color', 'score', 'winner',
           'termination', 'truncated', 'plies']


def label(candidate, mode):
    return f'{candidate}_{mode}'


def save_json(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(contents, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def save_csv(path, contents, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(contents)
    os.replace(temporary, path)


def forbidden_openings():
    """Exclude known training/evaluation STARTING positions where possible."""
    forbidden = past_opening_fens()
    if CHAMPIONSHIP_MANIFEST.is_file():
        previous = json.loads(CHAMPIONSHIP_MANIFEST.read_text(encoding='utf-8'))
        for opening in previous['openings']:
            forbidden.add(position_signature(board_from_opening(opening['uci'])))
    return forbidden


def make_manifest(pairs, seed, max_plies):
    if not V0.is_file() or not V100.is_file():
        raise FileNotFoundError('Both checkpoints/v0_supervised.pt and '
                                'checkpoints/ppo_v0100.pt are required.')
    hashes = {'v0': sha256(V0), 'v100': sha256(V100)}
    if MANIFEST.is_file():
        previous = json.loads(MANIFEST.read_text(encoding='utf-8'))
        if (previous.get('schema') != 1 or previous.get('pairs') != pairs or
            previous.get('seed') != seed or previous.get('max_plies') != max_plies or
            previous.get('model_sha256') != hashes or
            previous.get('conditions') != [list(x) for x in CONDITIONS] or
            len(previous.get('openings', [])) != pairs or
            previous.get('opponent') != 'v0_greedy'):
            raise RuntimeError('Saved comparison has different settings or checkpoints. '
                               'Results are preserved; do not mix experiments.')
        return previous
    if RESULTS.exists():
        raise RuntimeError('Game results exist without a manifest. Refusing to overwrite them.')
    print('Generating new opening positions shared by all four conditions...', flush=True)
    openings = fresh_openings(pairs, seed, forbidden_openings())
    manifest = {
        'schema': 1, 'pairs': pairs, 'seed': seed, 'max_plies': max_plies,
        'conditions': [list(x) for x in CONDITIONS], 'opponent': 'v0_greedy',
        'model_sha256': hashes, 'openings': openings, 'search_used': False,
        'note': ('Unique starting positions, distinct from recorded training/evaluation '
                 'starts and checkpoint championship starts. This does not guarantee '
                 'absence from intermediate states or the September training dataset. '
                 'Opponents always play deterministic greedy V0.'),
    }
    save_json(MANIFEST, manifest)
    return manifest


def read_games(manifest):
    saved = {}
    if not RESULTS.exists():
        return saved
    with RESULTS.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise RuntimeError('Unexpected CSV format; refusing to mix game results.')
        for row in reader:
            condition = row['condition']
            candidate, mode = row['candidate'], row['mode']
            oid, color = int(row['opening_id']), row['candidate_color']
            key = (condition, oid, color)
            expected = {(label(c, m), c, m) for c, m in CONDITIONS}
            if ((condition, candidate, mode) not in expected or
                row['opponent'] != 'v0_greedy' or
                not 0 <= oid < manifest['pairs'] or
                color not in ('white', 'black') or
                row['opening_uci'] != ' '.join(manifest['openings'][oid]['uci']) or
                row['truncated'] not in ('0', '1') or
                row['score'] not in ('0.0', '0.5', '1.0', '') or
                (row['truncated'] == '1') != (row['score'] == '') or
                key in saved):
                raise RuntimeError(f'Duplicate/corrupt/mismatched saved game: {key}')
            saved[key] = row
    return saved


def game(candidate, opponent, opening, candidate_color, device, max_plies,
         seed, candidate_mode):
    """Same chess rules for both conditions. Sample only candidate when requested."""
    board = board_from_opening(opening)
    # Per-game RNG reset supports exact restart and common random seeds.
    torch.manual_seed(seed)
    plies = 0
    while True:
        result = board.outcome(claim_draw=True)
        if result is not None:
            if result.winner is None:
                score, winner = 0.5, 'draw'
            else:
                score = 1.0 if result.winner == candidate_color else 0.0
                winner = 'white' if result.winner == chess.WHITE else 'black'
            return {'score': score, 'winner': winner,
                    'termination': result.termination.name,
                    'truncated': False, 'plies': plies}
        if plies >= max_plies:
            return {'score': None, 'winner': None, 'termination': 'MAX_PLIES',
                    'truncated': True, 'plies': plies}
        if board.turn == candidate_color:
            move, _ = one_move(candidate, board, device, sample=candidate_mode == 'sampled')
        else:
            move, _ = one_move(opponent, board, device, sample=False)
        board.push(move)
        plies += 1


def paired_opening_scores(saved, condition, pairs):
    """Only use complete pairs of outcomes, not partial or truncated pairs."""
    result = {}
    for oid in range(pairs):
        white = saved.get((condition, oid, 'white'))
        black = saved.get((condition, oid, 'black'))
        if (white is not None and black is not None and
            white['truncated'] == '0' and black['truncated'] == '0'):
            result[oid] = (float(white['score']) + float(black['score'])) / 2
    return result


def bootstrap_ci(data, seed):
    values = np.asarray(data, dtype=np.float64)
    if len(values) < 5:
        return None
    rng = np.random.default_rng(seed)
    inds = rng.integers(0, len(values), size=(4000, len(values)))
    return [round(float(v), 4) for v in
            np.quantile(values[inds].mean(axis=1), [0.025, 0.975])]


def summarize_condition(saved, candidate, mode, pairs, seed):
    name = label(candidate, mode)
    counters, reasons = Counter(), Counter()
    colors = {'white': Counter(), 'black': Counter()}
    for (condition, _, color), row in saved.items():
        if condition != name:
            continue
        counters['played'] += 1
        reasons[row['termination']] += 1
        if row['truncated'] == '1':
            counters['truncated'] += 1
            colors[color]['truncated'] += 1
        else:
            outcome = 'wins' if row['score'] == '1.0' else 'losses' if row['score'] == '0.0' else 'draws'
            counters[outcome] += 1
            colors[color][outcome] += 1
    n = counters['wins'] + counters['draws'] + counters['losses']
    score = (counters['wins'] + .5 * counters['draws']) / n if n else None
    pairs_scored = paired_opening_scores(saved, name, pairs)
    return {
        'condition': name, 'candidate': candidate, 'move_selection': mode,
        'opponent': 'v0_greedy', 'games_saved': counters['played'],
        'expected_games': pairs * 2, 'wins': counters['wins'], 'draws': counters['draws'],
        'losses': counters['losses'], 'truncated': counters['truncated'],
        'game_score': round(score, 4) if score is not None else None,
        'paired_openings': len(pairs_scored),
        'paired_score': round(float(np.mean(list(pairs_scored.values()))), 4)
        if pairs_scored else None,
        'paired_bootstrap_95pct': bootstrap_ci(list(pairs_scored.values()), seed + (0 if candidate == 'v0' else 100) + (0 if mode == 'greedy' else 10)),
        'by_color': {color: dict(rows) for color, rows in colors.items()},
        'termination_reasons': dict(reasons),
    }


def matched_comparisons(saved, pairs, seed):
    """Compare treatments on *the same opening pairs*, not unrelated games."""
    pairs_by_condition = {label(c, m): paired_opening_scores(saved, label(c, m), pairs)
                          for c, m in CONDITIONS}
    common_ids = sorted(set.intersection(*(set(x) for x in pairs_by_condition.values())))
    if not common_ids:
        return {'common_completed_opening_pairs': 0, 'comparisons': {}}
    scores = {name: np.array([d[i] for i in common_ids], dtype=np.float64)
              for name, d in pairs_by_condition.items()}
    deltas = {
        'greedy_v100_minus_v0': scores['v100_greedy'] - scores['v0_greedy'],
        'sampled_v100_minus_v0': scores['v100_sampled'] - scores['v0_sampled'],
    }
    deltas['sampled_gain_minus_greedy_gain'] = (
        deltas['sampled_v100_minus_v0'] - deltas['greedy_v100_minus_v0'])
    report = {}
    for i, (name, values) in enumerate(deltas.items()):
        report[name] = {
            'mean_score_difference': round(float(values.mean()), 4),
            'paired_bootstrap_95pct': bootstrap_ci(values, seed + 1000 + i),
        }
    return {'common_completed_opening_pairs': len(common_ids),
            'comparisons': report,
            'note': ('Deltas use identical starting positions and reversed colors across '
                     'all four conditions. Resampling unit is the complete opening pair. '
                     'A difference-of-differences above zero indicates a larger V100 gain '
                     'under sampled than under greedy play. These are exploratory results.')}


def save_summary(manifest, games):
    pairs, seed = manifest['pairs'], manifest['seed']
    rows = [summarize_condition(games, c, m, pairs, seed) for c, m in CONDITIONS]
    comparisons = matched_comparisons(games, pairs, seed)
    report = {
        'method': 'No search, 1 network pass per move, greedy V0 opponent',
        'pairs_per_condition': pairs,
        'conditions': rows,
        'matched_comparisons': comparisons,
    }
    save_json(SUMMARY_JSON, report)
    table = []
    for row in rows:
        ci = row['paired_bootstrap_95pct']
        table.append({k: row[k] for k in
                      ('condition', 'games_saved', 'expected_games', 'wins', 'draws',
                       'losses', 'truncated', 'game_score', 'paired_openings', 'paired_score')} |
                     {'ci95_low': ci[0] if ci else '', 'ci95_high': ci[1] if ci else ''})
    save_csv(SUMMARY_CSV, table, list(table[0]))
    return report


def run(pairs=DEFAULT_PAIRS, seed=DEFAULT_SEED, max_plies=MAX_PLIES):
    manifest = make_manifest(pairs, seed, max_plies)
    games = read_games(manifest)
    total = len(CONDITIONS) * pairs * 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Resuming comparison: {len(games)}/{total} games saved, device: {device}, NO SEARCH.', flush=True)
    v0 = read_historical(V0, device).eval()
    v100 = read_historical(V100, device).eval()
    candidate_models = {'v0': v0, 'v100': v100}
    for candidate, mode in CONDITIONS:
        name = label(candidate, mode)
        remaining = [(oid, color) for oid in range(pairs)
                     for color in ('white', 'black')
                     if (name, oid, color) not in games]
        if not remaining:
            print(f'{name}: already complete.', flush=True)
            continue
        print(f'{name}: {len(remaining)} games to play, opponent is greedy V0.', flush=True)
        with RESULTS.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if f.tell() == 0:
                writer.writeheader()
            for oid, color in remaining:
                opening = manifest['openings'][oid]['uci']
                is_white = color == 'white'
                game_seed = seed + oid * 101 + (1 if is_white else 0)
                result = game(candidate_models[candidate], v0, opening, is_white,
                              device, max_plies, game_seed, mode)
                row = {
                    'condition': name, 'candidate': candidate, 'mode': mode,
                    'opponent': 'v0_greedy', 'opening_id': oid,
                    'opening_uci': ' '.join(opening), 'candidate_color': color,
                    'score': '' if result['truncated'] else str(float(result['score'])),
                    'winner': result['winner'] or '', 'termination': result['termination'],
                    'truncated': int(result['truncated']), 'plies': result['plies'],
                }
                writer.writerow(row)
                f.flush()
                games[(name, oid, color)] = {key: str(value) for key, value in row.items()}
                done = sum(k[0] == name for k in games)
                if done % 40 == 0 or done == pairs * 2:
                    print(f'  {name}: {done}/{pairs * 2} games saved', flush=True)
        current = save_summary(manifest, games)
        summary = next(x for x in current['conditions'] if x['condition'] == name)
        print(f'  {name}: W={summary["wins"]} D={summary["draws"]} '
              f'L={summary["losses"]} truncated={summary["truncated"]}; '
              f'score={summary["game_score"]}; CI={summary["paired_bootstrap_95pct"]}', flush=True)
    report = save_summary(manifest, games)
    comparisons = report['matched_comparisons']
    if comparisons['common_completed_opening_pairs'] >= 5:
        print('V100-minus-V0 matched score gains (positive favors PPO):', flush=True)
        for k, row in comparisons['comparisons'].items():
            print(f'  {k}: {row["mean_score_difference"]:+.4f} '
                  f'95% CI={row["paired_bootstrap_95pct"]}', flush=True)
    print('Finished. Summary:', SUMMARY_CSV, flush=True)
    print('Detailed results:', SUMMARY_JSON, flush=True)
    print('Every game:', RESULTS, flush=True)
    print('No checkpoint or training file was modified.', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Search-free PPO greedy/sampled comparison')
    parser.add_argument('--pairs', type=int, default=DEFAULT_PAIRS)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--max-plies', type=int, default=MAX_PLIES)
    args = parser.parse_args()
    if args.pairs < 5 or args.seed < 0 or args.max_plies < 4:
        parser.error('Need at least 5 opening pairs, a nonnegative seed and max-plies >= 4.')
    run(args.pairs, args.seed, args.max_plies)


if __name__ == '__main__':
    main()
