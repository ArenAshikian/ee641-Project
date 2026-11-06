"""No-search temperature/entropy experiment for the frozen V0 and V100 policies.

Seven candidate conditions share fresh paired-color opening positions and a
frozen greedy V0 opponent. An independent set of V0-V0 games calibrates the
V0 temperature that matches V100's mean legal-move entropy at temperature 1.
No training, engine evaluations, or tree search.
"""

import argparse
import csv
import json
import os
from collections import Counter

import chess
import numpy as np
import torch

from chess_rl.checkpoint_championship import fresh_openings, position_signature
from chess_rl.encoding import encode_board, legal_actions
from chess_rl.ppo_policy_test import forbidden_openings, bootstrap_ci
from chess_rl.ppo_train import (CHECKPOINTS, OUTPUTS, V0, board_from_opening,
                                one_move, read_historical, sha256)

V100 = CHECKPOINTS / 'ppo_v0100.pt'
RESULTS = OUTPUTS / 'ppo_temperature_games.csv'
MANIFEST = OUTPUTS / 'ppo_temperature_manifest.json'
SUMMARY_CSV = OUTPUTS / 'ppo_temperature_summary.csv'
SUMMARY_JSON = OUTPUTS / 'ppo_temperature_summary.json'
CALIBRATION_JSON = OUTPUTS / 'ppo_temperature_calibration.json'
PREVIOUS_POLICY_MANIFEST = OUTPUTS / 'ppo_policy_comparison_manifest.json'

DEFAULT_PAIRS = 200
DEFAULT_SEED = 750193
CALIBRATION_SEED = 750194
CALIBRATION_GAMES = 32
CALIBRATION_POSITIONS = 400
MAX_PLIES = 240

FIXED = [
    ('v0_t1', 'v0', 1.0),
    ('v0_t07', 'v0', .7),
    ('v0_t05', 'v0', .5),
    ('v100_t1', 'v100', 1.0),
    ('v100_t07', 'v100', .7),
    ('v100_t05', 'v100', .5),
]
MATCHED = 'v0_entropy_matched'
COLUMNS = ['condition', 'candidate', 'temperature', 'opponent', 'opening_id',
           'opening_uci', 'candidate_color', 'score', 'winner', 'termination',
           'truncated', 'plies']


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def save_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def entropy(logits, temperature):
    """Entropy over legal actions only, using stable softmax of scaled logits."""
    if temperature <= 0:
        raise ValueError('Temperature must be positive.')
    a = np.asarray(logits, dtype=np.float64) / float(temperature)
    a -= np.max(a)
    p = np.exp(a)
    p /= np.sum(p)
    return -float(np.sum(p * np.log(np.maximum(p, 1e-300))))


def mean_entropy(logits_by_position, temperature):
    return float(np.mean([entropy(a, temperature) for a in logits_by_position]))


def solve_entropy_temperature(logits_v0, target, lower=.10, upper=2.0):
    """Choose V0 temperature for V100 T=1 mean legal-move entropy."""
    low_entropy = mean_entropy(logits_v0, lower)
    high_entropy = mean_entropy(logits_v0, upper)
    if target <= low_entropy:
        return lower, 'at_lower_bound'
    if target >= high_entropy:
        return upper, 'at_upper_bound'
    lo, hi = lower, upper
    for _ in range(32):
        mid = (lo + hi) / 2
        if mean_entropy(logits_v0, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, 'matched'


def calibration_states(model, device, forbidden):
    """Collect independent middle/late-game states from no-search V0-V0 games."""
    openings = fresh_openings(CALIBRATION_GAMES, CALIBRATION_SEED, forbidden)
    states, signatures = [], set()
    for opening in openings:
        board = board_from_opening(opening['uci'])
        for ply in range(MAX_PLIES):
            if board.is_game_over(claim_draw=True):
                break
            if ply % 4 == 0:
                sig = position_signature(board)
                if sig not in signatures:
                    signatures.add(sig)
                    states.append(board.copy(stack=True))
                    if len(states) >= CALIBRATION_POSITIONS:
                        return states, openings
            # Only a single neural forward pass per move, with legal masking.
            move, _ = one_move(model, board, device, sample=False)
            board.push(move)
    if len(states) < 50:
        raise RuntimeError('Too few calibration states for entropy matching.')
    return states, openings


def legal_logits_on_states(model, states, device):
    """Exactly one forward pass of the model on each calibration position."""
    model.eval()
    logits_by_state = []
    for board in states:
        actions = legal_actions(board)
        if len(actions) < 2:
            continue
        with torch.inference_mode():
            x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
            logits, _ = model(x)
        indices = sorted(actions)
        logits_by_state.append(logits[0, indices].cpu().numpy().astype(np.float64))
    return logits_by_state


def calibrate(v0, v100, device, forbidden):
    print('Calibrating V0 temperature on separate V0-vs-V0 positions...', flush=True)
    states, openings = calibration_states(v0, device, forbidden)
    v0_logits = legal_logits_on_states(v0, states, device)
    v100_logits = legal_logits_on_states(v100, states, device)
    if len(v0_logits) != len(v100_logits):
        raise RuntimeError('Calibration positions are mismatched.')
    target = mean_entropy(v100_logits, 1.0)
    temperature, status = solve_entropy_temperature(v0_logits, target)
    report = {
        'model': 'v0', 'reference_model': 'v100', 'reference_temperature': 1.0,
        'matched_temperature': round(temperature, 6), 'status': status,
        'v0_entropy_t1': round(mean_entropy(v0_logits, 1.0), 6),
        'v100_entropy_t1': round(target, 6),
        'v0_entropy_matched': round(mean_entropy(v0_logits, temperature), 6),
        'v0_entropy_t07': round(mean_entropy(v0_logits, .7), 6),
        'v0_entropy_t05': round(mean_entropy(v0_logits, .5), 6),
        'positions': len(v0_logits), 'calibration_seed': CALIBRATION_SEED,
        'calibration_openings': [x['uci'] for x in openings],
        'note': ('Entropy is calibrated on fixed V0-vs-V0 game positions, '
                 'not on positions arising from each tested sampling policy.'),
    }
    print(f'  {len(v0_logits)} states; entropy V0 T1={report["v0_entropy_t1"]:.4f}, '
          f'V100 T1={report["v100_entropy_t1"]:.4f}', flush=True)
    print(f'  Matching V0 temperature: {report["matched_temperature"]:.4f} '
          f'({status}; V0 entropy={report["v0_entropy_matched"]:.4f})', flush=True)
    return report


def condition_specs(matched_temperature):
    return [
        {'id': name, 'candidate': cand, 'temperature': round(temp, 6)}
        for name, cand, temp in FIXED
    ] + [{'id': MATCHED, 'candidate': 'v0',
          'temperature': round(matched_temperature, 6)}]


def make_manifest(pairs, seed, max_plies, v0, v100, device):
    if not V0.is_file() or not V100.is_file():
        raise FileNotFoundError('Need checkpoints/v0_supervised.pt and '
                                'checkpoints/ppo_v0100.pt.')
    hashes = {'v0': sha256(V0), 'v100': sha256(V100)}
    if MANIFEST.is_file():
        saved = json.loads(MANIFEST.read_text(encoding='utf-8'))
        if (saved.get('schema') != 1 or saved.get('pairs') != pairs or
            saved.get('seed') != seed or saved.get('max_plies') != max_plies or
            saved.get('model_sha256') != hashes or
            len(saved.get('openings', [])) != pairs or
            saved.get('opponent') != 'v0_greedy' or
            len(saved.get('conditions', [])) != len(FIXED) + 1 or
            saved.get('calibration_seed') != CALIBRATION_SEED):
            raise RuntimeError('Saved experiment has different settings or model files. '
                               'Existing results are preserved.')
        if saved['conditions'] != condition_specs(saved['calibration']['matched_temperature']):
            raise RuntimeError('Saved conditions mismatch. Refusing to mix experiments.')
        return saved
    if RESULTS.exists():
        raise RuntimeError('Game records exist without manifest. Refusing to overwrite.')
    forbidden = forbidden_openings()
    if PREVIOUS_POLICY_MANIFEST.is_file():
        previous = json.loads(PREVIOUS_POLICY_MANIFEST.read_text(encoding='utf-8'))
        for row in previous.get('openings', []):
            forbidden.add(position_signature(board_from_opening(row['uci'])))
    calibration = calibrate(v0, v100, device, forbidden)
    for uci in calibration['calibration_openings']:
        forbidden.add(position_signature(board_from_opening(uci)))
    print('Generating fresh paired openings for all temperature conditions...', flush=True)
    openings = fresh_openings(pairs, seed, forbidden)
    saved = {
        'schema': 1, 'pairs': pairs, 'seed': seed, 'max_plies': max_plies,
        'calibration_seed': CALIBRATION_SEED, 'conditions': condition_specs(calibration['matched_temperature']),
        'model_sha256': hashes, 'opponent': 'v0_greedy', 'openings': openings,
        'calibration': calibration, 'search_used': False,
        'note': ('All conditions use the same starting positions and swapped colors. '
                 'Opening states exclude known experiment starting boards where possible. '
                 'This cannot exclude every historical intermediate game position.'),
    }
    save_json(MANIFEST, saved)
    save_json(CALIBRATION_JSON, calibration)
    return saved


def temperature_move(model, board, device, temperature):
    """Sample one legal move from one NN forward pass; never search."""
    if temperature <= 0:
        raise ValueError('Temperature must be positive.')
    actions = legal_actions(board)
    if not actions:
        raise ValueError('No legal moves in this position.')
    ids = sorted(actions)
    x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits, _ = model(x)
        # Selecting among only legal moves is equivalent to a masked softmax.
        sampled = torch.distributions.Categorical(logits=logits[0, ids] / temperature).sample()
    return actions[ids[int(sampled.item())]]


def play_game(candidate, opponent, opening, candidate_color, device, max_plies, seed, temp):
    board = board_from_opening(opening)
    torch.manual_seed(seed)
    plies = 0
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                score, winner = .5, 'draw'
            else:
                score = 1.0 if outcome.winner == candidate_color else 0.0
                winner = 'white' if outcome.winner == chess.WHITE else 'black'
            return {'score': score, 'winner': winner,
                    'termination': outcome.termination.name,
                    'truncated': False, 'plies': plies}
        if plies >= max_plies:
            return {'score': None, 'winner': None,
                    'termination': 'MAX_PLIES', 'truncated': True, 'plies': plies}
        if board.turn == candidate_color:
            move = temperature_move(candidate, board, device, temp)
        else:
            move, _ = one_move(opponent, board, device, sample=False)
        board.push(move)
        plies += 1


def read_games(manifest):
    saved = {}
    if not RESULTS.exists():
        return saved
    conditions = {c['id']: c for c in manifest['conditions']}
    with RESULTS.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise RuntimeError('CSV columns differ from this experiment.')
        for row in reader:
            cond = conditions.get(row['condition'])
            oid, color = int(row['opening_id']), row['candidate_color']
            key = row['condition'], oid, color
            if (cond is None or row['candidate'] != cond['candidate'] or
                row['temperature'] != f'{cond["temperature"]:.6f}' or
                row['opponent'] != 'v0_greedy' or not 0 <= oid < manifest['pairs'] or
                color not in ('white', 'black') or
                row['opening_uci'] != ' '.join(manifest['openings'][oid]['uci']) or
                row['truncated'] not in ('0', '1') or
                row['score'] not in ('0.0', '0.5', '1.0', '') or
                (row['truncated'] == '1') != (row['score'] == '') or
                key in saved):
                raise RuntimeError(f'Inconsistent saved result: {key}')
            saved[key] = row
    return saved


def pair_scores(saved, condition, pairs):
    out = {}
    for i in range(pairs):
        a = saved.get((condition, i, 'white'))
        b = saved.get((condition, i, 'black'))
        if (a is not None and b is not None and
            a['truncated'] == '0' and b['truncated'] == '0'):
            out[i] = (float(a['score']) + float(b['score'])) / 2
    return out


def describe_condition(games, condition, pairs, seed):
    tally = Counter()
    for (name, _, _), row in games.items():
        if name != condition['id']:
            continue
        tally['played'] += 1
        if row['truncated'] == '1':
            tally['truncated'] += 1
        elif row['score'] == '1.0':
            tally['wins'] += 1
        elif row['score'] == '0.5':
            tally['draws'] += 1
        else:
            tally['losses'] += 1
    n = tally['wins'] + tally['draws'] + tally['losses']
    scores = pair_scores(games, condition['id'], pairs)
    return {
        'condition': condition['id'], 'candidate': condition['candidate'],
        'temperature': condition['temperature'], 'games': tally['played'],
        'wins': tally['wins'], 'draws': tally['draws'],
        'losses': tally['losses'], 'truncated': tally['truncated'],
        'game_score': round((tally['wins'] + .5 * tally['draws']) / n, 4) if n else None,
        'paired_openings': len(scores),
        'paired_score': round(float(np.mean(list(scores.values()))), 4) if scores else None,
        'paired_bootstrap_95pct': bootstrap_ci(list(scores.values()), seed + len(condition['id'])),
    }


def compare(games, pairs, seed):
    comparisons = [
        ('v100_t1_minus_v0_t1', 'v100_t1', 'v0_t1'),
        ('v100_t1_minus_v0_matched_entropy', 'v100_t1', MATCHED),
        ('v100_t07_minus_v0_t07', 'v100_t07', 'v0_t07'),
        ('v100_t05_minus_v0_t05', 'v100_t05', 'v0_t05'),
        ('v0_t07_minus_v0_t1', 'v0_t07', 'v0_t1'),
        ('v0_t05_minus_v0_t1', 'v0_t05', 'v0_t1'),
        ('v100_t07_minus_v100_t1', 'v100_t07', 'v100_t1'),
        ('v100_t05_minus_v100_t1', 'v100_t05', 'v100_t1'),
    ]
    pools = {name: pair_scores(games, name, pairs) for name in
             {name for _, a, b in comparisons for name in (a, b)}}
    out = {}
    for k, (label, a, b) in enumerate(comparisons):
        common = sorted(set(pools[a]) & set(pools[b]))
        if not common:
            continue
        differences = np.array([pools[a][i] - pools[b][i] for i in common])
        out[label] = {
            'opening_pairs': len(common),
            'score_difference': round(float(differences.mean()), 4),
            'paired_bootstrap_95pct': bootstrap_ci(differences, seed + k * 201),
        }
    return out


def write_summary(manifest, games):
    rows = [describe_condition(games, cond, manifest['pairs'], manifest['seed'])
            for cond in manifest['conditions']]
    comparisons = compare(games, manifest['pairs'], manifest['seed'])
    result = {
        'method': 'Search-free temperature sampling against fixed greedy V0',
        'games_per_condition': manifest['pairs'] * 2,
        'calibration': manifest['calibration'],
        'conditions': rows, 'comparisons': comparisons,
        'caveats': [
            'Entropy matching is computed on independent calibration states, not match trajectories.',
            'A temperature match does not by itself prove better top-move ranking or tactics.',
            'Game-result comparisons are paired by starting position with reversed colors.',
            'Truncated games are excluded from paired comparisons and not counted as draws.',
        ],
    }
    save_json(SUMMARY_JSON, result)
    table = [{key: row[key] for key in
              ['condition', 'candidate', 'temperature', 'games', 'wins',
               'draws', 'losses', 'truncated', 'game_score', 'paired_openings', 'paired_score']}
             | {'ci95_low': row['paired_bootstrap_95pct'][0] if row['paired_bootstrap_95pct'] else '',
                'ci95_high': row['paired_bootstrap_95pct'][1] if row['paired_bootstrap_95pct'] else ''}
             for row in rows]
    save_csv(SUMMARY_CSV, table, list(table[0]))
    return result


def run(pairs=DEFAULT_PAIRS, seed=DEFAULT_SEED, max_plies=MAX_PLIES):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'V0/V100 temperature evaluation | device: {device} | NO SEARCH', flush=True)
    v0 = read_historical(V0, device).eval()
    v100 = read_historical(V100, device).eval()
    manifest = make_manifest(pairs, seed, max_plies, v0, v100, device)
    if not CALIBRATION_JSON.is_file():
        save_json(CALIBRATION_JSON, manifest['calibration'])
    games = read_games(manifest)
    total = len(manifest['conditions']) * pairs * 2
    print(f'Resuming/starting experiment: {len(games)}/{total} games saved.', flush=True)
    models = {'v0': v0, 'v100': v100}
    for cond in manifest['conditions']:
        name = cond['id']
        remaining = [(i, color) for i in range(pairs) for color in ('white', 'black')
                     if (name, i, color) not in games]
        if not remaining:
            print(f'{name}: already complete.', flush=True)
            continue
        print(f'{name}: T={cond["temperature"]:.4f}, {len(remaining)} games remaining.', flush=True)
        with RESULTS.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if f.tell() == 0:
                writer.writeheader()
            for i, color in remaining:
                uci = manifest['openings'][i]['uci']
                is_white = color == 'white'
                game_seed = seed + 101 * i + (1 if is_white else 0)
                outcome = play_game(models[cond['candidate']], v0, uci, is_white,
                                    device, max_plies, game_seed, cond['temperature'])
                row = {
                    'condition': name, 'candidate': cond['candidate'],
                    'temperature': f'{cond["temperature"]:.6f}',
                    'opponent': 'v0_greedy', 'opening_id': i,
                    'opening_uci': ' '.join(uci), 'candidate_color': color,
                    'score': '' if outcome['truncated'] else str(float(outcome['score'])),
                    'winner': outcome['winner'] or '', 'termination': outcome['termination'],
                    'truncated': int(outcome['truncated']), 'plies': outcome['plies'],
                }
                writer.writerow(row)
                f.flush()
                games[(name, i, color)] = {k: str(v) for k, v in row.items()}
                done = sum(key[0] == name for key in games)
                if done % 40 == 0 or done == pairs * 2:
                    print(f'  {name}: {done}/{pairs * 2} games saved', flush=True)
        result = write_summary(manifest, games)
        summary = next(x for x in result['conditions'] if x['condition'] == name)
        print(f'  {name}: W={summary["wins"]} D={summary["draws"]} '
              f'L={summary["losses"]} truncated={summary["truncated"]}; '
              f'score={summary["game_score"]}; CI={summary["paired_bootstrap_95pct"]}', flush=True)
    result = write_summary(manifest, games)
    print('Matched score differences (positive favors first model):', flush=True)
    for name, row in result['comparisons'].items():
        print(f'  {name}: {row["score_difference"]:+.4f}; '
              f'CI={row["paired_bootstrap_95pct"]}; pairs={row["opening_pairs"]}', flush=True)
    print('Saved summary:', SUMMARY_CSV, flush=True)
    print('Saved calibration:', CALIBRATION_JSON, flush=True)
    print('Saved games:', RESULTS, flush=True)
    print('No model or training files were modified.', flush=True)


def main():
    parser = argparse.ArgumentParser(description='No-search PPO sampling temperature test')
    parser.add_argument('--pairs', type=int, default=DEFAULT_PAIRS)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--max-plies', type=int, default=MAX_PLIES)
    args = parser.parse_args()
    if args.pairs < 5 or args.seed < 0 or args.max_plies < 4:
        parser.error('Need >=5 pairs, seed >=0, max-plies >=4.')
    run(args.pairs, args.seed, args.max_plies)


if __name__ == '__main__':
    main()
