"""Read-only PPO plateau diagnostics. Chess moves use one neural pass; no search.

Compares sampled-game outcomes with value predictions and compares V0/V100
policy distributions on identical encountered legal positions.
No model weights, training histories, or checkpoints are modified.
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import chess
import numpy as np
import torch

from chess_rl.network import ChessNetwork
from chess_rl.ppo_train import (CHECKPOINTS, OUTPUTS, V0, load_supervised,
                                opening_sequences, read_historical, rollout_game,
                                sha256)

V100 = CHECKPOINTS / 'ppo_v0100.pt'
REPORT = OUTPUTS / 'ppo_plateau_diagnostics.json'


def regression_summary(predictions, outcomes):
    """Check whether a value head improves on a constant outcome prediction."""
    x = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if len(x) != len(y) or not len(x):
        return None
    mse = float(np.mean((x - y)**2))
    zero_mse = float(np.mean(y**2))
    constant_mse = float(np.mean((y - y.mean())**2))
    correlation = (float(np.corrcoef(x, y)[0, 1])
                   if len(x) > 2 and np.std(x) > 1e-8 and np.std(y) > 1e-8
                   else None)
    return {
        'positions': len(x),
        'value_mse_vs_terminal_result': round(mse, 5),
        'zero_prediction_mse': round(zero_mse, 5),
        'best_constant_mse': round(constant_mse, 5),
        'value_r2_vs_constant': round(1 - mse / constant_mse, 4)
        if constant_mse > 1e-12 else None,
        'value_outcome_correlation': round(correlation, 4)
        if correlation is not None else None,
        'mean_predicted_value': round(float(x.mean()), 4),
        'mean_terminal_result': round(float(y.mean()), 4),
        'mean_absolute_error': round(float(np.mean(np.abs(x-y))), 4),
        'mean_value_given_win': round(float(x[y == 1].mean()), 4)
        if np.any(y == 1) else None,
        'mean_value_given_draw': round(float(x[y == 0].mean()), 4)
        if np.any(y == 0) else None,
        'mean_value_given_loss': round(float(x[y == -1].mean()), 4)
        if np.any(y == -1) else None,
    }


def scored_game_summary(results):
    counts = Counter()
    for result in results:
        if result['truncated']:
            counts['truncated'] += 1
        else:
            score = result['outcome_for_learner']
            counts['win' if score > 0 else 'loss' if score < 0 else 'draw'] += 1
    done = counts['win'] + counts['draw'] + counts['loss']
    return {
        'games': len(results), 'wins': counts['win'], 'draws': counts['draw'],
        'losses': counts['loss'], 'truncated': counts['truncated'],
        'score_on_completed_games': round((counts['win'] + 0.5*counts['draw'])/done, 4)
        if done else None,
    }


def collect(model, opponent, openings, device, max_plies, seed,
            max_policy_states=512):
    """Evaluate *sampled* play, matching how PPO gathers experience."""
    model.eval()
    opponent.eval()
    results, values, targets = [], [], []
    per_color = {'white': ([], []), 'black': ([], [])}
    states = []
    rng = random.Random(seed)
    for j, opening in enumerate(openings):
        for color in (chess.WHITE, chess.BLACK):
            # Same deterministic RNG convention for each candidate.
            torch.manual_seed(seed + j*101 + int(color))
            transitions, result = rollout_game(model, opponent, opening, color,
                                               device, max_plies)
            results.append(result)
            if not result['truncated']:
                target = float(result['outcome_for_learner'])
                for row in transitions:
                    values.append(float(row['value']))
                    targets.append(target)
                    per_color['white' if color else 'black'][0].append(float(row['value']))
                    per_color['white' if color else 'black'][1].append(target)
            # Sample a few positions from every game rather than filling from game 1.
            sample = list(transitions)
            rng.shuffle(sample)
            for row in sample[:8]:
                if len(states) < max_policy_states:
                    states.append((row['board'], row['mask']))
    return {
        'game_results': scored_game_summary(results),
        'value_fit': regression_summary(values, targets),
        'value_fit_white': regression_summary(*per_color['white']),
        'value_fit_black': regression_summary(*per_color['black']),
        'value_note': ('Outcome targets are +1/0/-1 from learner perspective, repeated for '
                       'each learner decision in a completed game. Positions within a game '
                       'are correlated. Truncated games are omitted from value-fit metrics.'),
    }, states


@torch.no_grad()
def compare_policies(v0, v100, states, device, batch_size=32):
    """Evaluate BOTH frozen policies on the same board states and legal masks."""
    if not states:
        return None
    v0.eval()
    v100.eval()
    n = 0
    agreement_sum = 0
    entropy_0 = 0.
    entropy_100 = 0.
    kl_100_0 = 0.
    v_difference = 0.
    for begin in range(0, len(states), batch_size):
        batch = states[begin:begin+batch_size]
        x = torch.from_numpy(np.stack([s[0] for s in batch]).astype(np.float32)).to(device)
        legal = torch.from_numpy(np.stack([s[1] for s in batch])).to(device)
        l0, values0 = v0(x)
        l100, values100 = v100(x)
        l0 = l0.masked_fill(~legal, -1e9)
        l100 = l100.masked_fill(~legal, -1e9)
        log0 = torch.log_softmax(l0, dim=1)
        log100 = torch.log_softmax(l100, dim=1)
        p0 = log0.exp()
        p100 = log100.exp()
        # Avoid undefined 0 * -infinity on illegal actions.
        delta = torch.where(legal, log100 - log0, 0.)
        agreement_sum += int((l0.argmax(dim=1) == l100.argmax(dim=1)).sum())
        entropy_0 += float((-(p0 * torch.where(legal, log0, 0.)).sum(dim=1)).sum())
        entropy_100 += float((-(p100 * torch.where(legal, log100, 0.)).sum(dim=1)).sum())
        kl_100_0 += float(((p100 * delta).sum(dim=1)).sum())
        v_difference += float((values100 - values0).abs().sum())
        n += len(batch)
    return {
        'positions': n,
        'greedy_move_agreement_rate': round(agreement_sum/n, 4),
        'v0_legal_policy_entropy': round(entropy_0/n, 4),
        'v100_legal_policy_entropy': round(entropy_100/n, 4),
        'kl_v100_vs_v0_on_same_states': round(kl_100_0/n, 4),
        'mean_absolute_value_difference': round(v_difference/n, 4),
        'note': ('Policy comparisons use identical encoded positions and legal masks, '
                 'from sampled games. No engine or search is involved.'),
    }


def summarize_existing_training():
    file = OUTPUTS / 'ppo_long_metrics.csv'
    if not file.exists():
        return {'note': 'Long-training metrics CSV not found.'}
    with file.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {'note': 'Long-training metrics CSV is empty.'}
    def total(field):
        return sum(float(row.get(field, 0) or 0) for row in rows)
    g = total('games')
    result = {
        'iterations_logged': len(rows),
        'total_games_in_long_run': int(g),
        'total_learner_moves_in_long_run': int(total('transitions')),
        'total_optimizer_steps_in_long_run': int(total('optimizer_steps')),
        'recent_opponent_share': round(total('recent_opponent_games')/g, 4) if g else None,
        'historical_opponent_share': round(total('historical_opponent_games')/g, 4) if g else None,
        'warning': ('Loss, entropy, clip fraction, and approximate KL in this CSV '
                    'are from the final minibatch only, not iteration-wide means. '
                    'Per-historical-opponent game results were not stored.'),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='Search-free PPO plateau investigation')
    parser.add_argument('--opening-pairs', type=int, default=32)
    parser.add_argument('--max-plies', type=int, default=240)
    parser.add_argument('--seed', type=int, default=90831)
    args = parser.parse_args()
    if args.opening_pairs < 4 or args.max_plies < 4 or args.seed < 0:
        parser.error('At least 4 opening pairs, max plies >=4, and nonnegative seed required.')
    if not V0.exists() or not V100.exists():
        parser.error('Both v0_supervised.pt and ppo_v0100.pt are required.')
    OUTPUTS.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Read-only PPO plateau diagnosis | device:', device, '| NO SEARCH', flush=True)
    print('Reading original long-training metrics...', flush=True)
    training = summarize_existing_training()
    v0 = load_supervised(V0, device)
    v100 = read_historical(V100, device)
    openings = opening_sequences(args.opening_pairs, args.seed)
    result = {
        'experiment': 'PPO plateau diagnosis; NO SEARCH',
        'checkpoint_sha256': {'v0': sha256(V0), 'v100': sha256(V100)},
        'opening_pairs_per_model': args.opening_pairs,
        'training': training,
        'sampled_games': {},
        'policies_on_identical_states': {},
        'notes': [
            'This is a diagnostic, not a randomized trial proving the cause of a plateau.',
            'Sampled games match PPO collection, while championship uses greedy argmax play.',
            'Model weights and optimizer state are never changed.',
        ],
    }
    all_states = []
    for name, model in (('v0', v0), ('v100', v100)):
        print(f'Collecting {2*args.opening_pairs} sampled games: {name} vs frozen V0...', flush=True)
        summary, states = collect(model, v0, openings, device, args.max_plies, args.seed)
        result['sampled_games'][name] = summary
        all_states.extend(states[:256])
        scores = summary['game_results']
        value = summary['value_fit']
        print(f'  W={scores["wins"]} D={scores["draws"]} L={scores["losses"]} '
              f'truncated={scores["truncated"]}; score={scores["score_on_completed_games"]}', flush=True)
        if value:
            print(f'  Value MSE={value["value_mse_vs_terminal_result"]:.3f}; '
                  f'constant baseline={value["best_constant_mse"]:.3f}; '
                  f'R2={value["value_r2_vs_constant"]}', flush=True)
    print('Comparing V0/V100 on the same legal positions...', flush=True)
    result['policies_on_identical_states'] = compare_policies(v0, v100, all_states, device)
    agreement = result['policies_on_identical_states']
    print('  Greedy move agreement:', agreement['greedy_move_agreement_rate'],
          '| entropy V0/V100:', agreement['v0_legal_policy_entropy'],
          agreement['v100_legal_policy_entropy'], flush=True)
    REPORT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved report:', REPORT, flush=True)
    print('No checkpoints or training files were modified.', flush=True)


if __name__ == '__main__':
    main()
