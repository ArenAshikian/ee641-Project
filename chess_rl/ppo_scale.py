"""Controlled continuation from V100: historical-pool vs latest-only PPO.

One neural inference per played move, legal-mask only; no search or engine.
Old model files, old optimizer files, old metrics are strictly read-only.
"""

import argparse
import copy
import csv
import json
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chess
import numpy as np
import torch

from chess_rl.human_openings import (MANIFEST as OPENING_FILE, prepare,
                                      training_openings, validate_manifest)
from chess_rl.network import ChessNetwork
from chess_rl.ppo_long import play_eval_game, restore_optimizer, safe_csv
from chess_rl.ppo_scale_metrics import detailed_ppo_update
from chess_rl.ppo_train import (CHECKPOINTS, OUTPUTS, V0, V1M, atomic_save,
                                read_historical, rollout_game, sha256)

V100 = CHECKPOINTS / 'ppo_v0100.pt'
ORIGINAL_HISTORY = [(f'v{i}', V0 if i == 0 else CHECKPOINTS / f'ppo_v{i:04d}.pt')
                    for i in range(0, 101, 10)]
BRANCHES = ('pool', 'latest')
GAMES = 512
STEPS = 49  # 25,088 additional games per branch
SEED = 641528
MAX_PLIES = 240
PAIRS = 128
EVAL_SEED = 6412027
EVAL_OPPONENTS = [('v0', V0), ('v30', CHECKPOINTS / 'ppo_v0030.pt'), ('v100', V100)]


def branch_dir(branch):
    if branch not in BRANCHES:
        raise ValueError(f'Invalid branch: {branch}')
    return CHECKPOINTS / f'ppo_scale_{branch}'


def archive_file(branch, step):
    return branch_dir(branch) / f's{step:04d}.pt'


def progress_file(branch):
    return branch_dir(branch) / 'latest.pt'


def metrics_file(branch):
    return OUTPUTS / f'ppo_scale_{branch}_metrics.csv'


def opponents_file(branch):
    return OUTPUTS / f'ppo_scale_{branch}_opponents.csv'


def json_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def require_base_checkpoints():
    required = dict(ORIGINAL_HISTORY)
    if not all(path.is_file() for path in required.values()):
        missing = [str(p) for p in required.values() if not p.is_file()]
        raise FileNotFoundError('Missing original archived checkpoints: ' + ', '.join(missing))
    return {label: sha256(path) for label, path in ORIGINAL_HISTORY}


def config(branch, base_hashes, opening_hash, seed, games, max_plies, epochs,
           batch_size, lr):
    return {
        'schema': 1, 'branch': branch, 'initial_checkpoint': 'ppo_v0100.pt',
        'old_checkpoint_hashes': base_hashes, 'openings_sha256': opening_hash,
        'seed': seed, 'games_per_iteration': games, 'max_plies': max_plies,
        'ppo_epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr,
        'clip': .1, 'gae_lambda': .95, 'gamma': 1.0,
        'entropy_weight': .01, 'value_weight': .5,
        'opponent_policy': ('50pct recent+50pct uniform historical' if branch == 'pool'
                            else '100pct recent frozen'),
        'human_start_positions': True, 'no_search': True,
    }


def load_model_and_optimizer(saved, device, lr):
    network = ChessNetwork(channels=int(saved['model_channels']),
                           blocks=int(saved['model_blocks'])).to(device)
    network.load_state_dict(saved['model_state_dict'])
    network.eval()
    optimizer = restore_optimizer(network, saved, device, lr)
    return network, optimizer


def previous_snapshot(branch, step):
    """At most 1 iteration of work is lost when interrupted."""
    progress = progress_file(branch)
    if not progress.is_file():
        return None
    saved = torch.load(progress, map_location='cpu', weights_only=True)
    if int(saved.get('step', -1)) < 1 or int(saved['step']) > step:
        raise RuntimeError('Saved PPO experiment step is invalid.')
    return saved


def choose_rival(branch, rng, snapshot, historical_models):
    if branch == 'latest' or rng.random() < .5:
        return 'recent_frozen', snapshot
    name = rng.choice(list(historical_models))
    return name, historical_models[name]


def history_paths(branch):
    return metrics_file(branch), opponents_file(branch)


def write_branch_logs(history, opponent_history, branch):
    safe_csv(history, metrics_file(branch))
    safe_csv(opponent_history, opponents_file(branch))


def train_branch(branch, opening_manifest, steps=STEPS, games=GAMES,
                 seed=SEED, max_plies=MAX_PLIES, epochs=2,
                 batch_size=64, lr=1e-5, device=None, auto_eval=True,
                 eval_pairs=PAIRS):
    """Continue each branch from identical V100 weights and AdamW state."""
    if branch not in BRANCHES:
        raise ValueError('Branch must be pool or latest.')
    if games % 2 or games < 2 or steps < 1 or batch_size < 2 or epochs < 1:
        raise ValueError('Expected even games >= 2, positive steps/epochs and batch size >= 2.')
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_hashes = require_base_checkpoints()
    cfg = config(branch, base_hashes, opening_manifest['sha256'], seed,
                 games, max_plies, epochs, batch_size, lr)
    initial = torch.load(V100, map_location='cpu', weights_only=True)
    if initial.get('iteration') != 100:
        raise RuntimeError('ppo_v0100.pt does not contain V100.')
    saved = previous_snapshot(branch, steps)
    if saved is None:
        model, optimizer = load_model_and_optimizer(initial, device, lr)
        first = 1
        history, opponent_history = [], []
        print(f'New {branch} branch from unchanged V100, starting at step 1.', flush=True)
    else:
        if saved.get('config') != cfg:
            raise RuntimeError('Resume settings or archived checkpoint hashes changed. '
                               'Original experiment is preserved.')
        model, optimizer = load_model_and_optimizer(saved, device, lr)
        first = int(saved['step']) + 1
        history, opponent_history = saved['history'], saved['opponent_history']
        step = first - 1
        if (step % 10 == 0 or step == STEPS) and not archive_file(branch, step).exists():
            atomic_save(saved, archive_file(branch, step))
        print(f'Resuming {branch} at step {first}, preserving earlier progress.', flush=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    branch_dir(branch).mkdir(parents=True, exist_ok=True)
    write_branch_logs(history, opponent_history, branch)
    if first > steps:
        print(f'{branch} already has {games*(first-1):,} games. No retraining.', flush=True)
        return
    for step in range(first, steps + 1):
        start = time.perf_counter()
        rng = random.Random(seed + step * 9127)
        torch.manual_seed(seed + step * 167)
        snapshot = copy.deepcopy(model).to(device).eval()
        historical_models = {}
        if branch == 'pool':
            paths = ORIGINAL_HISTORY + [(f'branch_s{s:04d}', archive_file(branch, s))
                                        for s in range(10, step, 10)]
            for label, path in paths:
                if not path.exists():
                    raise FileNotFoundError(f'Missing frozen historical opponent: {path}')
                historical_models[label] = read_historical(path, device)
        openings = training_openings(opening_manifest, games, seed + step * 17)
        samples = []
        totals = Counter()
        per_opponent = defaultdict(Counter)
        game_lengths = []
        for game_idx, opening in enumerate(openings):
            name, rival = choose_rival(branch, rng, snapshot, historical_models)
            color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
            batch, result = rollout_game(snapshot, rival, opening, color, device, max_plies)
            samples.extend(batch)
            tally = per_opponent[name]
            tally['games'] += 1
            totals['games'] += 1
            totals['learner_decisions'] += len(batch)
            if result['truncated']:
                label = 'truncated'
            else:
                outcome = result['outcome_for_learner']
                label = 'wins' if outcome > 0 else 'losses' if outcome < 0 else 'draws'
            tally[label] += 1
            totals[label] += 1
            game_lengths.append(result['plies'])
            if (game_idx + 1) % 128 == 0:
                print(f'  {branch} step {step}/{steps}: '
                      f'{game_idx+1}/{games} games collected', flush=True)
        del historical_models, snapshot
        stats = detailed_ppo_update(model, optimizer, samples, device, epochs=epochs,
                                    batch_size=batch_size, seed=seed + step)
        # Save summary figures, not each transition's board tensor.
        row = {
            'step': step, 'cumulative_games': games*step, 'games': totals['games'],
            'wins': totals['wins'], 'draws': totals['draws'], 'losses': totals['losses'],
            'truncated': totals['truncated'], 'learner_decisions': len(samples),
            'mean_plies': float(np.mean(game_lengths)),
            'distinct_start_positions': games // 2,
            'recent_opponent_games': per_opponent['recent_frozen']['games'],
            'historical_opponent_games': totals['games']-per_opponent['recent_frozen']['games'],
            **stats, 'elapsed_seconds': round(time.perf_counter()-start, 1),
        }
        history.append(row)
        for label, count in sorted(per_opponent.items()):
            opponent_history.append({'step': step, 'opponent': label,
                                     'games': count['games'], 'wins': count['wins'],
                                     'draws': count['draws'], 'losses': count['losses'],
                                     'truncated': count['truncated']})
        stored = {
            'schema': 1, 'step': step, 'iteration': 100+step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_channels': int(initial['model_channels']),
            'model_blocks': int(initial['model_blocks']),
            'config': cfg, 'history': history,
            'opponent_history': opponent_history,
            'description': f'Isolated {branch} PPO branch from frozen V100',
            'saved_utc': datetime.now(timezone.utc).isoformat(),
        }
        archived = (step % 10 == 0 or step == steps)
        if archived and archive_file(branch, step).exists():
            raise FileExistsError(f'Never overwriting permanent branch archive: '
                                  f'{archive_file(branch,step)}')
        # Commit latest first. Crash between commits is repaired on resume.
        atomic_save(stored, progress_file(branch))
        if archived:
            atomic_save(stored, archive_file(branch, step))
        write_branch_logs(history, opponent_history, branch)
        print(f'{branch.upper()} step {step}/{steps}: W={totals["wins"]} '
              f'D={totals["draws"]} L={totals["losses"]} '
              f'T={totals["truncated"]}; '
              f'KL+={stats["approx_kl_positive"]:.5f}, '
              f'clip={stats["clip_fraction"]:.3f}, '
              f'EV={stats["rollout_value_explained_variance"]}; '
              f'{row["elapsed_seconds"]:.0f}s', flush=True)
        if auto_eval and archived:
            evaluate_snapshot(branch, step, opening_manifest, device=device,
                              pairs=eval_pairs, max_plies=max_plies)
    print(f'Completed {branch}: {steps*games:,} games. '
          f'Checkpoint {progress_file(branch)}', flush=True)


def eval_openings(manifest, pairs):
    if not 5 <= pairs <= len(manifest['eval']):
        raise ValueError(f'Evaluation pairs must be between 5 and {len(manifest["eval"])}.')
    rng = random.Random(EVAL_SEED)
    return [manifest['eval'][i] for i in rng.sample(range(len(manifest['eval'])), pairs)]


def eval_file(branch, step, pairs):
    return OUTPUTS / f'ppo_scale_eval_{branch}_s{step:04d}_p{pairs:03d}.csv'


def eval_manifest_file(branch, step, pairs):
    return OUTPUTS / f'ppo_scale_eval_{branch}_s{step:04d}_p{pairs:03d}_manifest.json'


def eval_summary_file(branch, step, pairs):
    return OUTPUTS / f'ppo_scale_eval_{branch}_s{step:04d}_p{pairs:03d}_summary.json'


def eval_candidate_file(branch, step):
    if step == 0:
        return V100
    archived = archive_file(branch, step)
    if archived.is_file():
        return archived
    progress = progress_file(branch)
    if progress.is_file():
        saved = torch.load(progress, map_location='cpu', weights_only=True)
        if int(saved['step']) == step:
            return progress
    raise FileNotFoundError(f'No saved snapshot for {branch} step {step}')


def eval_opponent_files():
    result = list(EVAL_OPPONENTS)
    if V1M.is_file():
        result.append(('supervised_1m', V1M))
    return result


def validate_eval_manifest(branch, step, pairs, openings, file, opponents, max_plies):
    config = {
        'schema': 1, 'branch': branch, 'step': step, 'pairs': pairs,
        'max_plies': max_plies, 'seed': EVAL_SEED,
        'candidate_sha256': sha256(file),
        'opponent_sha256': {label: sha256(path) for label, path in opponents},
        'openings': openings, 'search': False,
    }
    path = eval_manifest_file(branch, step, pairs)
    if path.is_file():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous != config:
            raise RuntimeError('Evaluation manifest mismatches checkpoints or starts; '
                               'never mixing old and new results.')
    else:
        if eval_file(branch, step, pairs).exists():
            raise RuntimeError('Evaluation game results exist without a manifest.')
        json_atomic(path, config)
    return config


def load_eval_rows(path, labels, openings):
    if not path.exists():
        return {}
    existing = {}
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = ['opponent','opening_id','candidate_color','score','truncated',
                    'termination','plies']
        if reader.fieldnames != required:
            raise RuntimeError('Evaluation CSV layout changed. Existing games preserved.')
        for row in reader:
            idx = int(row['opening_id'])
            key = (row['opponent'], idx, row['candidate_color'])
            if (key in existing or row['opponent'] not in labels or not 0 <= idx < len(openings)
                    or row['candidate_color'] not in ('white','black')
                    or row['truncated'] not in ('0','1')
                    or (row['score'] == '') != (row['truncated'] == '1')
                    or row['score'] not in ('', '0.0', '0.5', '1.0')):
                raise RuntimeError(f'Invalid saved evaluation row: {key}')
            existing[key] = row
    return existing


def bootstrap_pair_ci(values, seed=EVAL_SEED):
    """Resample paired-opening means; never treat two colors as independent."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 5:
        return None
    rng = np.random.default_rng(seed)
    # Memory bounded even for all 256 held-out openings.
    resamples = [values[rng.integers(0,len(values),len(values))].mean() for _ in range(3000)]
    lo, hi = np.quantile(resamples, [.025,.975])
    return [round(float(lo),4), round(float(hi),4)]


def summarize_eval(rows, opponents, pairs):
    results = {}
    for label, _ in opponents:
        tally = Counter()
        scores = defaultdict(dict)
        for (name, idx, color), row in rows.items():
            if name != label:
                continue
            tally['played'] += 1
            if row['truncated'] == '1':
                tally['truncated'] += 1
                continue
            s = float(row['score'])
            scores[idx][color] = s
            tally['wins' if s == 1.0 else 'losses' if s == 0.0 else 'draws'] += 1
        complete = tally['wins'] + tally['draws'] + tally['losses']
        paired = [.5 * (d['white'] + d['black']) for _, d in sorted(scores.items())
                  if 'white' in d and 'black' in d]
        results[label] = {
            'played': tally['played'], 'expected': pairs*2,
            'wins': tally['wins'], 'draws': tally['draws'], 'losses': tally['losses'],
            'truncated': tally['truncated'],
            'score': round((tally['wins']+.5*tally['draws'])/complete, 4) if complete else None,
            'paired_score': round(float(np.mean(paired)),4) if paired else None,
            'paired_pairs': len(paired),
            'paired_95pct_ci': bootstrap_pair_ci(paired),
        }
    return results


def evaluate_snapshot(branch, step, manifest, device=None, pairs=PAIRS,
                      max_plies=MAX_PLIES):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    starts = eval_openings(manifest, pairs)
    candidate_file = eval_candidate_file(branch, step)
    opponents = eval_opponent_files()
    validate_eval_manifest(branch, step, pairs, starts, candidate_file,
                           opponents, max_plies)
    file = eval_file(branch, step, pairs)
    saved = load_eval_rows(file, [x[0] for x in opponents], starts)
    total = pairs * 2 * len(opponents)
    if len(saved) == total:
        print(f'  Eval {branch} s{step:04d} already saved ({total} games).', flush=True)
    else:
        print(f'  Evaluating {branch} s{step:04d} on {pairs} held-out HUMAN openings '
              f'({total-len(saved)} games left), NO SEARCH.', flush=True)
        candidate = read_historical(candidate_file, device)
        candidate.eval()
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open('a', newline='', encoding='utf-8') as out:
            columns = ['opponent','opening_id','candidate_color','score','truncated',
                       'termination','plies']
            writer = csv.DictWriter(out, fieldnames=columns)
            if out.tell() == 0:
                writer.writeheader()
            for label, opponent_path in opponents:
                remaining = [(i, color) for i in range(pairs) for color in ('white','black')
                             if (label, i, color) not in saved]
                if not remaining:
                    continue
                rival = read_historical(opponent_path, device)
                rival.eval()
                for i, color in remaining:
                    result = play_eval_game(candidate, rival, starts[i],
                                            chess.WHITE if color=='white' else chess.BLACK,
                                            device, max_plies, EVAL_SEED + i*97 + int(color=='white'))
                    row = {
                        'opponent': label, 'opening_id': i,
                        'candidate_color': color,
                        'score': '' if result['truncated'] else str(float(result['score'])),
                        'truncated': int(result['truncated']),
                        'termination': result['termination'], 'plies': result['plies'],
                    }
                    writer.writerow(row)
                    out.flush()  # safe recovery after a game, not just after an iteration
                    row['truncated'] = str(row['truncated'])
                    row['plies'] = str(row['plies'])
                    saved[(label,i,color)] = row
                del rival
        del candidate
    result = {
        'branch': branch, 'step': step, 'cumulative_games': step*GAMES,
        'pairs': pairs, 'search_used': False,
        'opponents': summarize_eval(saved, opponents, pairs),
    }
    json_atomic(eval_summary_file(branch, step, pairs), result)
    for label, scores in result['opponents'].items():
        print(f'  {branch} s{step:04d} vs {label}: '
              f'W={scores["wins"]} D={scores["draws"]} L={scores["losses"]} '
              f'T={scores["truncated"]} score={scores["score"]} '
              f'CI={scores["paired_95pct_ci"]}', flush=True)
    return result




def direct_match_paths(step, pairs):
    stem = f'ppo_scale_pool_vs_latest_direct_s{step:04d}_p{pairs:03d}'
    return (OUTPUTS / (stem+'.csv'), OUTPUTS / (stem+'_manifest.json'),
            OUTPUTS / (stem+'_summary.json'))


def evaluate_direct_match(step, manifest, device=None, pairs=PAIRS,
                          max_plies=MAX_PLIES):
    """True pool-vs-latest greedy head-to-head at the SAME training step."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pool_file = eval_candidate_file('pool', step)
    latest_file = eval_candidate_file('latest', step)
    starts = eval_openings(manifest, pairs)
    results_path, manifest_path, summary_path = direct_match_paths(step, pairs)
    expected = {'schema': 1, 'step': step, 'pairs': pairs, 'max_plies': max_plies,
                'pool_sha256': sha256(pool_file), 'latest_sha256': sha256(latest_file),
                'opening_sha256': manifest['sha256'], 'openings': starts, 'search': False}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != expected:
            raise RuntimeError('Direct tournament settings differ. Refusing to alter existing games.')
    else:
        if results_path.exists():
            raise RuntimeError('Direct tournament CSV exists without a matching manifest.')
        json_atomic(manifest_path, expected)
    rows = {}
    if results_path.exists():
        with results_path.open(newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ['opening_id','pool_color','score','truncated','termination','plies']:
                raise RuntimeError('Direct tournament CSV format changed.')
            for row in reader:
                key = (int(row['opening_id']), row['pool_color'])
                if key in rows or not 0 <= key[0] < pairs or key[1] not in ('white','black'):
                    raise RuntimeError('Invalid or duplicate direct tournament game.')
                if (row['score'] == '') != (row['truncated'] == '1'):
                    raise RuntimeError('Truncation was incorrectly scored as a draw.')
                rows[key] = row
    if len(rows) < 2*pairs:
        pool_model = read_historical(pool_file, device)
        latest_model = read_historical(latest_file, device)
        with results_path.open('a', newline='', encoding='utf-8') as f:
            cols = ['opening_id','pool_color','score','truncated','termination','plies']
            writer = csv.DictWriter(f, fieldnames=cols)
            if f.tell() == 0:
                writer.writeheader()
            for i, opening in enumerate(starts):
                for color in ('white','black'):
                    key = (i,color)
                    if key in rows:
                        continue
                    game = play_eval_game(pool_model, latest_model, opening,
                                         chess.WHITE if color=='white' else chess.BLACK,
                                         device, max_plies, EVAL_SEED + i*97 + int(color=='white'))
                    row = {'opening_id': i, 'pool_color': color,
                           'score': '' if game['truncated'] else str(float(game['score'])),
                           'truncated': int(game['truncated']),
                           'termination': game['termination'], 'plies': game['plies']}
                    writer.writerow(row)
                    f.flush()
                    row['truncated'] = str(row['truncated'])
                    rows[key] = row
        del pool_model, latest_model
    scores = []
    tally = Counter()
    for i in range(pairs):
        scores_for_pair = []
        for color in ('white','black'):
            row = rows.get((i,color))
            if row is None:
                continue
            if row['truncated']=='1':
                tally['truncated'] += 1
            else:
                score=float(row['score'])
                tally['wins' if score==1 else 'losses' if score==0 else 'draws']+=1
                scores_for_pair.append(score)
        if len(scores_for_pair)==2:
            scores.append(float(np.mean(scores_for_pair)))
    report_data = {'step': step, 'pairs': pairs, 'pool_vs_latest': True,
                   'wins': tally['wins'], 'draws': tally['draws'],
                   'losses': tally['losses'], 'truncated': tally['truncated'],
                   'fully_completed_pairs': len(scores),
                   'paired_score': round(float(np.mean(scores)),4) if scores else None,
                   'paired_95pct_ci': bootstrap_pair_ci(scores),
                   'search_used': False}
    json_atomic(summary_path, report_data)
    print(f'  Direct pool vs latest s{step:04d}: '
          f'W={tally["wins"]} D={tally["draws"]} L={tally["losses"]} '
          f'T={tally["truncated"]} score={report_data["paired_score"]} '
          f'CI={report_data["paired_95pct_ci"]}', flush=True)
    return report_data


def saved_steps(branch):
    files = branch_dir(branch)
    if not files.exists():
        return []
    available = [int(p.stem[1:]) for p in files.glob('s????.pt') if p.stem[1:].isdigit()]
    return sorted(set(available))


def evaluate_saved(manifest, device, pairs=PAIRS, max_plies=MAX_PLIES):
    evaluate_snapshot('pool', 0, manifest, device=device, pairs=pairs, max_plies=max_plies)
    for branch in BRANCHES:
        for step in saved_steps(branch):
            evaluate_snapshot(branch, step, manifest, device=device,
                              pairs=pairs, max_plies=max_plies)
    for step in sorted(set(saved_steps('pool')) & set(saved_steps('latest'))):
        evaluate_direct_match(step, manifest, device=device, pairs=pairs, max_plies=max_plies)
    report(manifest, pairs=pairs)


def paired_difference(first, second, label, pairs):
    """Difference in scores on exactly matching opening+color games."""
    by_pair = defaultdict(dict)
    for i in range(pairs):
        for color in ('white', 'black'):
            key = (label, i, color)
            if key not in first or key not in second:
                continue
            a, b = first[key], second[key]
            if a['score'] == '' or b['score'] == '':
                continue
            by_pair[i][color] = float(a['score'])-float(b['score'])
    differences = [.5*(item['white']+item['black']) for _,item in sorted(by_pair.items())
                   if 'white' in item and 'black' in item]
    return {'pairs': len(differences),
            'delta': round(float(np.mean(differences)),4) if differences else None,
            'paired_95pct_ci': bootstrap_pair_ci(differences)}


def report(manifest, pairs=PAIRS):
    """Paired comparison of pool and latest, same held-out starts and opponent."""
    common_steps = sorted(set(saved_steps('pool')) & set(saved_steps('latest')))
    all_rows = []
    deltas = []
    for branch in BRANCHES:
        for step in saved_steps(branch):
            file = eval_summary_file(branch, step, pairs)
            if not file.exists():
                continue
            data = json.loads(file.read_text(encoding='utf-8'))
            for opponent, result in data['opponents'].items():
                ci = result['paired_95pct_ci']
                all_rows.append({
                    'branch': branch, 'step': step, 'cumulative_games': step*GAMES,
                    'opponent': opponent, 'wins': result['wins'], 'draws': result['draws'],
                    'losses': result['losses'], 'truncated': result['truncated'],
                    'score': result['score'], 'ci_low': ci[0] if ci else '',
                    'ci_high': ci[1] if ci else '',
                })
    for step in common_steps:
        one = eval_file('pool', step, pairs)
        two = eval_file('latest', step, pairs)
        if not one.is_file() or not two.is_file():
            continue
        labels = [name for name, _ in eval_opponent_files()]
        starts = eval_openings(manifest, pairs)
        pool_rows = load_eval_rows(one, labels, starts)
        latest_rows = load_eval_rows(two, labels, starts)
        for label in labels:
            delta = paired_difference(pool_rows, latest_rows, label, pairs)
            deltas.append({'step': step, 'opponent': label, **delta})
            print(f'  Pool - latest, step {step}, vs {label}: '
                  f'{delta["delta"]}; CI={delta["paired_95pct_ci"]}', flush=True)
    if all_rows:
        safe_csv(all_rows, OUTPUTS / f'ppo_scale_eval_summary_p{pairs:03d}.csv')
    if deltas:
        safe_csv(deltas, OUTPUTS / f'ppo_scale_pool_vs_latest_p{pairs:03d}.csv')
    direct = []
    for step in common_steps:
        _, _, summary_path = direct_match_paths(step, pairs)
        if summary_path.exists():
            direct.append(json.loads(summary_path.read_text(encoding='utf-8')))
    if direct:
        safe_csv([{k: (','.join(map(str,v)) if isinstance(v, list) else v)
                   for k,v in item.items()} for item in direct],
                 OUTPUTS / f'ppo_scale_direct_match_summary_p{pairs:03d}.csv')
    json_atomic(OUTPUTS / f'ppo_scale_report_p{pairs:03d}.json', {
        'hypothesis': 'Does a historical checkpoint opponent pool reduce cycling?',
        'comparison': 'Pool minus latest-only, matched color and human opening',
        'checkpoint_scores': all_rows, 'paired_branch_differences': deltas,
        'direct_pool_vs_latest_matches': direct,
        'caution': ('Held-out V0-validation games are a development evaluation set. '
                    'Test a final selected model against truly independent new PGN '
                    'games or an external policy; multiple seeds improve reliability.'),
    })
    print('Saved PPO branch report under:', OUTPUTS, flush=True)
    return deltas


def smoke_test(opening_manifest, device):
    """Real integration test using V100, but all new files go to a temp directory."""
    import tempfile
    original_branch_dir = globals()['branch_dir']
    original_outputs = globals()['OUTPUTS']
    try:
        with tempfile.TemporaryDirectory(prefix='ee641_ppo_scale_smoke_') as folder:
            root = Path(folder)
            globals()['branch_dir'] = lambda name: root / 'checkpoints' / name
            globals()['OUTPUTS'] = root / 'outputs'
            print('Running isolated 4-game PPO smoke test per branch, NO SEARCH.', flush=True)
            for branch in BRANCHES:
                train_branch(branch, opening_manifest, steps=1, games=4,
                             seed=3214, max_plies=60, epochs=1,
                             batch_size=4, lr=1e-5, device=device,
                             auto_eval=False)
            print('SMOKE SUCCESS: both branches made an update. '
                  'Temporary weights deleted; original model files unchanged.', flush=True)
    finally:
        globals()['branch_dir'] = original_branch_dir
        globals()['OUTPUTS'] = original_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare','smoke','pool','latest','both','eval','report'),
                        default='both')
    parser.add_argument('--iterations', type=int, default=STEPS)
    parser.add_argument('--games', type=int, default=GAMES)
    parser.add_argument('--eval-pairs', type=int, default=PAIRS)
    parser.add_argument('--max-plies', type=int, default=MAX_PLIES)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--ppo-epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--skip-auto-eval', action='store_true')
    args = parser.parse_args()
    if (args.iterations < 1 or args.games < 2 or args.games % 2 or args.max_plies < 4
            or args.eval_pairs < 5 or args.eval_pairs > 256 or args.ppo_epochs < 1
            or args.batch_size < 2 or args.seed < 0 or args.lr <= 0):
        parser.error('Invalid iterations, games, eval-pairs, max-plies, seed, batch size, epochs or lr.')
    opening_manifest = prepare()
    validate_manifest(opening_manifest)
    print('Human starts:', len(opening_manifest['train']), 'training,',
          len(opening_manifest['eval']), 'held-out.', flush=True)
    if args.mode == 'prepare':
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, '| GPU if available | NO SEARCH or engine.', flush=True)
    if args.mode == 'smoke':
        smoke_test(opening_manifest, device)
    elif args.mode in BRANCHES or args.mode == 'both':
        selected = BRANCHES if args.mode == 'both' else [args.mode]
        for branch in selected:
            train_branch(branch, opening_manifest, steps=args.iterations,
                         games=args.games, seed=args.seed, max_plies=args.max_plies,
                         epochs=args.ppo_epochs, batch_size=args.batch_size,
                         lr=args.lr, device=device,
                         auto_eval=not args.skip_auto_eval, eval_pairs=args.eval_pairs)
        if args.mode == 'both' and not args.skip_auto_eval:
            evaluate_saved(opening_manifest, device, pairs=args.eval_pairs,
                           max_plies=args.max_plies)
    elif args.mode == 'eval':
        evaluate_saved(opening_manifest, device, pairs=args.eval_pairs,
                       max_plies=args.max_plies)
    else:
        report(opening_manifest, pairs=args.eval_pairs)


if __name__ == '__main__':
    main()
