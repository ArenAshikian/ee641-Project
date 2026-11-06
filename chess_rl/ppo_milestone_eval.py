"""Exploratory, read-only evaluation of 50k/75k/100k PPO checkpoints.

Reuses the frozen October Maia1900 confirmation games for V0 and S49, after
verifying their hashes and every paired opening. Plays only the three NEW
models. The original confirmation files and checkpoints are never modified.

Our policy is one-forward-pass greedy. Maia uses independent Lc0 policyhead,
not a search engine. This is post-hoc learning-curve analysis, NOT a new
preregistered hypothesis test or a search-based benchmark.
"""
import argparse
import json
import time
from pathlib import Path

import chess
import torch

from chess_rl import maia_confirm as old
from chess_rl.maia_benchmark import (find_lc0, available_weights, launch_engine,
                                     run_game, read_records, append_record)
from chess_rl.maia_stats import summarize_games, compare_models
from chess_rl.ppo_train import CHECKPOINTS, OUTPUTS, V0, read_historical, sha256
from chess_rl.human_openings import write_json
from chess_rl.ppo_batch_continue import SOURCE_SHA256
from chess_rl.ppo_long import safe_csv

RATING = '1900'
NAME = 'ppo_milestones_oct2026'
REFERENCES = {'v0': V0, 'latest_s49': old.MODELS['latest_s49']}
NEW = {
    'ppo_50k': (CHECKPOINTS / 'ppo_batch_continue' / 's0026.pt', 26, 50240),
    'ppo_75k': (CHECKPOINTS / 'ppo_batch_continue' / 's0075.pt', 75, 75328),
    'ppo_100k': (CHECKPOINTS / 'ppo_batch_continue' / 's0124.pt', 124, 100416),
}
ORDER = list(REFERENCES) + list(NEW)
COUNTS = {'v0': 0, 'latest_s49': 36928, 'ppo_50k': 50240,
          'ppo_75k': 75328, 'ppo_100k': 100416}
COMPARE = [('ppo_50k', 'latest_s49'),
           ('ppo_75k', 'latest_s49'),
           ('ppo_100k', 'latest_s49'),
           ('ppo_75k', 'ppo_50k'),
           ('ppo_100k', 'ppo_75k'),
           ('ppo_100k', 'ppo_50k'),
           ('ppo_100k', 'v0')]


def paths(mode):
    suffix = '' if mode == 'eval' else '_smoke'
    prefix = OUTPUTS / (NAME + suffix)
    return (Path(str(prefix) + '_manifest.json'),
            Path(str(prefix) + '_games.jsonl'),
            Path(str(prefix) + '_summary.json'))


def inspect_milestones():
    """Check the archived *milestones*, never the changing latest.pt."""
    hashes = {name: sha256(path) for name, path in REFERENCES.items()}
    if hashes['latest_s49'] != SOURCE_SHA256:
        raise RuntimeError('Original S49 no longer has its expected SHA256.')
    for name, (path, step, cumulative) in NEW.items():
        if not path.is_file():
            raise FileNotFoundError('Missing archived milestone: ' + str(path))
        state = torch.load(path, map_location='cpu', weights_only=True)
        if (state.get('step') != step or state.get('cumulative_games') != cumulative or
            state.get('config', {}).get('source_sha256') != SOURCE_SHA256 or
            'model_state_dict' not in state):
            raise RuntimeError(f'{name} has unexpected checkpoint metadata.')
        del state
        hashes[name] = sha256(path)
    return hashes


def source_records(doc, hashes):
    """Use only COMPLETE pre-existing October Maia1900 matches; no rewriting."""
    manifest, games, _ = old.output_paths('eval')
    if not manifest.is_file() or not games.is_file():
        raise FileNotFoundError('Original October confirmation is missing. '
                                'Finish run.cmd maia-confirm-eval first.')
    cfg = json.loads(manifest.read_text(encoding='utf-8'))
    if (cfg.get('evaluation_mode') != 'eval' or
        cfg.get('october_openings_sha256') != doc['sha256'] or
        cfg.get('max_plies') != old.GAME_LIMIT or
        cfg.get('paired_openings') != old.TEST_PAIRS or
        cfg.get('policy') != 'single_forward_pass_greedy_legal_mask' or
        cfg.get('engine_algorithm') != 'policyhead' or
        cfg.get('models', {}).get('v0') != hashes['v0'] or
        cfg.get('models', {}).get('latest_s49') != hashes['latest_s49']):
        raise RuntimeError('Original October confirmation configuration or models differ.')
    prior = read_records(games)
    reference = {}
    for label in REFERENCES:
        for i in range(old.TEST_PAIRS):
            for color in ('white', 'black'):
                key = (label, RATING, i, color)
                if key not in prior:
                    raise RuntimeError(f'October confirmation lacks reference game {key}.')
                row = prior[key]
                if (row['opening_uci'] != ' '.join(doc['openings'][i]) or
                    row['truncated'] and row['score'] is not None or
                    not row['truncated'] and row['score'] not in (0., .5, 1.)):
                    raise RuntimeError(f'Invalid original reference game {key}.')
                reference[key] = row
    return cfg, reference


def settings(doc, hashes, lc0, weights, mode, reference_settings=None):
    if mode == 'eval':
        assert reference_settings is not None
        if (reference_settings.get('maia_weights', {}).get(RATING) != weights[RATING]['sha256'] or
            reference_settings.get('lc0_sha256') != sha256(lc0)):
            raise RuntimeError('Maia1900 weights or Lc0 binary changed since original confirmation.')
    return {
        'schema': 1, 'mode': mode,
        'purpose': 'exploratory post-hoc 50k/75k/100k learning curve',
        'opening_manifest_sha256': doc['sha256'],
        'maia_1900_weights_sha256': weights[RATING]['sha256'],
        'lc0_sha256': sha256(lc0),
        'max_plies': old.GAME_LIMIT, 'maia_algorithm': 'policyhead',
        'our_policy': 'single_forward_pass_greedy_legal_mask',
        'pairs': old.TEST_PAIRS if mode == 'eval' else old.SMOKE_PAIRS,
        'indices': [0, old.TEST_PAIRS] if mode == 'eval'
                   else [old.TEST_PAIRS, old.TOTAL_STARTS],
        'checkpoint_sha256': hashes,
        'reference_source': 'October confirm (read only)' if mode == 'eval' else None,
        'statistical_note': 'Exploratory; repeated testing on previously used October openings.',
    }


def validate_saved_game(row, key, doc):
    label, rating, i, color = key
    if (label not in NEW or rating != RATING or
        i not in range(old.TOTAL_STARTS) or
        color not in ('white', 'black') or
        row['opening_uci'] != ' '.join(doc['openings'][i]) or
        (row['truncated'] and row['score'] is not None) or
        (not row['truncated'] and row['score'] not in (0., 0.5, 1.0))):
        raise RuntimeError(f'Saved milestone game has invalid metadata: {key}.')


def summarize(references, fresh, mode):
    combined = {**references, **fresh}
    pairs = old.TEST_PAIRS if mode == 'eval' else old.SMOKE_PAIRS
    names = ORDER if mode == 'eval' else list(NEW)
    results = {}
    for i, name in enumerate(names):
        matches = [row for key, row in combined.items() if key[0] == name]
        results[name] = summarize_games(matches, pairs, seed=old.SEED + 100 + i)
    comparisons = {}
    if mode == 'eval':
        for i, (a, b) in enumerate(COMPARE):
            ar = [row for key, row in combined.items() if key[0] == a]
            br = [row for key, row in combined.items() if key[0] == b]
            comparisons[a + '_minus_' + b] = compare_models(ar, br,
                                                            seed=old.SEED + 200 + i)
    return {'schema': 1, 'mode': mode, 'rating': 1900,
            'completed': len(fresh) == 2*pairs*len(NEW),
            'models': results, 'comparisons': comparisons,
            'note': 'Exploratory comparison; no new confirmatory p-value. '
                    'Truncated games excluded from game scores and paired comparisons.'}


def run(mode):
    if mode not in ('smoke', 'eval'):
        raise ValueError(mode)
    if not old.FROZEN_OPENINGS.is_file():
        raise FileNotFoundError('October frozen openings are missing: ' + str(old.FROZEN_OPENINGS))
    doc = old.validate_openings(json.loads(old.FROZEN_OPENINGS.read_text(encoding='utf-8')))
    hashes = inspect_milestones()
    references = {}
    ref_settings = None
    if mode == 'eval':
        ref_settings, references = source_records(doc, hashes)
    lc0 = find_lc0()
    weights = available_weights((1900,))
    cfg = settings(doc, hashes, lc0, weights, mode, ref_settings)
    manifest, games, summary_path = paths(mode)
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding='utf-8')) != cfg:
            raise RuntimeError('Milestone evaluation manifest differs. Refusing to mix data.')
    else:
        if games.exists():
            raise RuntimeError('Games exist without a manifest. Refusing to overwrite.')
        write_json(manifest, cfg)
    saved = read_records(games)
    lo, hi = cfg['indices']
    allowed = {(name, RATING, i, color) for name in NEW
               for i in range(lo, hi) for color in ('white', 'black')}
    if not set(saved).issubset(allowed):
        raise RuntimeError('Saved milestone games contain unexpected keys.')
    for key, row in saved.items():
        validate_saved_game(row, key, doc)
    total = len(allowed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Exploratory milestones vs Maia1900: {len(saved)}/{total} NEW games saved; '
          f'{len(references)} original reference games reused; device={device}.', flush=True)
    if len(saved) < total:
        engine = launch_engine(lc0, Path(weights[RATING]['path']))
        try:
            for label, (path, _, _) in NEW.items():
                remaining = [(i, c) for i in range(lo, hi) for c in ('white', 'black')
                             if (label, RATING, i, c) not in saved]
                if not remaining:
                    continue
                print(f'  {label}: {len(remaining)} remaining games', flush=True)
                model = read_historical(path, device)
                model.eval()
                try:
                    for i, color in remaining:
                        t0 = time.perf_counter()
                        outcome = run_game(model, engine, doc['openings'][i],
                                           chess.WHITE if color == 'white' else chess.BLACK,
                                           device, old.GAME_LIMIT)
                        row = {'model': label, 'maia': RATING, 'opening_id': i,
                               'color': color, 'opening_uci': ' '.join(doc['openings'][i]),
                               'elapsed_seconds': round(time.perf_counter()-t0, 3), **outcome}
                        append_record(games, row)
                        saved[(label, RATING, i, color)] = row
                        count = sum(key[0] == label for key in saved)
                        if count % 32 == 0 or count == 2*(hi-lo):
                            print(f'    {label}: {count}/{2*(hi-lo)} games', flush=True)
                finally:
                    del model
                save_results(references, saved, mode)
        finally:
            engine.quit()
    summary = save_results(references, saved, mode)
    print('Finished. Summary:', summary_path, flush=True)
    show_summary(summary)
    return summary


def save_results(references, fresh, mode):
    report = summarize(references, fresh, mode)
    _, _, summary_path = paths(mode)
    write_json(summary_path, report)
    csv_path = summary_path.with_name(summary_path.name.replace('_summary.json', '_learning_curve.csv'))
    rows = []
    for label, stats in report['models'].items():
        ci = stats['ci95_paired']
        rows.append({'model': label, 'cumulative_ppo_games': COUNTS[label],
                     'games': stats['total_games'], 'expected_games': stats['expected_games'],
                     'wins': stats['wins'], 'draws': stats['draws'],
                     'losses': stats['losses'], 'truncated': stats['truncated'],
                     'completed_game_score': stats['score_completed'],
                     'complete_opening_pairs': stats['complete_pairs'],
                     'paired_ci95_low': ci[0] if ci else None,
                     'paired_ci95_high': ci[1] if ci else None})
    safe_csv(rows, csv_path)
    return report


def show_summary(report):
    for label, row in report['models'].items():
        print(f'  {label}: {row["total_games"]}/{row["expected_games"]} games, '
              f'W={row["wins"]} D={row["draws"]} L={row["losses"]} '
              f'T={row["truncated"]}, score={row["score_completed"]}, '
              f'paired 95% CI={row["ci95_paired"]}', flush=True)
    for name, row in report['comparisons'].items():
        print(f'  {name}: delta={row["score_delta"]}, '
              f'paired 95% CI={row["ci95_paired_difference"]}, '
              f'matched pairs={row["matched_opening_pairs"]}', flush=True)
    print('Reminder: these are exploratory comparisons, not an independent '
          'confirmatory test.', flush=True)


def report(mode='eval'):
    manifest, games, summary_path = paths(mode)
    if not manifest.exists() or not games.exists():
        raise FileNotFoundError(f'No saved {mode} milestone results yet.')
    if not summary_path.exists():
        raise FileNotFoundError('Summary does not exist; rerun the evaluation to reconstruct it.')
    show_summary(json.loads(summary_path.read_text(encoding='utf-8')))


def main():
    parser = argparse.ArgumentParser(description='Read-only search-free PPO milestone evaluation')
    parser.add_argument('mode', choices=('smoke', 'eval', 'report'))
    args = parser.parse_args()
    if args.mode == 'report':
        report()
    else:
        run(args.mode)


if __name__ == '__main__':
    main()
