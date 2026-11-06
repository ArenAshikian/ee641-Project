"""Fresh October 2026 Maia confirmation of supervised V0 versus PPO.

No model weights are updated. Our policy uses one neural-network inference.
The Maia opponent uses Lc0 policyhead without tree search. All openings
come from October games, not September model training and PPO evaluations.
"""
import argparse
import hashlib
import io
import json
import os
import random
import time
import urllib.request
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
import zstandard

from chess_rl.human_openings import opening_from_game, write_json
from chess_rl.lichess_data import read_raw_games, read_headers, selection_reason
from chess_rl.maia_benchmark import (find_lc0, available_weights, launch_engine,
                                     run_game, append_record, read_records)
from chess_rl.maia_stats import summarize_games, compare_models, score_from_record
from chess_rl.ppo_scale import branch_dir
from chess_rl.ppo_train import CHECKPOINTS, OUTPUTS, V0, board_from_opening, read_historical, sha256

ROOT = Path(__file__).resolve().parent.parent
OCTOBER_URL = ('https://database.lichess.org/standard/'
               'lichess_db_standard_rated_2026-10.pgn.zst')
FROZEN_OPENINGS = ROOT / 'data' / 'processed' / 'maia_confirmation_oct2026.json'
SEED = 641202611
TEST_PAIRS = 256  # 512 games per model per Maia level, with paired colors
SMOKE_PAIRS = 2   # separate openings, NOT used in confirmation
TOTAL_STARTS = TEST_PAIRS + SMOKE_PAIRS
MAX_SCANNED = 250000
GAME_LIMIT = 240
MODELS = {
    'v0': V0,
    'v100': CHECKPOINTS / 'ppo_v0100.pt',
    'latest_s49': branch_dir('latest') / 's0049.pt',
}
RATINGS = (1500, 1900)
PRIMARY = 'latest_s49_minus_v0_vs_maia1900'
SCHEMA = 1


def digest(doc):
    return hashlib.sha256(json.dumps({k: v for k, v in doc.items() if k != 'sha256'},
                                     sort_keys=True).encode('utf-8')).hexdigest()


def validate_openings(doc):
    if doc.get('schema') != SCHEMA or doc.get('sha256') != digest(doc):
        raise RuntimeError('Confirmation openings manifest was modified.')
    if doc.get('source_url') != OCTOBER_URL or doc.get('seed') != SEED:
        raise RuntimeError('Confirmation must use the frozen October 2026 dataset.')
    if len(doc.get('openings', [])) != TOTAL_STARTS:
        raise RuntimeError('Wrong number of October human opening positions.')
    fens = set()
    for sequence in doc['openings']:
        board = board_from_opening(sequence)
        if board.is_game_over(claim_draw=True) or board.fen() in fens:
            raise RuntimeError('Duplicate or terminal October opening.')
        fens.add(board.fen())
    return doc


def old_fens():
    """Exclude exact matches from PPO training and prior PPO/Maia evaluations."""
    older = [ROOT / 'data' / 'processed' / 'ppo_scale_human_openings.json',
             ROOT / 'data' / 'processed' / 'maia_openings.json']
    excluded = set()
    names = {}
    for path in older:
        if not path.is_file():
            raise FileNotFoundError('Missing previous opening manifest: ' + str(path))
        doc = json.loads(path.read_text(encoding='utf-8'))
        # Both original scripts saved a manifest content checksum.
        if path.name == 'maia_openings.json':
            from chess_rl.maia_benchmark import verify_openings
            verify_openings(doc)
            sequences = doc['openings']
        else:
            from chess_rl.human_openings import validate_manifest
            validate_manifest(doc)
            sequences = doc['train'] + doc['eval']
        names[path.name] = doc['sha256']
        for moves in sequences:
            excluded.add(board_from_opening(moves).fen())
    return excluded, names


@contextmanager
def october_stream(source=None):
    """Read a fraction of the 31 GB archive without saving the full download."""
    if source:
        path = Path(source)
        if not path.is_file() or path.suffix.lower() not in ('.pgn', '.zst'):
            raise FileNotFoundError('Expected a local October .pgn or .pgn.zst file: ' + str(path))
        if path.suffix.lower() == '.zst':
            with zstandard.open(path, 'rt', encoding='utf-8', errors='replace') as text:
                yield text
        else:
            with path.open('r', encoding='utf-8', errors='replace') as text:
                yield text
    else:
        request = urllib.request.Request(OCTOBER_URL,
            headers={'User-Agent': 'EE641-Research-Confirmation/1.0',
                     'Accept-Encoding': 'identity'})
        print('Streaming first portion of official October 2026 Lichess archive...', flush=True)
        with urllib.request.urlopen(request, timeout=90) as response:
            with zstandard.ZstdDecompressor().stream_reader(response) as decompressed:
                with io.TextIOWrapper(decompressed, encoding='utf-8', errors='replace') as text:
                    yield text


def prepare(source=None, max_scanned=MAX_SCANNED):
    if FROZEN_OPENINGS.exists():
        doc = validate_openings(json.loads(FROZEN_OPENINGS.read_text(encoding='utf-8')))
        print(f'Using {TEST_PAIRS} frozen October confirmation openings and '
              f'{SMOKE_PAIRS} separate smoke openings: {FROZEN_OPENINGS}', flush=True)
        return doc
    banned, existing_hashes = old_fens()
    selected, seen = [], set(banned)
    identities = set()
    scanned, accepted = 0, 0
    with october_stream(source) as stream:
        for raw in read_raw_games(stream):
            scanned += 1
            if scanned % 5000 == 0:
                print(f'  scanned {scanned:,}, accepted {accepted:,}, unique unused starts {len(selected)}/{TOTAL_STARTS}', flush=True)
            headers = read_headers(raw)
            if selection_reason(headers) != 'accepted':
                if scanned >= max_scanned:
                    break
                continue
            date = headers.get('UTCDate', headers.get('Date', ''))
            if not date.startswith('2026.10.'):
                if scanned >= max_scanned:
                    break
                continue
            accepted += 1
            game_id = headers.get('Site') or hashlib.sha256(raw.encode()).hexdigest()
            if game_id in identities:
                continue
            game = chess.pgn.read_game(io.StringIO(raw))
            if game is None or game.errors:
                continue
            moves = opening_from_game(game, game_id, SEED)
            if moves is None:
                continue
            fen = board_from_opening(moves).fen()
            if fen in seen:
                continue
            selected.append(moves)
            seen.add(fen)
            identities.add(game_id)
            if len(selected) == TOTAL_STARTS or scanned >= max_scanned:
                break
    if len(selected) < TOTAL_STARTS:
        raise RuntimeError(f'Only {len(selected)}/{TOTAL_STARTS} starts from {scanned:,} games. '
                           'Increase --max-scanned or supply local October PGN.')
    random.Random(SEED).shuffle(selected)
    doc = {
        'schema': SCHEMA, 'seed': SEED, 'source_url': OCTOBER_URL,
        'source_note': ('October 2026 rated standard human games. Source games are different '
                        'from September training; opening positions may naturally recur.'),
        'source_mode': 'local_october_file' if source else 'official_partial_http_stream',
        'source_local_path': str(Path(source).resolve()) if source else None,
        'filters': '1800-2200 on both sides; rated blitz/rapid; no bots, board terminations',
        'scanned_games': scanned, 'accepted_games': accepted,
        'prior_manifest_sha256': existing_hashes,
        'openings': selected,
        'test_indices': [0, TEST_PAIRS],
        'smoke_indices': [TEST_PAIRS, TOTAL_STARTS],
    }
    doc['sha256'] = digest(doc)
    write_json(FROZEN_OPENINGS, doc)
    print(f'Frozen {TEST_PAIRS} CONFIRMATION opening pairs + {SMOKE_PAIRS} '
          f'disjoint smoke pairs: {FROZEN_OPENINGS}', flush=True)
    return doc


def output_paths(mode):
    name = 'maia_confirm_oct2026' if mode == 'eval' else 'maia_confirm_oct2026_smoke'
    return (OUTPUTS / (name + '_manifest.json'),
            OUTPUTS / (name + '_games.jsonl'),
            OUTPUTS / (name + '_summary.json'))


def freeze_settings(doc, engine, weights, mode):
    for name, path in MODELS.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing {name} checkpoint: {path}')
    return {
        'schema': SCHEMA, 'experiment': 'maia_confirm_oct2026',
        'evaluation_mode': mode,
        'pre_registered_primary': PRIMARY,
        'primary_is_two_sided': True,
        'october_openings_sha256': doc['sha256'],
        'paired_openings': TEST_PAIRS if mode == 'eval' else SMOKE_PAIRS,
        'models': {name: sha256(path) for name, path in MODELS.items()},
        'maia_weights': {name: row['sha256'] for name, row in weights.items()},
        'lc0_sha256': sha256(engine),
        'engine_algorithm': 'policyhead', 'engine_nodes_command': 1,
        'policy': 'single_forward_pass_greedy_legal_mask',
        'max_plies': GAME_LIMIT,
    }


def paired_scores(rows):
    paired = defaultdict(dict)
    for row in rows:
        paired[row['opening_id']][row['color']] = score_from_record(row)
    return {key: (vals['white'] + vals['black']) / 2
            for key, vals in paired.items()
            if 'white' in vals and 'black' in vals and
               vals['white'] is not None and vals['black'] is not None}


def primary_test(rows, seed=SEED, draws=20000):
    """Two-sided paired sign-flip permutation test, respecting opening pairs."""
    a = paired_scores([r for r in rows if r['model'] == 'latest_s49' and r['maia'] == '1900'])
    b = paired_scores([r for r in rows if r['model'] == 'v0' and r['maia'] == '1900'])
    indices = sorted(set(a) & set(b))
    if not indices:
        return {'matched_pairs': 0, 'score_difference': None, 'ci95': None, 'p_two_sided': None}
    differences = np.asarray([a[i] - b[i] for i in indices], dtype=np.float64)
    observed = float(differences.mean())
    if len(indices) < 5:
        return {'matched_pairs': len(indices), 'score_difference': round(observed, 5),
                'ci95': None, 'p_two_sided': None}
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(draws, dtype=np.float64)
    as_extreme = 0
    for start in range(0, draws, 1000):
        n = min(1000, draws - start)
        which = rng.integers(len(indices), size=(n, len(indices)))
        bootstrap_means[start:start+n] = differences[which].mean(axis=1)
        flip = rng.choice([-1., 1.], size=(n, len(indices)))
        null_means = (differences * flip).mean(axis=1)
        as_extreme += int((np.abs(null_means) >= abs(observed) - 1e-12).sum())
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {'matched_pairs': len(indices), 'score_difference': round(observed, 5),
            'ci95': [round(float(low), 5), round(float(high), 5)],
            'p_two_sided': round((as_extreme + 1) / (draws + 1), 6),
            'bootstrap_draws': draws,
            'note': 'Pre-specified Latest S49 vs V0 against Maia 1900; paired colors/openings.'}


def write_report(records, ratings, pairs, mode, completed=False):
    _, _, path = output_paths(mode)
    groups = defaultdict(list)
    for row in records.values():
        groups[(row['model'], row['maia'])].append(row)
    stats = {}
    for maia in ratings:
        for label in MODELS:
            stats[f'{label}_vs_maia{maia}'] = summarize_games(
                groups[(label, str(maia))], pairs, seed=SEED + maia)
    comparisons = {}
    for maia in ratings:
        for label in ('v100', 'latest_s49'):
            comparisons[f'{label}_minus_v0_vs_maia{maia}'] = compare_models(
                groups[(label, str(maia))], groups[('v0', str(maia))],
                seed=SEED + maia + 3)
    for maia in ratings:
        comparisons[f'latest_s49_minus_v100_vs_maia{maia}'] = compare_models(
            groups[('latest_s49', str(maia))], groups[('v100', str(maia))],
            seed=SEED + maia + 47)
    # Only calculate confirmatory p value for the actual complete dataset.
    primary = primary_test(list(records.values())) if mode == 'eval' and completed else None
    report = {'schema': SCHEMA, 'mode': mode, 'completed': completed,
              'frozen_primary': PRIMARY, 'matches': stats, 'comparisons': comparisons,
              'primary_test': primary,
              'note': ('Confirmation is on October human games, unseen as a dataset by the '
                       'September trainer. This test compares a selected PPO model to frozen V0.')}
    write_json(path, report)
    return report


def evaluate(mode):
    if mode not in ('smoke', 'eval'):
        raise ValueError('Unknown confirmation mode')
    if not FROZEN_OPENINGS.is_file():
        raise FileNotFoundError('First run: run.cmd maia-confirm-prepare')
    doc = validate_openings(json.loads(FROZEN_OPENINGS.read_text(encoding='utf-8')))
    engine_path = find_lc0()
    weights = available_weights(RATINGS if mode == 'eval' else (1500,))
    settings = freeze_settings(doc, engine_path, weights, mode)
    manifest, games, _ = output_paths(mode)
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != settings:
            raise RuntimeError('Frozen benchmark settings have changed; refusing to mix games.')
    else:
        if games.exists():
            raise RuntimeError('Games exist but settings manifest is missing.')
        write_json(manifest, settings)
    records = read_records(games)
    pairs = TEST_PAIRS if mode == 'eval' else SMOKE_PAIRS
    indices = list(range(TEST_PAIRS)) if mode == 'eval' else list(range(TEST_PAIRS, TOTAL_STARTS))
    ratings = RATINGS if mode == 'eval' else (1500,)
    total = len(MODELS) * len(ratings) * len(indices) * 2
    allowed_keys = {(m, str(r), i, c) for m in MODELS for r in ratings for i in indices
                    for c in ('white', 'black')}
    if not set(records).issubset(allowed_keys):
        raise RuntimeError('Saved games include unexpected models or openings.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'October confirmation: {len(indices)} opening pairs, {len(ratings)} Maia opponents, '
          f'{len(MODELS)} frozen models. {len(records)}/{total} games saved; '
          f'network device={device}; Lc0 POLICYHEAD, no search.', flush=True)
    for maia in ratings:
        engine = launch_engine(engine_path, Path(weights[str(maia)]['path']))
        try:
            for label, path in MODELS.items():
                count = sum((label,str(maia),i,color) in records
                            for i in indices for color in ('white', 'black'))
                if count == 2 * pairs:
                    print(f'  {label} versus Maia {maia}: already finished.', flush=True)
                    continue
                print(f'  {label} versus Maia {maia}: {2 * pairs-count} games remaining', flush=True)
                model = read_historical(path, device)
                model.eval()
                try:
                    for i in indices:
                        for color in ('white', 'black'):
                            key = (label, str(maia), i, color)
                            if key in records:
                                continue
                            started = time.perf_counter()
                            result = run_game(model, engine, doc['openings'][i],
                                              chess.WHITE if color == 'white' else chess.BLACK,
                                              device, GAME_LIMIT)
                            row = {'model': label, 'maia': str(maia), 'opening_id': i,
                                   'color': color, 'opening_uci': ' '.join(doc['openings'][i]),
                                   'elapsed_seconds': round(time.perf_counter()-started, 3), **result}
                            append_record(games, row)
                            records[key] = row
                            count += 1
                            if count % 32 == 0 or count == 2 * pairs:
                                print(f'    {label} vs Maia {maia}: {count}/{2 * pairs}', flush=True)
                finally:
                    del model
                preview = write_report(records, ratings, pairs, mode)
                s = preview['matches'][f'{label}_vs_maia{maia}']
                print(f'  {label} vs Maia {maia}: W={s["wins"]} D={s["draws"]} '
                      f'L={s["losses"]} T={s["truncated"]}; score={s["score_completed"]}', flush=True)
        finally:
            engine.quit()
    report = write_report(records, ratings, pairs, mode, completed=True)
    print('ALL GAMES COMPLETE. Summary: ' + str(output_paths(mode)[2]), flush=True)
    if mode == 'eval':
        print('PRE-SPECIFIED PRIMARY Latest S49 vs V0 versus Maia 1900:', report['primary_test'], flush=True)
        for k, v in report['comparisons'].items():
            print(f'  {k}: delta={v["score_delta"]}, CI={v["ci95_paired_difference"]}', flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('prepare', 'smoke', 'eval', 'report'))
    p.add_argument('--source', type=Path, default=None,
                   help='Optional local October 2026 PGN or PGN.zst, if streaming is unavailable')
    p.add_argument('--max-scanned', type=int, default=MAX_SCANNED)
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare(source=args.source, max_scanned=args.max_scanned)
    elif args.mode == 'report':
        manifest, games, _ = output_paths('eval')
        if not manifest.exists() or not games.exists():
            raise FileNotFoundError('Run maia-confirm-eval before requesting a report.')
        records = read_records(games)
        # Do not call an incomplete set a final result.
        expected = len(MODELS) * len(RATINGS) * TEST_PAIRS * 2
        done = len(records) == expected
        result = write_report(records, RATINGS, TEST_PAIRS, 'eval', completed=done)
        print('Games saved:', len(records), '/', expected)
        print('Pre-registered comparison:', result['primary_test'])
    else:
        evaluate(args.mode)


if __name__ == '__main__':
    main()
