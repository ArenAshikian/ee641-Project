"""One-node Maia external benchmark; no engine enters our PPO/training pipeline.

Our chess network selects each move by one forward pass and legal-move masking.
Maia's independent weights use Lc0 policyhead (one network evaluation,
no tree search) with a UCI ``go nodes 1`` request.
"""
import argparse
import hashlib
import io
import json
import os
import random
import shutil
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch
import zstandard

from chess_rl.human_openings import opening_from_game, file_id, write_json
from chess_rl.lichess_data import (DEFAULT_PGN, read_raw_games, read_headers,
                                  selection_reason, validation_game)
from chess_rl.maia_stats import compare_models, summarize_games
from chess_rl.ppo_scale import branch_dir
from chess_rl.ppo_train import CHECKPOINTS, OUTPUTS, V0, board_from_opening, one_move, read_historical, sha256

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'processed' / 'maia_openings.json'
TOOLS = ROOT / 'benchmarks'
WEIGHTS = TOOLS / 'maia'
LC0 = TOOLS / 'lc0' / 'lc0.exe'
POLICY_LC0 = LC0.with_name('lc0-policyhead.exe')
V100 = CHECKPOINTS / 'ppo_v0100.pt'
CHECKPOINT_FILES = {
    'v0': V0,
    'v100': V100,
    'pool_s49': branch_dir('pool') / 's0049.pt',
    'latest_s49': branch_dir('latest') / 's0049.pt',
}
MAIAS = (1100, 1500, 1900)
SEED = 6412028
OPENING_COUNT = 256
GAME_LIMIT = 240
SOURCE_LIMIT = 800000
SCHEMA = 1


def saved_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def manifest_digest(doc):
    data = {k: doc[k] for k in ('schema', 'seed', 'openings', 'source', 'exclusions_sha256', 'source_kind')}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def verify_openings(doc):
    if doc.get('schema') != SCHEMA or doc.get('sha256') != manifest_digest(doc):
        raise RuntimeError('Maia opening manifest was changed or is not supported.')
    if len(doc['openings']) < OPENING_COUNT:
        raise RuntimeError('Not enough frozen Maia test positions.')
    previous = set()
    for moves in doc['openings']:
        key = tuple(moves)
        if key in previous:
            raise RuntimeError('Duplicate Maia opening.')
        previous.add(key)
        if board_from_opening(moves).is_game_over(claim_draw=True):
            raise RuntimeError('Maia opening is already terminal.')
    return doc


def previous_evaluation_keys():
    # Deliberately reject all PPO-scale evaluation openings; this new benchmark
    # must not reuse those same test positions.
    previous = ROOT / 'data' / 'processed' / 'ppo_scale_human_openings.json'
    if not previous.is_file():
        raise FileNotFoundError('Run the previous PPO-scale preparation first: ' + str(previous))
    previous_doc = saved_json(previous)
    from chess_rl.human_openings import validate_manifest
    validate_manifest(previous_doc)
    sequences = previous_doc['eval']
    fens = {board_from_opening(seq).fen() for seq in sequences}
    keys = {tuple(seq) for seq in sequences}
    return keys, fens, previous_doc['sha256']


def prepare_openings(source=DEFAULT_PGN, count=OPENING_COUNT, seed=SEED,
                     max_scanned=SOURCE_LIMIT):
    source = Path(source)
    old_keys, old_fens, old_digest = previous_evaluation_keys()
    if DATA.exists():
        doc = verify_openings(saved_json(DATA))
        if doc['source'] != file_id(source) or doc['seed'] != seed or doc['exclusions_sha256'] != old_digest:
            raise RuntimeError('Existing benchmark openings have different settings. Do not overwrite them.')
        print(f'Reusing {len(doc["openings"])} locked Maia opening positions.', flush=True)
        return doc
    if not source.exists():
        raise FileNotFoundError('Missing human PGN dataset: ' + str(source))
    external = source.resolve() != Path(DEFAULT_PGN).resolve()
    chosen, seen_fens, seen_keys = [], set(), set()
    scanned = 0
    def consider(game, game_id):
        if game is None or game.errors:
            return
        moves = opening_from_game(game, game_id, seed)
        if moves is None:
            return
        key = tuple(moves)
        fen = board_from_opening(moves).fen()
        if key in old_keys or fen in old_fens or key in seen_keys or fen in seen_fens:
            return
        seen_fens.add(fen)
        seen_keys.add(key)
        chosen.append(moves)
    print('Preparing fresh human-game Maia openings:', source, flush=True)
    if source.suffix.lower() == '.zst':
        with zstandard.open(source, 'rt', encoding='utf-8', errors='replace') as stream:
            for raw in read_raw_games(stream):
                scanned += 1
                headers = read_headers(raw)
                if selection_reason(headers) == 'accepted' and (external or validation_game(headers)):
                    game = chess.pgn.read_game(io.StringIO(raw))
                    identity = headers.get('Site') or hashlib.sha256(raw.encode()).hexdigest()
                    consider(game, identity)
                if len(chosen) >= count or scanned >= max_scanned:
                    break
    else:
        if source.suffix.lower() != '.pgn':
            raise ValueError('Use a .pgn or .pgn.zst human-game file.')
        with source.open('r', encoding='utf-8', errors='replace') as stream:
            while scanned < max_scanned and len(chosen) < count:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                scanned += 1
                if game.headers.get('Variant', 'Standard').lower() not in ('standard', 'chess'):
                    continue
                identity = game.headers.get('Site') or str(scanned)
                consider(game, identity)
    if len(chosen) < count:
        raise RuntimeError(f'Only {len(chosen)} distinct new human starts after {scanned:,} games. '
                           'Supply more human games or raise max_scanned.')
    random.Random(seed).shuffle(chosen)
    doc = {
        'schema': SCHEMA, 'seed': seed, 'source': file_id(source),
        'source_kind': ('external_human_games' if external else 'september_v0_validation_game_partition'),
        'exclusions_sha256': old_digest,
        'caution': ('September validation games were not PPO training games and these starts '
                    'were excluded from earlier PPO test openings, but V0 validation used the same '
                    'source split. For a fully independent final test supply a later external PGN.'
                    if not external else 'External human-game source; verify no overlap with model training.'),
        'scanned_games': scanned, 'openings': chosen,
    }
    doc['sha256'] = manifest_digest(doc)
    write_json(DATA, doc)
    print(f'Saved {len(chosen)} new human starts from {scanned:,} scanned games: {DATA}', flush=True)
    return doc


def find_lc0():
    # Lc0 >=0.32 selects 'policyhead' from the executable filename.
    # This algorithm evaluates just the current position, without MCTS.
    env = os.environ.get('LC0_PATH', '').strip().strip('"')
    original = Path(env) if env else LC0
    if not original.is_file() and not env:
        found = shutil.which('lc0')
        if found:
            original = Path(found)
    if not original.is_file():
        raise FileNotFoundError('Missing Lc0 executable: ' + str(original))
    if original.name.lower() == 'lc0-policyhead.exe':
        return original.resolve()
    if original.name.lower() != 'lc0.exe':
        raise RuntimeError('LC0_PATH must point to lc0.exe or lc0-policyhead.exe.')
    # Keep original Lc0 and all DLLs in place. Only copy the small executable.
    policy = original.with_name('lc0-policyhead.exe')
    if not policy.exists():
        shutil.copy2(original, policy)
        print('Created policy-only Lc0 executable: ' + str(policy), flush=True)
    elif sha256(original) != sha256(policy):
        raise RuntimeError('Lc0 and lc0-policyhead.exe differ. Remove the old '
                           'lc0-policyhead.exe so it can be recreated from lc0.exe.')
    return policy.resolve()


def available_weights(ratings):
    result = {}
    for rating in ratings:
        path = WEIGHTS / f'maia-{rating}.pb.gz'
        if (not path.is_file() or path.stat().st_size < 1024 or
                path.open("rb").read(2) != b"\x1f\x8b"):
            raise FileNotFoundError(f'Missing Maia {rating}: {path}. Run run.cmd maia-download.')
        result[str(rating)] = {'path': str(path.resolve()), 'sha256': sha256(path)}
    return result


def download_weights():
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for rating in MAIAS:
        target = WEIGHTS / f'maia-{rating}.pb.gz'
        if target.exists() and target.stat().st_size > 1024:
            print(f'Found Maia {rating}: {target}', flush=True)
            continue
        url = f'https://github.com/CSSLab/maia-chess/releases/download/v1.0/maia-{rating}.pb.gz'
        print('Downloading official Maia network:', url, flush=True)
        temp = target.with_suffix('.download')
        req = urllib.request.Request(url, headers={'User-Agent': 'EE641-research-benchmark'})
        try:
            with urllib.request.urlopen(req, timeout=90) as response, temp.open('wb') as out:
                shutil.copyfileobj(response, out)
            with temp.open('rb') as stream:
                if stream.read(2) != b'\x1f\x8b' or temp.stat().st_size < 1024:
                    raise RuntimeError(f'Invalid .pb.gz download for Maia {rating}.')
            os.replace(temp, target)
        finally:
            if temp.exists():
                temp.unlink()
        print(f'  Saved {target.name}; sha256 {sha256(target)}', flush=True)
    print('Maia networks ready. Lc0 engine executable must be installed separately.', flush=True)


def move_engine(engine, board):
    # The policyhead search algorithm, selected by lc0-policyhead.exe, evaluates
    # one board position and returns the largest legal policy logit. There is no
    # game-tree search. We still request 'go nodes 1' for compatibility.
    result = engine.play(board, chess.engine.Limit(nodes=1), info=chess.engine.INFO_ALL)
    if result.move is None or result.move not in board.legal_moves:
        raise RuntimeError('Maia failed to return a legal move.')
    # The UCI 'nodes' INFO field is diagnostic, not a reliable count of neural
    # evaluations. Some Lc0 configurations report 2 after 'go nodes 1'.
    # We enforce no search by selecting policyhead, not by checking this field.
    count = result.info.get('nodes')
    return result.move, count


def run_game(candidate, engine, opening, color, device, max_plies):
    board = board_from_opening(opening)
    moves = []
    nodes_observed = 0
    for ply in range(max_plies + 1):
        result = board.outcome(claim_draw=True)
        if result is not None:
            winner = result.winner
            score = .5 if winner is None else 1.0 if winner == color else 0.0
            return {'score': score, 'truncated': False, 'termination': result.termination.name,
                    'plies': ply, 'moves': ' '.join(moves), 'maia_max_reported_nodes': nodes_observed}
        if ply == max_plies:
            return {'score': None, 'truncated': True, 'termination': 'MAX_PLIES',
                    'plies': ply, 'moves': ' '.join(moves), 'maia_max_reported_nodes': nodes_observed}
        if board.turn == color:
            move, _ = one_move(candidate, board, device, sample=False)
        else:
            move, nodes = move_engine(engine, board)
            if nodes is not None:
                nodes_observed = max(nodes_observed, nodes)
        moves.append(move.uci())
        board.push(move)
    raise AssertionError('Loop should terminate.')


def paths(pairs):
    name = f'maia_policy_p{pairs:03d}'
    return (OUTPUTS / (name + '_manifest.json'),
            OUTPUTS / (name + '_games.jsonl'),
            OUTPUTS / (name + '_summary.json'),
            OUTPUTS / (name + '_summary.csv'))


def create_settings(openings, engine_path, weights, pairs, max_plies):
    for label, path in CHECKPOINT_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing model {label}: {path}')
    return {'schema': 2, 'maia_search_algorithm': 'policyhead', 'openings_sha256': openings['sha256'],
            'source_kind': openings['source_kind'], 'pairs': pairs,
            'max_plies': max_plies, 'model_hashes': {k: sha256(v) for k,v in CHECKPOINT_FILES.items()},
            'maia_weight_hashes': {k: weights[k]['sha256'] for k in weights},
            'lc0_executable_hash': sha256(engine_path),
            'lc0_executable_name': engine_path.name,
            'maia_uci_nodes': 1, 'our_action_selection': 'one_pass_greedy_legal_mask',
            'candidate_models': list(CHECKPOINT_FILES), 'maia_ratings': list(map(int, weights))}


def append_record(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(row, separators=(',', ':')) + '\n')
        file.flush()
        os.fsync(file.fileno())


def read_records(path):
    if not path.is_file():
        return {}
    existing = {}
    with path.open('r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row['model'], row['maia'], row['opening_id'], row['color'])
            if key in existing:
                raise RuntimeError(f'Duplicate saved match {key}, refusing to count twice.')
            existing[key] = row
    return existing


def summary(records, ratings, pairs, final=False):
    by_group = defaultdict(list)
    for row in records.values():
        by_group[(row['maia'], row['model'])].append(row)
    result = {'schema': SCHEMA, 'pairs_per_matchup': pairs,
              'final': final, 'matches': {}, 'comparisons': {}}
    csv_rows = []
    for rating in ratings:
        for label in CHECKPOINT_FILES:
            group = by_group.get((rating, label), [])
            stats = summarize_games(group, pairs, seed=SEED+int(rating))
            result['matches'][f'{label}_vs_maia{rating}'] = stats
            csv_rows.append({'model': label, 'maia': rating, **stats})
        for label in ('v100','pool_s49','latest_s49'):
            a = by_group.get((rating, label), [])
            b = by_group.get((rating, 'v0'), [])
            result['comparisons'][f'{label}_minus_v0_vs_maia{rating}'] = compare_models(a,b,seed=SEED+int(rating)+3)
        for label in ('pool_s49', 'latest_s49'):
            result['comparisons'][f'{label}_minus_v100_vs_maia{rating}'] = compare_models(
                by_group.get((rating, label), []), by_group.get((rating, 'v100'), []),
                seed=SEED+int(rating)+47)
    return result, csv_rows


def save_summary(records, ratings, pairs, finished=False):
    from chess_rl.ppo_long import safe_csv
    _, _, summary_path, csv_path = paths(pairs)
    content, rows = summary(records, ratings, pairs, finished)
    write_json(summary_path, content)
    safe_csv(rows, csv_path)
    return content


def launch_engine(executable, weights_path):
    # Use an absolute weight path so an accidental different network is impossible.
    # Keep Lc0 and its DLLs together in the same folder.
    if executable.name.lower() != 'lc0-policyhead.exe':
        raise RuntimeError('Maia benchmarking requires the policyhead Lc0 executable.')
    print('Launching Lc0 policyhead (single neural evaluation, no search).', flush=True)
    command = [str(executable), '--weights=' + str(weights_path)]
    optional_backend = os.environ.get('MAIA_LC0_BACKEND', '').strip()
    if optional_backend:
        command.append('--backend=' + optional_backend)
    try:
        engine = chess.engine.SimpleEngine.popen_uci(command, timeout=120)
    except Exception as err:
        raise RuntimeError(f'Could not start Lc0 with Maia weights: {err}\n'
                           'See README_MAIA_BENCHMARK.md for setup and CPU fallback.') from err
    return engine


def evaluate(pairs=64, max_plies=GAME_LIMIT, ratings=MAIAS):
    if pairs < 1 or pairs > OPENING_COUNT or max_plies < 20:
        raise ValueError(f'Choose 1..{OPENING_COUNT} paired openings, max_plies>=20.')
    if not DATA.exists():
        raise FileNotFoundError('Run run.cmd maia-prepare first.')
    openings = verify_openings(saved_json(DATA))
    engine_path = find_lc0()
    weights = available_weights(ratings)
    settings = create_settings(openings, engine_path, weights, pairs, max_plies)
    manifest_path, games_path, _, _ = paths(pairs)
    if manifest_path.exists():
        if saved_json(manifest_path) != settings:
            raise RuntimeError('The saved experiment uses different inputs. Refusing to mix results.')
    else:
        if games_path.exists():
            raise RuntimeError('Games exist without a matching configuration manifest.')
        write_json(manifest_path, settings)
    records = read_records(games_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Maia external benchmark: {pairs} opening pairs, {len(ratings)} Maia ratings, '
          f'{len(CHECKPOINT_FILES)} own checkpoints; device={device}; Lc0 policyhead + go nodes 1.', flush=True)
    print(f'Saved {len(records)} / {2*pairs*len(ratings)*len(CHECKPOINT_FILES)} games already.',flush=True)
    for rating in ratings:
        code = str(rating)
        needed = any((label, code, i, color) not in records
                     for label in CHECKPOINT_FILES for i in range(pairs)
                     for color in ('white','black'))
        if not needed:
            print(f'Maia {rating} already complete.',flush=True)
            continue
        engine = launch_engine(engine_path, Path(weights[code]['path']))
        try:
            for label, checkpoint in CHECKPOINT_FILES.items():
                count = sum((label,code,i,color) in records for i in range(pairs)
                            for color in ('white','black'))
                if count == 2*pairs:
                    print(f'  {label} vs Maia {rating}: already complete.',flush=True)
                    continue
                print(f'  {label} vs Maia {rating}: {2*pairs-count} games remaining.',flush=True)
                model = read_historical(checkpoint, device)
                model.eval()
                try:
                    for i in range(pairs):
                        opening = openings['openings'][i]
                        for color in ('white','black'):
                            key = (label, code, i, color)
                            if key in records:
                                continue
                            start = time.perf_counter()
                            played = run_game(model, engine, opening,
                                              chess.WHITE if color=='white' else chess.BLACK,
                                              device, max_plies)
                            row = {'model':label,'maia':code,'opening_id':i,'color':color,
                                   'opening_uci':' '.join(opening),
                                   'elapsed_seconds':round(time.perf_counter()-start,3),**played}
                            append_record(games_path, row)
                            records[key] = row
                            count += 1
                            if count % 16 == 0 or count == 2*pairs:
                                print(f'    {label} vs Maia {rating}: {count}/{2*pairs}',flush=True)
                finally:
                    del model
                report = save_summary(records, [str(x) for x in ratings], pairs)
                stat = report['matches'][f'{label}_vs_maia{rating}']
                print(f'  {label} vs Maia {rating}: W={stat["wins"]} D={stat["draws"]} '
                      f'L={stat["losses"]} T={stat["truncated"]}, '
                      f'score={stat["score_completed"]}, paired CI={stat["ci95_paired"]}',flush=True)
        finally:
            engine.quit()
    report = save_summary(records, [str(x) for x in ratings], pairs, finished=True)
    print(f'Complete. Report: {paths(pairs)[2]}\nEvery game: {games_path}',flush=True)
    for k, val in report['comparisons'].items():
        print(f'  {k}: delta={val["score_delta"]}, CI={val["ci95_paired_difference"]}',flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['download','prepare','smoke','eval','full'],required=True)
    parser.add_argument('--source',type=Path,default=DEFAULT_PGN)
    parser.add_argument('--pairs',type=int,default=None)
    args=parser.parse_args()
    if args.mode=='download':
        download_weights()
    elif args.mode=='prepare':
        prepare_openings(source=args.source)
    elif args.mode=='smoke':
        evaluate(pairs=2,ratings=(1100,))
    elif args.mode=='eval':
        evaluate(pairs=args.pairs or 64)
    else:
        evaluate(pairs=args.pairs or 128)


if __name__=='__main__':
    main()
