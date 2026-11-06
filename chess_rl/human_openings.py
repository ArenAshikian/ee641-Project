"""Human-game start positions for controlled PPO research.

Uses September 2026 Lichess games filtered the same way as V0.
V0 training games feed PPO rollout starts; V0 validation games are used
ONLY for evaluations. PGN moves are used as starting states, never labels
or external reward signals for PPO.
"""

import hashlib
import io
import json
import os
import random
from pathlib import Path

import chess
import chess.pgn
import zstandard

from chess_rl.lichess_data import (DEFAULT_PGN, read_raw_games, read_headers,
                                  selection_reason, validation_game)
from chess_rl.ppo_train import board_from_opening

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'data' / 'processed' / 'ppo_scale_human_openings.json'
DEFAULT_TRAIN = 8192
DEFAULT_EVAL = 256
SEED = 6412026


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(content, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def file_id(path):
    """Fast source identity, not a cryptographic hash of the huge PGN."""
    info = path.stat()
    return {'path': str(path.resolve()), 'size': info.st_size,
            'modified_ns': info.st_mtime_ns}


def digest_openings(manifest):
    values = {'train': manifest['train'], 'eval': manifest['eval'],
              'source': manifest['source'], 'seed': manifest['seed']}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode('utf-8')).hexdigest()


def opening_from_game(game, game_id, seed):
    """Take one realistically reached early/middle opening position per game."""
    rng = random.Random(seed + int.from_bytes(hashlib.sha256(game_id.encode()).digest()[:8], 'big'))
    total = game.end().ply()
    even_lengths = [p for p in range(8, 21, 2) if p + 8 <= total]
    if not even_lengths:
        return None
    chosen = rng.choice(even_lengths)
    board = game.board()
    moves = []
    for node in game.mainline():
        board.push(node.move)
        moves.append(node.move.uci())
        if len(moves) == chosen:
            break
    if len(moves) != chosen or board.is_game_over(claim_draw=True):
        return None
    return moves


def validate_manifest(doc, train_count=DEFAULT_TRAIN, eval_count=DEFAULT_EVAL):
    if doc.get('schema') != 1 or len(doc.get('train', [])) < train_count or len(doc.get('eval', [])) < eval_count:
        raise RuntimeError('Human opening manifest is missing or has fewer starts than requested.')
    if doc.get('sha256') != digest_openings(doc):
        raise RuntimeError('Human opening manifest content changed; refusing to mix runs.')
    seen = set()
    for partition in ('train', 'eval'):
        for opening in doc[partition]:
            key = tuple(opening)
            if key in seen:
                raise RuntimeError('Training and evaluation share a start position.')
            seen.add(key)
            board = board_from_opening(opening)
            if board.is_game_over(claim_draw=True):
                raise RuntimeError('Human starting position is already terminal.')
    return doc


def prepare(source=DEFAULT_PGN, train_count=DEFAULT_TRAIN, eval_count=DEFAULT_EVAL,
            seed=SEED, max_scanned=1500000):
    """Build fixed disjoint starting positions once, then reuse forever."""
    source = Path(source)
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        if (manifest.get('seed') != seed or manifest.get('source') != file_id(source) or
                manifest.get('requested') != {'train': train_count, 'eval': eval_count}):
            raise RuntimeError('Existing human opening manifest uses different settings. '
                               'Do not delete it while training results depend on it.')
        return validate_manifest(manifest, train_count, eval_count)
    if not source.is_file():
        raise FileNotFoundError(f'Expected Lichess PGN archive: {source}')
    train, eval_set = [], []
    train_keys, eval_keys = set(), set()
    counts = {'scanned': 0, 'accepted': 0, 'parsed': 0, 'train': 0, 'eval': 0}
    print('Building held-out HUMAN opening pool from:', source, flush=True)
    print('  V0 training partition for PPO rollouts; V0 validation partition for tests.', flush=True)
    with zstandard.open(source, 'rt', encoding='utf-8', errors='replace') as stream:
        for raw in read_raw_games(stream):
            counts['scanned'] += 1
            if selection_reason(read_headers(raw)) != 'accepted':
                if counts['scanned'] >= max_scanned:
                    break
                continue
            header = read_headers(raw)
            validation = validation_game(header)
            if validation and len(eval_set) >= eval_count:
                continue
            if not validation and len(train) >= train_count:
                continue
            counts['accepted'] += 1
            parsed = chess.pgn.read_game(io.StringIO(raw))
            if parsed is None or parsed.errors:
                continue
            counts['parsed'] += 1
            game_id = header.get('Site') or hashlib.sha256(raw.encode()).hexdigest()
            moves = opening_from_game(parsed, game_id, seed)
            if moves is None:
                continue
            # Opening sequences rather than FEN preserve 4-position history and repetitions.
            key = tuple(moves)
            if key in train_keys or key in eval_keys:
                continue
            if validation:
                eval_set.append(moves)
                eval_keys.add(key)
            else:
                train.append(moves)
                train_keys.add(key)
            if counts['scanned'] % 50000 < 10:
                print(f'  scanned {counts["scanned"]:,} / starts: {len(train):,} train, '
                      f'{len(eval_set):,} held-out', flush=True)
            if len(train) >= train_count and len(eval_set) >= eval_count:
                break
            if counts['scanned'] >= max_scanned:
                break
    if len(train) < train_count or len(eval_set) < eval_count:
        raise RuntimeError(f'Not enough accepted human games ({len(train)} train, '
                           f'{len(eval_set)} held-out). Increase max_scanned or check source.')
    manifest = {
        'schema': 1, 'seed': seed, 'requested': {'train': train_count, 'eval': eval_count},
        'source': file_id(source), 'counts': counts,
        'split': ('Same whole-game deterministic hash partition as supervised V0: '
                  'validation_game(headers) = 10% held-out; 90% training.'),
        'training_note': ('Starting sequences only. PPO never uses human next-move '
                          'labels or game-result annotations. No chess engine.'),
        'train': train, 'eval': eval_set,
    }
    manifest['sha256'] = digest_openings(manifest)
    write_json(MANIFEST, manifest)
    print(f'Saved {len(train):,} training and {len(eval_set):,} held-out starts:',
          MANIFEST, flush=True)
    return manifest


def training_openings(manifest, count, seed):
    """Paired order: same start twice with colors swapped each iteration."""
    if count % 2 or count < 2:
        raise ValueError('Games per iteration must be positive and even.')
    rng = random.Random(seed)
    n = count // 2
    if n <= len(manifest['train']):
        indices = rng.sample(range(len(manifest['train'])), n)
    else:
        indices = [rng.randrange(len(manifest['train'])) for _ in range(n)]
    return [manifest['train'][idx] for idx in indices for _ in range(2)]
