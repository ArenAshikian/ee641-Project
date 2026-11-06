"""Stream Lichess .pgn.zst archives and build a small supervised pilot dataset.

The trainer will eventually scale this into sharded data for millions of samples.
This module deliberately loads only a bounded number of positions into memory.
"""

import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import zstandard

from chess_rl.encoding import action_for_move, encode_board


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PGN = PROJECT_ROOT / "data" / "lichess" / "lichess_db_standard_rated_2026-09.pgn.zst"


def read_raw_games(handle):
    """Yield one raw PGN game at a time without parsing unselected games.

    Lichess PGN archives begin every game with a new [Event ...] header.
    """
    lines = []
    for line in handle:
        if line.startswith('[Event ') and lines:
            yield ''.join(lines)
            lines = [line]
        elif lines or line.startswith('[Event '):
            lines.append(line)
    if lines:
        yield ''.join(lines)


def read_headers(raw_pgn):
    """Read common PGN tags without running the expensive chess move parser."""
    headers = {}
    for line in raw_pgn.splitlines():
        if not line.startswith('['):
            if headers:
                break
            continue
        try:
            key, value = line[1:-1].split(' ', 1)
            headers[key] = value.strip().strip('"')
        except ValueError:
            continue
    return headers


def selection_reason(headers):
    """Return 'accepted' or a named rejection reason for one complete game."""
    event = headers.get('Event', '').lower()
    if not event.startswith('rated ') or not ('blitz' in event or 'rapid' in event):
        return 'not_rated_blitz_or_rapid'
    if headers.get('Variant', 'Standard').lower() not in ('standard', 'chess'):
        return 'not_standard_chess'
    try:
        white_elo = int(headers['WhiteElo'])
        black_elo = int(headers['BlackElo'])
    except (KeyError, ValueError):
        return 'missing_rating'
    if not (1800 <= white_elo <= 2200 and 1800 <= black_elo <= 2200):
        return 'rating_out_of_range'
    if headers.get('WhiteTitle', '').upper() == 'BOT' or headers.get('BlackTitle', '').upper() == 'BOT':
        return 'bot_game'
    if headers.get('Result') not in ('1-0', '0-1', '1/2-1/2'):
        return 'missing_result'
    termination = headers.get('Termination', '').lower()
    if any(word in termination for word in ('time forfeit', 'abandon', 'unterminated', 'disconnect')):
        return 'non_board_termination'
    return 'accepted'


def validation_game(headers):
    """Assign whole games (not individual positions) to a 10% pilot validation set."""
    site = headers.get('Site', '')
    identity = site or '|'.join(headers.get(x, '') for x in ('Date', 'White', 'Black', 'Result'))
    digest = hashlib.sha256(identity.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'big') % 10 == 0


def positions_from_game(game, max_positions_per_game=8, seed=7, include_legal_count=False):
    """Yield (board tensor, move class, game outcome from player-to-move view).

    The PGN clock is the clock AFTER the move. We conservatively exclude moves
    if its recorded remaining clock is <30 seconds, but keep unknown clocks.
    """
    result = game.headers['Result']
    winner = chess.WHITE if result == '1-0' else chess.BLACK if result == '0-1' else None
    nodes = list(game.mainline())
    good = []
    for i, node in enumerate(nodes):
        after_move_seconds = node.clock()
        if after_move_seconds is None or after_move_seconds >= 30:
            good.append(i)
    if not good:
        return

    rng = random.Random(seed)
    chosen = set(rng.sample(good, min(len(good), max_positions_per_game)))
    board = game.board()
    for i, node in enumerate(nodes):
        if i in chosen:
            outcome = 0.0 if winner is None else (1.0 if winner == board.turn else -1.0)
            example = (encode_board(board), action_for_move(board, node.move), outcome)
            if include_legal_count:
                yield (*example, board.legal_moves.count())
            else:
                yield example
        board.push(node.move)


def build_pilot_dataset(source=DEFAULT_PGN, max_positions=3000, max_positions_per_game=8,
                        seed=7, folder=None, max_scanned_games=500000):
    """Read compressed PGN and write small train/validation .npz files.

    This is a pilot, NOT the promised 20-million-position research dataset.
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(
            f'No Lichess archive found at {source}\n'
            'Move the September 2026 .pgn.zst file to data\\lichess\\ first.'
        )
    folder = Path(folder) if folder else PROJECT_ROOT / 'data' / 'processed'
    folder.mkdir(parents=True, exist_ok=True)
    train_x, train_a, train_v = [], [], []
    valid_x, valid_a, valid_v = [], [], []
    reasons = Counter()
    game_count = 0
    accepted_games = 0

    print('Streaming from:', source, flush=True)
    print('Extracting up to', max_positions, 'positions. Archive stays compressed.', flush=True)
    with zstandard.open(source, 'rt', encoding='utf-8', errors='replace') as handle:
        for raw in read_raw_games(handle):
            game_count += 1
            headers = read_headers(raw)
            reason = selection_reason(headers)
            reasons[reason] += 1
            if reason != 'accepted':
                if game_count >= max_scanned_games:
                    break
                continue
            game = chess.pgn.read_game(io.StringIO(raw))
            if game is None or game.errors:
                reasons['parse_error'] += 1
                continue
            accepted_games += 1
            rng_seed = seed + int.from_bytes(hashlib.sha256(
                headers.get('Site', str(game_count)).encode('utf-8')).digest()[:4], 'big')
            is_valid = validation_game(headers)
            xs, actions, values = (valid_x, valid_a, valid_v) if is_valid else (train_x, train_a, train_v)
            for x, action, outcome in positions_from_game(game, max_positions_per_game, rng_seed):
                if len(train_a) + len(valid_a) >= max_positions:
                    break
                xs.append(x)
                actions.append(action)
                values.append(outcome)
            if accepted_games % 50 == 0:
                print(f'  Scanned {game_count:,} games; accepted {accepted_games:,}; '
                      f'positions {len(train_a) + len(valid_a):,}/{max_positions:,}', flush=True)
            if len(train_a) + len(valid_a) >= max_positions or game_count >= max_scanned_games:
                break

    if not train_a or not valid_a:
        raise RuntimeError('Too few valid positions for training/validation. '
                           'Try a larger --positions setting and check the PGN file.')

    # Float16 reduces pilot disk usage and preserves the normalized halfmove plane.
    np.savez_compressed(folder / 'pilot_train.npz',
                        boards=np.asarray(train_x, dtype=np.float16),
                        actions=np.asarray(train_a, dtype=np.int32),
                        outcomes=np.asarray(train_v, dtype=np.float32))
    np.savez_compressed(folder / 'pilot_val.npz',
                        boards=np.asarray(valid_x, dtype=np.float16),
                        actions=np.asarray(valid_a, dtype=np.int32),
                        outcomes=np.asarray(valid_v, dtype=np.float32))
    stats = {
        'source': source.name,
        'seed': seed,
        'games_scanned': game_count,
        'games_accepted': accepted_games,
        'train_positions': len(train_a),
        'validation_positions': len(valid_a),
        'excluded_or_accepted_games_by_reason': dict(reasons),
        'notes': [
            'Pilot only; September games split by game ID for training and validation.',
            'An independent October archive is still needed for final testing.',
            'Only game-result labels are used. PGN engine evaluations are ignored.',
            'PGN clock is recorded after moves; <30 second post-move clocks are excluded.',
            'Games ending by time forfeit/abandonment are excluded from value training.',
        ],
    }
    (folder / 'pilot_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(f'Finished: {len(train_a):,} train / {len(valid_a):,} validation positions; '
          f'{accepted_games:,} accepted games.', flush=True)
    print('Stats:', folder / 'pilot_stats.json', flush=True)
    return folder
