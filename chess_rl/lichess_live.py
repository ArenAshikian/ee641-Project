"""Lichess Bot API bridge for the frozen, search-free Latest S49 checkpoint.

Uses the SAME greedy neural move function as the Maia benchmark: exactly one
policy forward pass per move, legal moves from python-chess, no search, books,
external engines, or training. Only one live game is handled at a time.

Run: python -m chess_rl.lichess_live selftest|check|upgrade|play
"""
import argparse
import hashlib
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import chess
import torch

from chess_rl.ppo_scale import branch_dir
from chess_rl.ppo_train import OUTPUTS, one_move, read_historical, sha256

API = 'https://lichess.org'
MODEL_FILE = branch_dir('latest') / 's0049.pt'
GAME_LOG = OUTPUTS / 'lichess_latest_s49_games.jsonl'


def reconstruct_board(initial_fen, moves_text):
    """Reconstruct the complete move stack needed by encode_board()."""
    board = chess.Board() if not initial_fen or initial_fen == 'startpos' else chess.Board(initial_fen)
    for text in (moves_text or '').split():
        board.push_uci(text)
    return board


def choose_live_move(model, board, device):
    """Exactly the greedy one-pass move policy used for Maia evaluation."""
    if board.is_game_over(claim_draw=False):
        return None
    move, _ = one_move(model, board, device, sample=False)
    if move not in board.legal_moves:
        raise RuntimeError('Network returned an illegal move.')
    return move


def challenge_reason(challenge, allowed_username='', allow_rated=False):
    """Reject nonstandard, non-3+2, or unexpected incoming challenges."""
    variant = challenge.get('variant') or {}
    if variant.get('key') != 'standard':
        return 'variant'
    clock = challenge.get('timeControl') or {}
    if clock.get('type') != 'clock' or clock.get('limit') != 180 or clock.get('increment') != 2:
        return 'timeControl'
    if challenge.get('rated', False) and not allow_rated:
        return 'casual'
    if allowed_username:
        challenger = challenge.get('challenger') or {}
        username = challenger.get('id') or challenger.get('name') or ''
        if username.lower() != allowed_username.lower():
            return 'generic'
    return ''


def state_from_event(event):
    if event.get('type') == 'gameFull':
        return event.get('state') or {}
    if event.get('type') == 'gameState':
        return event
    return None


def move_to_submit(state, initial_fen, my_color, last_submitted_moves, model, device):
    """Return one legal neural move, unless the move was already submitted."""
    if not state or state.get('status') != 'started':
        return None
    moves_text = state.get('moves') or ''
    if moves_text == last_submitted_moves:
        return None
    board = reconstruct_board(initial_fen, moves_text)
    if board.turn != my_color:
        return None
    return choose_live_move(model, board, device)


class LichessApi:
    def __init__(self, token):
        if not token:
            raise RuntimeError('Set LICHESS_BOT_TOKEN in this Command Prompt first.')
        self.token = token
        self.write_lock = threading.Lock()

    def request(self, path, method='GET', data=None):
        """Small JSON POST/GET client; retry 429 conservatively."""
        headers = {'Authorization': f'Bearer {self.token}', 'Accept': 'application/json',
                   'User-Agent': 'EE641-OnePassChessBot/1.0'}
        if method == 'POST':
            headers['Content-Type'] = 'application/x-www-form-urlencoded'
        request = urllib.request.Request(API + path, data=(data if method == 'POST' else None),
                                         method=method, headers=headers)
        for attempt in range(3):
            try:
                with self.write_lock:
                    with urllib.request.urlopen(request, timeout=25) as response:
                        body = response.read().decode('utf-8')
                return json.loads(body) if body.strip() else {}
            except urllib.error.HTTPError as error:
                if error.code == 429 and attempt < 2:
                    print('Lichess rate limit (429). Pausing 65 seconds.', flush=True)
                    time.sleep(65)
                    continue
                message = error.read(250).decode('utf-8', 'replace')
                raise RuntimeError(f'Lichess HTTP {error.code} on {path}: {message}') from error
        raise RuntimeError(f'Failed request: {path}')

    def events(self, path):
        headers = {'Authorization': f'Bearer {self.token}',
                   'Accept': 'application/x-ndjson',
                   'User-Agent': 'EE641-OnePassChessBot/1.0'}
        request = urllib.request.Request(API + path, headers=headers)
        with urllib.request.urlopen(request, timeout=90) as response:
            for raw in response:
                if raw.strip():
                    yield json.loads(raw.decode('utf-8'))

    def account(self):
        return self.request('/api/account')

    def accept(self, challenge_id):
        return self.request('/api/challenge/' + challenge_id + '/accept', method='POST', data=b'')

    def decline(self, challenge_id, reason='generic'):
        return self.request('/api/challenge/' + challenge_id + '/decline?reason=' + reason,
                            method='POST', data=b'')

    def move(self, game_id, uci):
        return self.request('/api/bot/game/' + game_id + '/move/' + uci, method='POST', data=b'')


def append_game(record):
    GAME_LOG.parent.mkdir(parents=True, exist_ok=True)
    # Avoid logging duplicate final results if the bot is restarted.
    if GAME_LOG.exists():
        for line in GAME_LOG.read_text(encoding='utf-8').splitlines():
            if line.strip() and json.loads(line).get('game_id') == record['game_id']:
                return
    with GAME_LOG.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, sort_keys=True) + '\n')
        file.flush()
        os.fsync(file.fileno())


class LiveBot:
    def __init__(self, api, model, device, me, checkpoint_hash, allowed_username='', allow_rated=False):
        self.api, self.model, self.device = api, model, device
        self.me = me.lower()
        self.checkpoint_hash = checkpoint_hash
        self.allowed_username = allowed_username
        self.allow_rated = allow_rated
        self.active = set()
        self.pending = set()
        self.state_lock = threading.Lock()

    def stream_game(self, game_id):
        """The game stream owns the board state; do not speculate on unseen moves."""
        initial_fen = 'startpos'
        my_color = None
        other_user = None
        state = None
        last_submitted = None
        initial_clock = None
        rated = None
        for _ in range(20):
            try:
                for event in self.api.events('/api/bot/game/stream/' + game_id):
                    kind = event.get('type')
                    if kind == 'gameFull':
                        initial_fen = event.get('initialFen') or 'startpos'
                        white = (event.get('white') or {}).get('id', '').lower()
                        black = (event.get('black') or {}).get('id', '').lower()
                        if self.me not in (white, black):
                            raise RuntimeError(f'Game {game_id} is not our bot account.')
                        my_color = chess.WHITE if self.me == white else chess.BLACK
                        other_user = black if my_color == chess.WHITE else white
                        initial_clock = event.get('clock')
                        rated = event.get('rated')
                    updated = state_from_event(event)
                    if updated is None:
                        continue
                    state = updated
                    if state.get('status') != 'started':
                        self.record_result(game_id, state, my_color, other_user, rated, initial_clock)
                        print(f'Game finished: https://lichess.org/{game_id} ({state.get("status")})', flush=True)
                        return
                    if my_color is None:
                        continue
                    move = move_to_submit(state, initial_fen, my_color, last_submitted,
                                          self.model, self.device)
                    if move is None:
                        continue
                    moves_text = state.get('moves') or ''
                    print(f'  {game_id} ply {len(moves_text.split()) + 1}: {move.uci()}', flush=True)
                    self.api.move(game_id, move.uci())
                    last_submitted = moves_text
                print(f'Game stream ended; reconnecting {game_id}.', flush=True)
            except (OSError, TimeoutError, socket.timeout, urllib.error.URLError) as error:
                print(f'Game stream reconnecting {game_id}: {error}', flush=True)
            except RuntimeError as error:
                print(f'Game {game_id}: {error}', flush=True)
                if 'HTTP 401' in str(error) or 'HTTP 403' in str(error):
                    return
            time.sleep(3)
        print(f'Game stream retries exhausted for {game_id}; check Lichess game status.', flush=True)

    def record_result(self, game_id, state, my_color, other_user, rated, initial_clock):
        winner = state.get('winner')
        color = 'white' if my_color == chess.WHITE else 'black' if my_color == chess.BLACK else None
        score = None if state.get('status') in ('aborted', 'noStart') else (.5 if not winner else 1.0 if color == winner else 0.0)
        append_game({'game_id': game_id, 'time_utc': datetime.now(timezone.utc).isoformat(),
                     'model': 'latest_s49', 'checkpoint_sha256': self.checkpoint_hash,
                     'opponent': other_user, 'our_color': color, 'rated': rated,
                     'clock': initial_clock, 'status': state.get('status'),
                     'winner': winner, 'score': score, 'moves_uci': state.get('moves', '')})

    def game_worker(self, game_id):
        try:
            self.stream_game(game_id)
        finally:
            with self.state_lock:
                self.active.discard(game_id)
            print(f'Now available for another 3+2 challenge.', flush=True)

    def handle_event(self, event):
        kind = event.get('type')
        if kind == 'challenge':
            challenge = event.get('challenge') or {}
            challenge_id = challenge.get('id')
            if not challenge_id:
                return
            reason = challenge_reason(challenge, self.allowed_username, self.allow_rated)
            with self.state_lock:
                if self.active or self.pending:
                    reason = 'later'
                if not reason:
                    self.pending.add(challenge_id)
            if reason:
                print(f'Declining challenge {challenge_id}: {reason}', flush=True)
                self.api.decline(challenge_id, reason)
            else:
                print(f'Accepting casual 3+2 challenge {challenge_id}', flush=True)
                try:
                    self.api.accept(challenge_id)
                except Exception:
                    with self.state_lock:
                        self.pending.discard(challenge_id)
                    raise
        elif kind in ('challengeCanceled', 'challengeDeclined'):
            item = event.get('challenge') or {}
            with self.state_lock:
                self.pending.discard(item.get('id'))
        elif kind == 'gameStart':
            game_id = (event.get('game') or {}).get('id')
            if not game_id:
                return
            with self.state_lock:
                self.pending.clear()
                if game_id in self.active:
                    return
                if self.active:
                    print(f'Additional active game {game_id}. Stop other games before playing.', flush=True)
                    return
                self.active.add(game_id)
            print(f'Live game: https://lichess.org/{game_id}', flush=True)
            thread = threading.Thread(target=self.game_worker, args=(game_id,), daemon=True)
            thread.start()
        elif kind == 'gameFinish':
            pass

    def play(self):
        mode = 'rated + casual' if self.allow_rated else 'casual only'
        print(f'Bot online as {self.me} | Latest S49 | 3+2 standard | {mode}', flush=True)
        print('NO SEARCH: same one-forward-pass move choice used in Maia tournament.', flush=True)
        print('Invite: https://lichess.org/@/' + self.me, flush=True)
        print('Stop with Ctrl+C. This program does not train or modify the checkpoint.', flush=True)
        while True:
            try:
                for event in self.api.events('/api/stream/event'):
                    try:
                        self.handle_event(event)
                    except Exception as error:
                        print(f'Event handling error: {error}', flush=True)
                print('Event stream ended. Reconnecting.', flush=True)
            except urllib.error.HTTPError as error:
                print(f'Event HTTP {error.code}.', flush=True)
                if error.code in (401, 403):
                    raise RuntimeError('Bot account or API token is not authorized.') from error
                if error.code == 429:
                    time.sleep(65)
            except (OSError, TimeoutError, socket.timeout, urllib.error.URLError) as error:
                print(f'Event stream reconnecting: {error}', flush=True)
            time.sleep(3)


def load_network():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError('Missing frozen Latest S49 checkpoint: ' + str(MODEL_FILE))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = read_historical(MODEL_FILE, device)
    model.eval()
    return model, device, sha256(MODEL_FILE)


def local_selftest():
    model, device, fingerprint = load_network()
    board = chess.Board()
    print('Frozen model: ' + str(MODEL_FILE))
    print('SHA-256: ' + fingerprint)
    print('Device: ' + str(device))
    for i in range(4):
        move = choose_live_move(model, board, device)
        if move not in board.legal_moves:
            raise RuntimeError('Illegal move in self-test.')
        print(f'  ply {i+1}: {board.san(move)} ({move.uci()})')
        board.push(move)
    print('Local neural self-test successful. No Lichess connection used.')


def run_command(command):
    if command == 'selftest':
        local_selftest()
        return
    token = os.environ.get('LICHESS_BOT_TOKEN', '').strip()
    api = LichessApi(token)
    account = api.account()
    username = account.get('username') or account.get('id') or ''
    title = account.get('title') or ''
    print('Authenticated Lichess account: ' + username + ' | title=' + (title or '(none)'), flush=True)
    if command == 'upgrade':
        if title == 'BOT':
            print('Account already upgraded to BOT.')
            return
        print('WARNING: upgrading a new, zero-game account to BOT is IRREVERSIBLE.')
        print('Do not do this to your personal Lichess account.')
        if input('Type UPGRADE to permanently convert this account: ').strip() != 'UPGRADE':
            print('Canceled. No account changes made.')
            return
        print(api.request('/api/bot/account/upgrade', method='POST', data=b''))
        print('Upgrade request submitted. Verify BOT title before starting live games.')
        return
    if title != 'BOT':
        raise RuntimeError('This is not a BOT account. Create a new zero-game account and upgrade it first.')
    model, device, fingerprint = load_network()
    print('Latest S49 verified (read only), CUDA=' + str(device) + ', SHA256=' + fingerprint, flush=True)
    if command == 'check':
        print('Ready. Run: run.cmd lichess-play')
        return
    allowed_username = os.environ.get('LICHESS_ALLOWED_CHALLENGER', '').strip()
    allow_rated = os.environ.get('LICHESS_ALLOW_RATED', '0').strip() == '1'
    LiveBot(api, model, device, account.get('id') or username, fingerprint,
            allowed_username, allow_rated).play()


def main():
    parser = argparse.ArgumentParser(description='Play Latest S49 on Lichess with no chess search.')
    parser.add_argument('command', choices=('selftest', 'check', 'upgrade', 'play'))
    args = parser.parse_args()
    try:
        run_command(args.command)
    except KeyboardInterrupt:
        print('\nStopped. No training files modified.')


if __name__ == '__main__':
    main()
