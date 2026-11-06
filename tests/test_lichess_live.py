"""Offline checks for the Lichess bridge. No token or network calls."""
import unittest
from unittest.mock import patch

import chess

from chess_rl.lichess_live import (LiveBot, challenge_reason, reconstruct_board,
                                  state_from_event, move_to_submit)


class FakeApi:
    def __init__(self):
        self.accepted = []
        self.declined = []
    def accept(self, challenge_id):
        self.accepted.append(challenge_id)
    def decline(self, challenge_id, reason):
        self.declined.append((challenge_id, reason))


class TestLive(unittest.TestCase):
    def challenge(self, rated=False, limit=180, increment=2):
        return {'id': 'abcd', 'variant': {'key': 'standard'}, 'rated': rated,
                'timeControl': {'type': 'clock', 'limit': limit, 'increment': increment},
                'challenger': {'id': 'MyFriend'}}

    def test_only_standard_3plus2(self):
        c = self.challenge()
        self.assertEqual(challenge_reason(c, 'myfriend'), '')
        self.assertEqual(challenge_reason(c, 'another'), 'generic')
        self.assertEqual(challenge_reason(self.challenge(limit=600)), 'timeControl')
        self.assertEqual(challenge_reason(self.challenge(rated=True)), 'casual')
        self.assertEqual(challenge_reason(self.challenge(rated=True), allow_rated=True), '')
        c['variant'] = {'key': 'chess960'}
        self.assertEqual(challenge_reason(c), 'variant')

    def test_move_stack_preserved(self):
        board = reconstruct_board('startpos', 'e2e4 e7e5 g1f3')
        self.assertEqual(len(board.move_stack), 3)
        self.assertEqual(board.turn, chess.BLACK)
        self.assertEqual(board.fullmove_number, 2)

    def test_one_inference_and_no_duplicate(self):
        state = {'status': 'started', 'moves': ''}
        with patch('chess_rl.lichess_live.one_move', return_value=(chess.Move.from_uci('e2e4'), None)) as fn:
            move = move_to_submit(state, 'startpos', chess.WHITE, None, object(), 'cpu')
            self.assertEqual(move.uci(), 'e2e4')
            self.assertEqual(fn.call_count, 1)
            duplicate = move_to_submit(state, 'startpos', chess.WHITE, '', object(), 'cpu')
            self.assertIsNone(duplicate)
            self.assertEqual(fn.call_count, 1)
            state['moves'] = 'e2e4'
            self.assertIsNone(move_to_submit(state, 'startpos', chess.WHITE, None, object(), 'cpu'))
            self.assertEqual(fn.call_count, 1)

    def test_ignore_game_over(self):
        state = {'status': 'mate', 'moves': 'f2f3 e7e5 g2g4 d8h4'}
        with patch('chess_rl.lichess_live.one_move') as fn:
            self.assertIsNone(move_to_submit(state, 'startpos', chess.WHITE, None, object(), 'cpu'))
            fn.assert_not_called()

    def test_game_state(self):
        self.assertEqual(state_from_event({'type': 'gameFull', 'state': {'moves': 'e2e4'}}), {'moves': 'e2e4'})
        self.assertEqual(state_from_event({'type': 'gameState', 'moves': 'd2d4'})['moves'], 'd2d4')
        self.assertIsNone(state_from_event({'type': 'chatLine'}))

    def test_challenge_accept_and_decline(self):
        api = FakeApi()
        bot = LiveBot(api, None, 'cpu', 'testbot', 'hash', allowed_username='myfriend')
        bot.handle_event({'type': 'challenge', 'challenge': self.challenge()})
        self.assertEqual(api.accepted, ['abcd'])
        self.assertEqual(len(bot.pending), 1)
        another = self.challenge()
        another['id'] = 'efgh'
        bot.handle_event({'type': 'challenge', 'challenge': another})
        self.assertEqual(api.declined, [('efgh', 'later')])
        bot.handle_event({'type': 'challengeCanceled', 'challenge': {'id': 'abcd'}})
        self.assertEqual(len(bot.pending), 0)


if __name__ == '__main__':
    unittest.main()
