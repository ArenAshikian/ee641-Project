"""Regression tests for PPO games ending before the learner takes a turn."""
import unittest
from unittest.mock import patch

import chess

from chess_rl.ppo_train import rollout_game


class TestTerminalBeforeLearner(unittest.TestCase):
    def test_opponent_checkmates_before_first_learner_move(self):
        # Fool's mate: Black is to move and Qh4# immediately ends the game.
        board = chess.Board()
        for uci in ('f2f3', 'e7e5', 'g2g4'):
            board.push_uci(uci)
        with patch('chess_rl.ppo_train.board_from_opening', return_value=board), \
             patch('chess_rl.ppo_train.one_move', return_value=(chess.Move.from_uci('d8h4'), None)):
            samples, result = rollout_game(None, None, (), chess.WHITE, 'cpu', max_plies=20)
        self.assertEqual(samples, [])
        self.assertTrue(result['terminated'])
        self.assertFalse(result['truncated'])
        self.assertEqual(result['reason'], 'CHECKMATE')
        self.assertEqual(result['outcome_for_learner'], -1.0)
        self.assertEqual(result['plies'], 1)

    def test_already_terminal_opening_still_has_result(self):
        board = chess.Board()
        for uci in ('f2f3', 'e7e5', 'g2g4', 'd8h4'):
            board.push_uci(uci)
        with patch('chess_rl.ppo_train.board_from_opening', return_value=board):
            samples, result = rollout_game(None, None, (), chess.WHITE, 'cpu', max_plies=20)
        self.assertEqual(samples, [])
        self.assertTrue(result['terminated'])
        self.assertEqual(result['outcome_for_learner'], -1.0)
        self.assertEqual(result['plies'], 0)


if __name__ == '__main__':
    unittest.main()
