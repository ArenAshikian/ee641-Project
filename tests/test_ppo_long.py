"""Regression checks for extended search-free PPO. Uses no real checkpoints."""
import unittest
from unittest.mock import patch

import chess

from chess_rl.ppo_long import (check_start, checkpoint_for, paired_interval,
                               play_eval_game, result_score, settings)


class TestLongPPO(unittest.TestCase):
    def test_v10_archive_shape_is_required(self):
        saved = {'iteration': 10, 'model_blocks': 10, 'model_channels': 128,
                 'model_state_dict': {}, 'optimizer_state_dict': {}}
        check_start(saved)
        with self.assertRaises(RuntimeError):
            check_start({**saved, 'iteration': 9})
        with self.assertRaises(RuntimeError):
            check_start({**saved, 'model_channels': 64})
        with self.assertRaises(RuntimeError):
            check_start({k: v for k, v in saved.items() if k != 'optimizer_state_dict'})

    def test_archive_paths_do_not_reuse_v10(self):
        self.assertEqual(checkpoint_for(10).name, 'ppo_v0010.pt')
        self.assertEqual(checkpoint_for(20).name, 'ppo_v0020.pt')
        self.assertEqual(checkpoint_for(100).name, 'ppo_v0100.pt')

    def test_run_settings_lock_training_configuration(self):
        a = settings('shaA', 'shaB', 128, 7, 240, 2, 64, 1e-5)
        b = settings('shaA', 'shaB', 128, 7, 240, 2, 64, 1e-5)
        c = settings('shaA', 'shaB', 32, 7, 240, 2, 64, 1e-5)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_win_draw_loss_scoring(self):
        self.assertEqual(result_score(chess.WHITE, chess.WHITE), 1)
        self.assertEqual(result_score(chess.BLACK, chess.WHITE), 0)
        self.assertEqual(result_score(None, chess.WHITE), 0.5)

    def test_paired_bootstrap_interval(self):
        self.assertEqual(paired_interval([[1, 1]] * 15, 8), [1.0, 1.0])
        self.assertIsNone(paired_interval([[1, None]] * 15, 8))
        interval = paired_interval([[1, 0], [0, 1], [1, 1], [0, 0]] * 10, 2)
        self.assertTrue(0 <= interval[0] <= interval[1] <= 1)

    def test_cutoff_is_not_a_draw(self):
        def first_legal(model, board, device, sample=False):
            return next(iter(board.legal_moves)), None
        with patch('chess_rl.ppo_long.one_move', side_effect=first_legal):
            result = play_eval_game(object(), object(), [], chess.WHITE, 'cpu',
                                    max_plies=4, seed=1)
        self.assertTrue(result['truncated'])
        self.assertIsNone(result['score'])
        self.assertEqual(result['termination'], 'MAX_PLIES')

    def test_fools_mate_scores_opponent_win(self):
        board = chess.Board()
        for move in ('f2f3', 'e7e5', 'g2g4'):
            board.push_uci(move)
        with patch('chess_rl.ppo_long.board_from_opening', return_value=board), \
             patch('chess_rl.ppo_long.one_move', return_value=(chess.Move.from_uci('d8h4'), None)):
            result = play_eval_game(object(), object(), [], chess.WHITE, 'cpu',
                                    max_plies=4, seed=1)
        self.assertFalse(result['truncated'])
        self.assertEqual(result['termination'], 'CHECKMATE')
        self.assertEqual(result['score'], 0.0)


if __name__ == '__main__':
    unittest.main()
