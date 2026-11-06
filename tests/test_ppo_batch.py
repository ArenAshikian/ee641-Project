"""Small CPU-only tests for the read-only batched PPO rollout path."""

import unittest

import chess
import numpy as np
import torch

from chess_rl.network import ChessNetwork
from chess_rl.ppo_batch import batched_moves, rollout_latest_batched
from chess_rl.ppo_train import one_move, rollout_game


class TestBatchedSearchFreeRollout(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(641)
        self.model = ChessNetwork(channels=8, blocks=0).eval()
        self.device = torch.device('cpu')

    def test_greedy_move_and_ppo_record_match_single(self):
        boards = []
        for moves in [[], ['e2e4'], ['e2e4', 'e7e5'],
                      ['d2d4', 'd7d5', 'c2c4'], ['g1f3', 'd7d5']]:
            board = chess.Board()
            for move in moves:
                board.push_uci(move)
            boards.append(board)
        batched = batched_moves(self.model, boards, self.device, sample=False,
                                record_flags=[True] * len(boards))
        for board, (move, record) in zip(boards, batched):
            expected_move, expected = one_move(self.model, board, self.device,
                                               sample=False, keep_record=True)
            self.assertEqual(move, expected_move)
            self.assertTrue(board.is_legal(move))
            self.assertEqual(record['action'], expected['action'])
            np.testing.assert_array_equal(record['mask'], expected['mask'])
            np.testing.assert_array_equal(record['board'], expected['board'])
            self.assertAlmostEqual(record['value'], expected['value'], places=4)
            self.assertAlmostEqual(record['old_log_prob'], expected['old_log_prob'], places=4)

    def test_sampled_actions_are_legal_and_records_only_for_learner(self):
        boards = [chess.Board(), chess.Board()]
        boards[1].push_uci('e2e4')
        selected = batched_moves(self.model, boards, self.device, sample=True,
                                 record_flags=[True, False])
        self.assertTrue(boards[0].is_legal(selected[0][0]))
        self.assertTrue(boards[1].is_legal(selected[1][0]))
        self.assertIsNotNone(selected[0][1])
        self.assertIsNone(selected[1][1])

    def test_one_game_rollout_equivalence_and_cutoff(self):
        opening = ['e2e4', 'e7e5']
        for color in (chess.WHITE, chess.BLACK):
            torch.manual_seed(511)
            old_samples, old_result = rollout_game(self.model, self.model, opening,
                                                   color, self.device, max_plies=8)
            torch.manual_seed(511)
            new_samples, new_result = rollout_latest_batched(
                self.model, [opening], [color], self.device, max_plies=8)[0]
            self.assertEqual(new_result, old_result)
            self.assertEqual(len(new_samples), len(old_samples))
            for old, new in zip(old_samples, new_samples):
                self.assertEqual(old['action'], new['action'])
                np.testing.assert_array_equal(old['mask'], new['mask'])
                np.testing.assert_array_equal(old['board'], new['board'])
                for value in ('old_log_prob', 'value', 'advantage', 'return'):
                    self.assertAlmostEqual(old[value], new[value], places=4)

    def test_multiple_games_preserve_order_and_experience_structure(self):
        openings = [['e2e4', 'e7e5'], ['e2e4', 'e7e5'],
                    ['d2d4', 'd7d5'], ['d2d4', 'd7d5']]
        results = rollout_latest_batched(self.model, openings,
                                        [chess.WHITE, chess.BLACK,
                                         chess.WHITE, chess.BLACK],
                                        self.device, max_plies=12)
        self.assertEqual(len(results), len(openings))
        for samples, result in results:
            self.assertTrue(result['truncated'] or result['terminated'])
            for item in samples:
                self.assertEqual(item['board'].shape, (61, 8, 8))
                self.assertEqual(item['mask'].shape, (4672,))
                self.assertTrue(item['mask'][item['action']])
                self.assertTrue(np.isfinite(item['return']))
                self.assertTrue(np.isfinite(item['advantage']))


if __name__ == '__main__':
    unittest.main()
