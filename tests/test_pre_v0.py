"""Quick deterministic tests for the pre-V0 evaluation and benchmark helpers."""

import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

from chess_rl.benchmark_v0 import load_real_batches
from chess_rl.evaluate_v0 import opening_positions, paired_bootstrap_ci, play_game, summary


class TestPreV0(unittest.TestCase):
    def test_openings_reproducible_and_unique(self):
        a = opening_positions(12, 7)
        b = opening_positions(12, 7)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 12)
        for fen in a:
            board = chess.Board(fen)
            self.assertTrue(board.is_valid())
            self.assertFalse(board.is_game_over(claim_draw=True))

    def test_terminal_result_is_not_truncation(self):
        # Checkmate is already on the board: no network inference should occur.
        fen = '7k/6Q1/5K2/8/8/8/8/8 b - - 0 1'
        status, plies, score = play_game(None, None, 'random_legal', fen,
                                         chess.WHITE, 123, 'cpu', max_plies=5)
        self.assertTrue(status.terminated)
        self.assertFalse(status.truncated)
        self.assertEqual(plies, 0)
        self.assertEqual(score, 1.0)

    def test_truncation_excluded_from_score(self):
        row = lambda idx, score, trunc: {
            'opening_index': str(idx),
            'candidate_score': score,
            'truncated': str(trunc),
        }
        records = [row(0, '1.0', False), row(0, '0.0', False),
                   row(1, '0.5', False), row(1, '', True)]
        info = summary(records, 42)
        self.assertEqual(info['completed_games'], 3)
        self.assertEqual(info['truncated'], 1)
        self.assertAlmostEqual(info['mean_score_completed_only'], 0.5)
        self.assertIsNone(info['paired_opening_bootstrap_95_percent_ci'])

    def test_paired_bootstrap_bounded(self):
        rows = []
        for i in range(8):
            rows.extend([{'opening_index': str(i), 'candidate_score': '1.0'},
                         {'opening_index': str(i), 'candidate_score': '0.5'}])
        ci = paired_bootstrap_ci(rows)
        self.assertEqual(ci, [0.75, 0.75])

    def test_real_batch_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train_0000.npz'
            np.savez(path, boards=np.zeros((12, 61, 8, 8), dtype=np.float16),
                     actions=np.arange(12, dtype=np.int32),
                     outcomes=np.zeros(12, dtype=np.float32))
            batches = load_real_batches(Path(tmp), batch_size=2,
                                        device=torch.device('cpu'), batch_count=4)
            self.assertEqual(len(batches), 4)
            self.assertEqual(tuple(batches[0][0].shape), (2, 61, 8, 8))
            self.assertEqual(batches[0][1].dtype, torch.int64)


if __name__ == '__main__':
    unittest.main()
