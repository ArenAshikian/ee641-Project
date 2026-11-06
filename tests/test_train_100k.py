import io
import tempfile
import unittest
from pathlib import Path

import chess.pgn
import numpy as np

from chess_rl.lichess_data import positions_from_game
from chess_rl.train_100k import calculate_baselines, flush_shard


class Test100kPipeline(unittest.TestCase):
    def test_legal_move_counts_in_position_samples(self):
        pgn = '''[Event "Rated Blitz game"]
[Site "example"]
[Result "1-0"]

1. e4 e5 1-0
'''
        game = chess.pgn.read_game(io.StringIO(pgn))
        samples = list(positions_from_game(game, 8, 7, include_legal_count=True))
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0][3], 20)  # White has 20 moves in the initial position.
        self.assertEqual(samples[1][3], 20)  # Black has 20 replies after 1. e4.

    def test_write_shards_and_baselines(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            board = np.zeros((61, 8, 8), dtype=np.float32)
            train_rows = [(board, 10, 1.0, 20), (board, 10, -1.0, 30)]
            val_rows = [(board, 10, 0.0, 20), (board, 11, 1.0, 30)]
            self.assertEqual(flush_shard(folder, 'train', train_rows, 0), 1)
            self.assertEqual(flush_shard(folder, 'val', val_rows, 0), 1)
            self.assertFalse(train_rows)
            with np.load(folder / 'train_0000.npz') as data:
                self.assertEqual(data['boards'].shape, (2, 61, 8, 8))
            baselines = calculate_baselines([folder / 'train_0000.npz'],
                                            [folder / 'val_0000.npz'])
            self.assertAlmostEqual(baselines['uniform_random_legal_accuracy'],
                                   (1 / 20 + 1 / 30) / 2)
            self.assertAlmostEqual(baselines['majority_action_accuracy'], 0.5)


if __name__ == '__main__':
    unittest.main()
