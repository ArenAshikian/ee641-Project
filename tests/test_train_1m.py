"""Lightweight tests for the one-million-position sampling pipeline."""

import io
import tempfile
import unittest
from pathlib import Path

import chess.pgn
import numpy as np

from chess_rl.train_1m import (
    PHASE_NAMES, examples_from_game, make_loader, measure_baselines,
    pick_moves_by_game_third, save_shard,
)


FULL_PGN = '''[Event "Rated Blitz game"]
[Site "https://lichess.org/phase_test"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "2000"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7
6. Re1 b5 7. Bb3 O-O 8. c3 d6 9. h3 Nb8 10. d4 Nbd7 1-0
'''


class TestTrain1M(unittest.TestCase):
    def test_sampling_covers_all_thirds(self):
        game = chess.pgn.read_game(io.StringIO(FULL_PGN))
        selected = pick_moves_by_game_third(game, limit=12, seed=7)
        self.assertEqual(len(selected), 12)
        self.assertEqual({phase for _, phase in selected}, {0, 1, 2})
        self.assertEqual([sum(p == i for _, p in selected) for i in range(3)], [4, 4, 4])
        self.assertEqual(selected, pick_moves_by_game_third(game, limit=12, seed=7))
        self.assertEqual(list(PHASE_NAMES), ['early', 'middle', 'late'])

    def test_full_game_examples_keep_correct_label_perspective(self):
        game = chess.pgn.read_game(io.StringIO(FULL_PGN))
        examples = list(examples_from_game(game, limit=12, seed=2))
        self.assertEqual(len(examples), 12)
        for board, action, outcome, legal_count, third in examples:
            self.assertEqual(board.shape, (61, 8, 8))
            self.assertTrue(0 <= action < 4672)
            self.assertGreater(legal_count, 0)
            self.assertIn(outcome, (-1.0, 1.0))
            self.assertIn(third, (0, 1, 2))

    def test_low_clock_moves_are_excluded(self):
        pgn = '''[Event "Rated Blitz game"]
[Site "lowclock"]
[Result "1-0"]

1. e4 { [%clk 0:00:20] } e5 { [%clk 0:03:00] }
2. Nf3 { [%clk 0:02:58] } Nc6 1-0
'''
        game = chess.pgn.read_game(io.StringIO(pgn))
        picked = pick_moves_by_game_third(game, limit=12)
        self.assertEqual(len(picked), 3)
        self.assertNotIn(0, [i for i, _ in picked])

    def test_shards_include_thirds_and_legal_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            board = np.zeros((61, 8, 8), dtype=np.float32)
            train = [(board, 10, 1., 20, 0), (board, 10, -1., 30, 1)]
            val = [(board, 10, 0., 20, 0), (board, 11, 1., 30, 2)]
            self.assertEqual(save_shard(folder, 'train', 0, train), 1)
            self.assertEqual(save_shard(folder, 'val', 0, val), 1)
            self.assertEqual(len(train), 0)
            with np.load(folder / 'val_0000.npz') as data:
                self.assertEqual(tuple(data['boards'].shape), (2, 61, 8, 8))
                self.assertEqual(data['thirds'].tolist(), [0, 2])
            loader = make_loader(folder / 'train_0000.npz', 2, shuffle=False)
            boards, actions, outcomes, phases = next(iter(loader))
            self.assertEqual(tuple(boards.shape), (2, 61, 8, 8))
            self.assertEqual(phases.tolist(), [0, 1])
            baselines = measure_baselines([folder / 'train_0000.npz'],
                                          [folder / 'val_0000.npz'])
            self.assertAlmostEqual(baselines['random_legal_accuracy'],
                                   (1/20 + 1/30)/2)
            self.assertAlmostEqual(baselines['most_common_action_accuracy'], 0.5)


if __name__ == '__main__':
    unittest.main()
