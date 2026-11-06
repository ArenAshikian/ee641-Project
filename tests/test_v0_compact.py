"""Verify compact dataset storage preserves the board encoding and labels."""

import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

from chess_rl.compact_dataset import pack_boards, unpack_boards, write_shard, load_shard
from chess_rl.encoding import encode_board, action_for_move


class TestV0Compact(unittest.TestCase):
    def test_round_trip_normal_and_halfmove(self):
        board = chess.Board()
        states = [encode_board(board)]
        for uci in ['g1f3', 'g8f6', 'f3g1', 'f6g8']:
            board.push_uci(uci)
            states.append(encode_board(board))
        x = np.stack(states)
        packed, halfmove = pack_boards(x)
        self.assertEqual(packed.shape, (5, 480))
        recovered = unpack_boards(packed, halfmove)
        self.assertEqual(recovered.shape, (5, 61, 8, 8))
        self.assertTrue(np.allclose(x, recovered.astype(np.float32), atol=0.0002))

    def test_shard_round_trip(self):
        board = chess.Board()
        move = next(iter(board.legal_moves))
        row = (encode_board(board), action_for_move(board, move), 1.0, 20, 0)
        with tempfile.TemporaryDirectory() as tmp:
            rows = [row]
            write_shard(Path(tmp), 'train', 0, rows)
            self.assertEqual(rows, [])
            x, a, v, thirds, counts = load_shard(Path(tmp)/'train_00000.npz')
            self.assertEqual(x.shape, (1, 61, 8, 8))
            self.assertEqual(a.tolist(), [row[1]])
            self.assertEqual(v.tolist(), [1.0])
            self.assertEqual(thirds.tolist(), [0])
            self.assertEqual(counts.tolist(), [20])
            self.assertTrue(np.allclose(x[0], row[0], atol=0.0002))

    def test_reject_nonbinary_planes(self):
        x = np.zeros((1, 61, 8, 8), dtype=np.float32)
        x[0, 0, 2, 2] = 0.7
        with self.assertRaises(ValueError):
            pack_boards(x)


if __name__ == '__main__':
    unittest.main()
