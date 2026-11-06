import tempfile
import unittest
from pathlib import Path

import chess
import torch

from chess_rl.checkpoints import is_archive_version, save_checkpoint
from chess_rl.network import ChessNetwork


class TestNetwork(unittest.TestCase):
    def test_output_shapes_and_legal_move(self):
        net = ChessNetwork(channels=16, blocks=1).eval()
        board = chess.Board()
        x = torch.zeros(1, 61, 8, 8)
        with torch.no_grad():
            logits, values = net(x)
        self.assertEqual(tuple(logits.shape), (1, 4672))
        self.assertEqual(tuple(values.shape), (1,))
        self.assertIn(net.choose_move(board), board.legal_moves)

    def test_checkpoint_versions(self):
        self.assertTrue(is_archive_version(0))
        self.assertFalse(is_archive_version(9))
        self.assertTrue(is_archive_version(10))
        self.assertTrue(is_archive_version(20))

        net = ChessNetwork(channels=8, blocks=1)
        with tempfile.TemporaryDirectory() as temp:
            save_checkpoint(net, 0, folder=temp)
            self.assertTrue((Path(temp) / "v0000.pt").exists())
            save_checkpoint(net, 1, folder=temp)
            self.assertTrue((Path(temp) / "latest.pt").exists())
            save_checkpoint(net, 10, folder=temp)
            self.assertTrue((Path(temp) / "v0010.pt").exists())
            with self.assertRaises(FileExistsError):
                save_checkpoint(net, 10, folder=temp)


if __name__ == "__main__":
    unittest.main()
