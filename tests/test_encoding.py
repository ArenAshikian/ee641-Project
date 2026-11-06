import unittest

import chess

from chess_rl.encoding import ACTION_COUNT, encode_board, action_for_move, legal_actions, legal_mask, move_for_action


class TestEncoding(unittest.TestCase):
    def test_board_shape(self):
        board = chess.Board()
        x = encode_board(board)
        self.assertEqual(x.shape, (61, 8, 8))
        self.assertEqual(x.dtype.name, "float32")
        self.assertEqual(x[0, 6, 4], 1.0)  # White pawn at e2, viewed as us.

    def test_black_orientation(self):
        board = chess.Board()
        board.push_uci("e2e4")
        x = encode_board(board)
        self.assertEqual(x[0, 6, 3], 1.0)  # Black pawn at e7, rotated.

    def test_starting_legal_actions(self):
        board = chess.Board()
        self.assertEqual(len(legal_actions(board)), 20)
        self.assertEqual(int(legal_mask(board).sum()), 20)
        self.assertEqual(len(legal_mask(board)), ACTION_COUNT)

    def test_round_trip_for_special_moves(self):
        boards = [
            chess.Board(),
            chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
            chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1"),
            chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
        ]
        black = chess.Board()
        black.push_uci("e2e4")
        boards.append(black)

        for board in boards:
            moves = list(board.legal_moves)
            encoded = [action_for_move(board, move) for move in moves]
            self.assertEqual(len(set(encoded)), len(moves))
            for move, action in zip(moves, encoded):
                self.assertGreaterEqual(action, 0)
                self.assertLess(action, ACTION_COUNT)
                self.assertEqual(move_for_action(board, action), move)

    def test_promotion_options_all_distinct(self):
        board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
        choices = [chess.Move.from_uci("a7a8" + p) for p in "qrbn"]
        self.assertTrue(all(m in board.legal_moves for m in choices))
        self.assertEqual(len({action_for_move(board, m) for m in choices}), 4)


if __name__ == "__main__":
    unittest.main()
