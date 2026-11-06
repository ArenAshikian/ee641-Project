import unittest

import chess

from chess_rl.environment import ChessEnvironment


class TestEnvironment(unittest.TestCase):
    def test_checkmate(self):
        env = ChessEnvironment()
        env.reset("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        info = env.status()
        self.assertEqual(info.termination, "CHECKMATE")
        self.assertEqual(info.result, "1-0")
        self.assertEqual(env.terminal_reward(chess.WHITE), 1.0)
        self.assertEqual(env.terminal_reward(chess.BLACK), -1.0)

    def test_stalemate(self):
        env = ChessEnvironment()
        env.reset("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        info = env.status()
        self.assertTrue(info.terminated)
        self.assertEqual(info.termination, "STALEMATE")
        self.assertEqual(env.terminal_reward(chess.WHITE), 0.0)

    def test_insufficient_material(self):
        env = ChessEnvironment()
        env.reset("8/8/8/8/8/8/7k/K7 w - - 0 1")
        self.assertEqual(env.status().termination, "INSUFFICIENT_MATERIAL")

    def test_fifty_move_claim(self):
        env = ChessEnvironment(claim_draw=True)
        env.reset("7k/8/8/8/8/8/R7/K7 w - - 100 1")
        self.assertTrue(env.status().terminated)
        self.assertEqual(env.status().result, "1/2-1/2")

    def test_threefold_claim(self):
        env = ChessEnvironment(claim_draw=True)
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
            if env.status().terminated:
                break
            env.step(chess.Move.from_uci(uci))
        self.assertTrue(env.status().terminated)
        self.assertEqual(env.status().result, "1/2-1/2")

    def test_truncation_is_not_draw(self):
        env = ChessEnvironment(max_plies=1)
        status = env.step(chess.Move.from_uci("e2e4"))
        self.assertTrue(status.truncated)
        self.assertFalse(status.terminated)
        self.assertEqual(status.result, "*")
        with self.assertRaises(ValueError):
            env.terminal_reward(chess.WHITE)

    def test_illegal_move_rejected(self):
        env = ChessEnvironment()
        with self.assertRaises(ValueError):
            env.step(chess.Move.from_uci("e2e5"))

    def test_real_game_move(self):
        env = ChessEnvironment()
        info = env.step(chess.Move.from_uci("e2e4"))
        self.assertFalse(info.terminated)
        self.assertFalse(info.truncated)
        self.assertEqual(env.plies, 1)


if __name__ == "__main__":
    unittest.main()
