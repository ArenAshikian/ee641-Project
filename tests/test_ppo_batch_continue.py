"""Quick tests for the isolated batched PPO continuation configuration."""
import unittest

from chess_rl import ppo_batch_continue as c


class TestBatchContinuation(unittest.TestCase):
    def test_game_accounting(self):
        self.assertEqual(c.START_GAMES, 36928)
        self.assertEqual(c.total_games(26), 50240)
        self.assertEqual(c.total_games(75), 75328)
        self.assertEqual(c.total_games(124), 100416)

    def test_old_training_hyperparameters_unchanged(self):
        self.assertEqual(c.GAMES, 512)
        self.assertEqual(c.MAX_PLIES, 240)
        self.assertEqual(c.PPO_EPOCHS, 2)
        self.assertEqual(c.PPO_BATCH, 64)
        self.assertEqual(c.LR, 1e-5)
        self.assertEqual(c.ROLLOUT_BATCH, 32)

    def test_batch_rollout_exclusively_latest_opponent(self):
        self.assertEqual(c.MILESTONES[26], 50240)
        self.assertEqual(c.MILESTONES[75], 75328)
        self.assertEqual(c.MILESTONES[124], 100416)
        self.assertIn('ppo_batch_continue', str(c.PROGRESS))
        self.assertNotEqual(c.PROGRESS, c.SOURCE)


if __name__ == '__main__':
    unittest.main()
