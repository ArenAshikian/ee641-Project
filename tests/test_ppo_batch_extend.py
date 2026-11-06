"""Safe long-run continuation configuration checks (no GPU training)."""
import unittest
from chess_rl import ppo_batch_continue as c

class TestLongContinuation(unittest.TestCase):
    def test_target_count_and_milestones(self):
        self.assertEqual(c.total_games(124), 100416)
        self.assertEqual(c.total_games(1100), 600128)
        self.assertEqual(c.total_games(1881), 1000000)
        for step, total in c.MILESTONES.items():
            self.assertEqual(c.total_games(step), total)

    def test_hyperparameters_and_paths_unchanged(self):
        self.assertEqual((c.GAMES, c.MAX_PLIES, c.ROLLOUT_BATCH), (512, 240, 32))
        self.assertEqual((c.PPO_EPOCHS, c.PPO_BATCH, c.LR), (2, 64, 1e-5))
        self.assertEqual(c.archive_path(1100).name, 's1100.pt')
        self.assertEqual(c.PROGRESS.name, 'latest.pt')
        self.assertNotEqual(c.SOURCE, c.PROGRESS)

if __name__ == '__main__':
    unittest.main()
