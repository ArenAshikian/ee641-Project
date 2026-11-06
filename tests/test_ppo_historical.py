"""Small unit tests for independent batched mixed-opponent PPO."""
import unittest

import chess
import torch

from chess_rl.network import ChessNetwork
from chess_rl.ppo_batch import rollout_latest_batched
from chess_rl.ppo_batch_mixed import rollout_mixed_batched
from chess_rl import ppo_historical_continue as h


class TestHistoricalOpponentScheduling(unittest.TestCase):
    def test_game_targets_and_separate_paths(self):
        self.assertEqual(h.total_games(0), 11840)
        self.assertEqual(h.total_games(1149), 600128)
        self.assertEqual(h.total_games(1930), 1000000)
        self.assertNotEqual(h.PROGRESS, h.PARENT)
        self.assertTrue(h.should_archive(10))
        self.assertTrue(h.should_archive(49))
        self.assertFalse(h.should_archive(11))

    def test_pairing_and_exact_opponent_mix(self):
        labels = h.choose_opponents(512, ['r1','r2','r3'], ['o1','o2'], 101)
        self.assertEqual(len(labels), 512)
        for i in range(0, 512, 2):
            self.assertEqual(labels[i], labels[i + 1])
        self.assertEqual(labels.count('current'), 128)
        self.assertEqual(sum(x in ['r1','r2','r3'] for x in labels), 256)
        self.assertEqual(sum(x in ['o1','o2'] for x in labels), 128)
        self.assertEqual(labels, h.choose_opponents(512, ['r1','r2','r3'], ['o1','o2'], 101))
        self.assertNotEqual(labels, h.choose_opponents(512, ['r1','r2','r3'], ['o1','o2'], 102))

    def test_history_never_contains_future(self):
        names = h.all_ancestors(21)
        self.assertIn('hist_s0010', names)
        self.assertIn('original_v100', names)
        self.assertNotIn('original_s49', names)
        self.assertNotIn('original_50k', names)
        self.assertIn('hist_s0020', names)
        self.assertNotIn('hist_s0030', names)
        recent, older, selected = h.opponent_pools(21)
        self.assertEqual(len(set(recent + older)), len(selected))
        self.assertTrue(set(recent).isdisjoint(older))
        self.assertTrue(set(selected).issubset(names))

    def test_matching_current_only_opening_schedule(self):
        self.assertEqual(h.schedule_seed(1), (641528, 1))
        self.assertEqual(h.schedule_seed(49), (641528, 49))
        self.assertEqual(h.schedule_seed(50), (6412028, 1))
        self.assertEqual(h.schedule_seed(173), (6412028, 124))
        self.assertEqual(h.total_games(49), 36928)
        self.assertEqual(h.total_games(173), 100416)
        self.assertEqual(h.total_games(367), 199744)
        self.assertEqual(h.total_games(563), 300096)

    def test_no_future_opponents_and_new_branch_namespace(self):
        self.assertEqual(len(h.EARLIER), 11)
        self.assertNotIn('ppo_scale_latest', str(h.EARLIER))
        self.assertNotIn('ppo_batch_continue', str(h.EARLIER))
        self.assertIn('ppo_historical_v100', str(h.BRANCH))
        first_recent, first_old, first_selected = h.opponent_pools(1)
        self.assertEqual(first_recent, ['original_v100', 'original_v90',
                                        'original_v80', 'original_v70', 'original_v60'])
        self.assertEqual(len(first_old), 5)
        self.assertTrue(all(name.startswith('original_') for name in first_selected))

    def test_incorrect_games_rejected(self):
        with self.assertRaises(ValueError):
            h.choose_opponents(10, ['r'], ['o'], 1)
        with self.assertRaises(ValueError):
            h.choose_opponents(8, [], ['o'], 1)


class TestMixedBatchRollout(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(641)
        self.device = torch.device('cpu')
        self.model = ChessNetwork(channels=8, blocks=0).eval()

    def test_all_current_matches_existing_batch_rollout(self):
        openings = [['e2e4', 'e7e5'], ['d2d4', 'd7d5']]
        colors = [chess.WHITE, chess.BLACK]
        torch.manual_seed(900)
        old = rollout_latest_batched(self.model, openings, colors, self.device, max_plies=8)
        torch.manual_seed(900)
        new = rollout_mixed_batched(self.model, {}, openings, colors,
                                   ['current'] * 2, self.device, max_plies=8)
        for (old_samples, old_result), (new_samples, new_result) in zip(old, new):
            self.assertEqual(old_result, new_result)
            self.assertEqual(len(old_samples), len(new_samples))
            for a, b in zip(old_samples, new_samples):
                self.assertEqual(a['action'], b['action'])
                self.assertAlmostEqual(a['value'], b['value'], places=4)
                self.assertAlmostEqual(a['advantage'], b['advantage'], places=4)

    def test_mixed_opponents_and_legal_move_records(self):
        torch.manual_seed(640)
        rival = ChessNetwork(channels=8, blocks=0).eval()
        res = rollout_mixed_batched(
            self.model, {'rival': rival},
            [['e2e4', 'e7e5'], ['e2e4', 'e7e5']],
            [chess.WHITE, chess.BLACK], ['rival', 'current'],
            self.device, max_plies=8)
        self.assertEqual(len(res), 2)
        for samples, result in res:
            self.assertTrue(result['terminated'] or result['truncated'])
            for item in samples:
                self.assertTrue(item['mask'][item['action']])

    def test_invalid_opponent_rejected(self):
        with self.assertRaises(ValueError):
            rollout_mixed_batched(self.model, {}, [[]], [chess.WHITE],
                                  ['nonexistent'], self.device)


if __name__ == '__main__':
    unittest.main()
