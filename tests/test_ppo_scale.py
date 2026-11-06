"""Small correctness tests for the new isolated PPO scaling experiments."""

import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from chess_rl.human_openings import training_openings, digest_openings
from chess_rl.ppo_scale import (BRANCHES, GAMES, STEPS, archive_file,
                                bootstrap_pair_ci, choose_rival, config,
                                paired_difference, summarize_eval)
from chess_rl.ppo_scale_metrics import detailed_ppo_update, explained_variance


class TestHumanPPOStarts(unittest.TestCase):
    def test_paired_games_use_same_start_and_opposite_color(self):
        info = {'train': [[str(i)] for i in range(100)], 'eval': [],
                'source': {}, 'seed': 1}
        a = training_openings(info, 40, seed=100)
        self.assertEqual(40, len(a))
        self.assertEqual(a[::2], a[1::2])
        self.assertEqual(len({tuple(x) for x in a[::2]}), 20)
        self.assertEqual(a, training_openings(info, 40, seed=100))
        self.assertNotEqual(a, training_openings(info, 40, seed=200))

    def test_fingerprint_changes_when_starts_change(self):
        info = {'train': [['e2e4']], 'eval': [['d2d4']], 'source': {}, 'seed': 3}
        a = digest_openings(info)
        info['eval'] = [['c2c4']]
        self.assertNotEqual(a, digest_openings(info))


class TestOpponentPool(unittest.TestCase):
    def test_latest_is_always_frozen_snapshot(self):
        for seed in range(10):
            label, model = choose_rival('latest', random.Random(seed), 'recent', {'v0':'old'})
            self.assertEqual((label, model), ('recent_frozen', 'recent'))

    def test_pool_includes_both_recent_and_historical(self):
        values = {choose_rival('pool', random.Random(seed), 'recent', {'v0':'old'})[0]
                  for seed in range(100)}
        self.assertEqual(values, {'recent_frozen', 'v0'})

    def test_branch_files_do_not_overlap(self):
        self.assertNotEqual(archive_file('pool', 10), archive_file('latest', 10))
        self.assertNotEqual(archive_file('pool', 10).parent, archive_file('latest', 10).parent)
        self.assertEqual(GAMES * STEPS, 25088)

    def test_settings_lock_data_and_seed(self):
        base = {'v100': 'abc', 'v0': 'def'}
        one = config('pool', base, 'hash', 1, 512, 240, 2, 64, 1e-5)
        two = config('latest', base, 'hash', 1, 512, 240, 2, 64, 1e-5)
        other = config('pool', base, 'hash2', 1, 512, 240, 2, 64, 1e-5)
        self.assertNotEqual(one, two)
        self.assertNotEqual(one, other)
        self.assertTrue(one['no_search'])


class TestPairScoring(unittest.TestCase):
    @staticmethod
    def row(score, truncated=False):
        return {'score': '' if truncated else str(float(score)),
                'truncated': '1' if truncated else '0'}

    def test_paired_ci_is_not_400_independent_games(self):
        self.assertEqual(bootstrap_pair_ci([.5]*30), [.5,.5])
        self.assertIsNone(bootstrap_pair_ci([0,1,0]))

    def test_compares_same_position_and_color_only(self):
        first, second = {}, {}
        for opening in range(20):
            for color in ('white','black'):
                key = ('v0', opening, color)
                first[key] = self.row(1)
                second[key] = self.row(0)
        difference = paired_difference(first, second, 'v0', 20)
        self.assertEqual(difference['delta'], 1.0)
        self.assertEqual(difference['paired_95pct_ci'], [1.,1.])
        first[('v0', 0, 'black')] = self.row(1, truncated=True)
        difference = paired_difference(first, second, 'v0', 20)
        self.assertEqual(difference['pairs'], 19)

    def test_truncation_is_not_counted_as_draw(self):
        records = {('v0', 0,'white'): {**self.row(1), 'termination':'CHECKMATE'},
                   ('v0', 0,'black'): {**self.row(0, True), 'termination':'MAX_PLIES'},
                   ('v0', 1,'white'): {**self.row(.5), 'termination':'STALEMATE'}}
        totals = summarize_eval(records, [('v0', Path('mock'))], 10)['v0']
        self.assertEqual(totals['wins'], 1)
        self.assertEqual(totals['draws'], 1)
        self.assertEqual(totals['losses'], 0)
        self.assertEqual(totals['truncated'], 1)
        self.assertEqual(totals['paired_pairs'], 0)


class TestMetrics(unittest.TestCase):
    def test_explained_variance_basics(self):
        self.assertAlmostEqual(explained_variance([0,1,-1], [0,1,-1]), 1.0)
        self.assertIsNone(explained_variance([0,0], [0,0]))

    def test_optimizer_collects_all_minibatches(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.policy = torch.nn.Parameter(torch.zeros(3))
                self.value = torch.nn.Parameter(torch.tensor(0.0))
            def forward(self, x):
                return (self.policy.expand(len(x), -1),
                        self.value.tanh().expand(len(x)))
        model = Tiny()
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        examples = []
        for i in range(8):
            examples.append({
                'board': np.zeros((61, 8, 8), dtype=np.float16),
                'mask': np.array([True,True,False], dtype=bool),
                'action': i % 2,
                'old_log_prob': -math.log(2), 'value': 0.0,
                'advantage': 1.0 if i % 2 == 0 else -1.0,
                'return': 1.0 if i % 2 == 0 else -1.0,
            })
        info = detailed_ppo_update(model, opt, examples, 'cpu',
                                   epochs=2, batch_size=4, seed=17)
        self.assertEqual(info['optimizer_steps'], 4)
        self.assertEqual(info['transitions'], 8)
        self.assertGreaterEqual(info['approx_kl_positive'], 0.0)
        self.assertGreater(model.policy[0].item(), model.policy[1].item())


if __name__ == '__main__':
    unittest.main()
