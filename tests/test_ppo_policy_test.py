"""Small tests for search-free, matched greedy/sampled PPO evaluation."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess

from chess_rl import ppo_policy_test as trial


class DummyModel:
    def eval(self):
        return self


class TestPolicyComparison(unittest.TestCase):
    def test_four_conditions_and_paths(self):
        self.assertEqual(trial.CONDITIONS, [
            ('v0', 'greedy'), ('v100', 'greedy'),
            ('v0', 'sampled'), ('v100', 'sampled')])
        self.assertEqual(trial.V0.name, 'v0_supervised.pt')
        self.assertEqual(trial.V100.name, 'ppo_v0100.pt')

    def test_pair_scores_drop_truncation(self):
        name = trial.label('v100', 'sampled')
        rows = {
            (name, 0, 'white'): {'score': '1.0', 'truncated': '0'},
            (name, 0, 'black'): {'score': '0.0', 'truncated': '0'},
            (name, 1, 'white'): {'score': '', 'truncated': '1'},
            (name, 1, 'black'): {'score': '1.0', 'truncated': '0'},
        }
        self.assertEqual(trial.paired_opening_scores(rows, name, 2), {0: .5})
        result = trial.summarize_condition({
            key: {**value, 'termination': 'MAX_PLIES' if value['truncated'] == '1' else 'CHECKMATE'}
            for key, value in rows.items()
        }, 'v100', 'sampled', 2, 123)
        self.assertEqual(result['truncated'], 1)
        self.assertEqual(result['game_score'], round((2 + 0) / 3, 4))
        self.assertEqual(result['paired_openings'], 1)
        self.assertIsNone(result['paired_bootstrap_95pct'])

    def test_bootstrap_and_gains(self):
        self.assertEqual(trial.bootstrap_ci([.5] * 10, 1), [.5, .5])
        self.assertIsNone(trial.bootstrap_ci([.5] * 4, 1))
        games = {}
        expected = {
            'v0_greedy': .5,
            'v100_greedy': .5,
            'v0_sampled': .25,
            'v100_sampled': .75,
        }
        for name, score in expected.items():
            for oid in range(10):
                for color in ('white', 'black'):
                    games[(name, oid, color)] = {
                        'score': str(float(score)), 'truncated': '0'}
        gains = trial.matched_comparisons(games, 10, 2)
        self.assertEqual(gains['common_completed_opening_pairs'], 10)
        self.assertAlmostEqual(gains['comparisons']['greedy_v100_minus_v0']['mean_score_difference'], 0)
        self.assertAlmostEqual(gains['comparisons']['sampled_v100_minus_v0']['mean_score_difference'], .5)
        self.assertEqual(gains['comparisons']['sampled_gain_minus_greedy_gain']['paired_bootstrap_95pct'], [.5, .5])

    def test_resume_and_common_openings(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            fake_manifest = {
                'pairs': 5, 'seed': 31, 'max_plies': 20,
                'openings': [{'uci': ['e2e4', 'e7e5']} for _ in range(5)]}
            made = []

            def fake_game(candidate, opponent, opening, candidate_color, device,
                          max_plies, seed, candidate_mode):
                made.append((tuple(opening), candidate_color, candidate_mode, seed))
                return {
                    'score': 1.0 if candidate_color == chess.WHITE else 0.0,
                    'winner': 'white', 'truncated': False,
                    'termination': 'CHECKMATE', 'plies': 24,
                }

            with patch.object(trial, 'make_manifest', return_value=fake_manifest), \
                 patch.object(trial, 'read_historical', return_value=DummyModel()), \
                 patch.object(trial, 'game', side_effect=fake_game), \
                 patch.object(trial, 'RESULTS', folder / 'games.csv'), \
                 patch.object(trial, 'SUMMARY_JSON', folder / 'summary.json'), \
                 patch.object(trial, 'SUMMARY_CSV', folder / 'summary.csv'):
                trial.run(pairs=5, seed=31, max_plies=20)
                self.assertEqual(len(made), 40)
                # Every condition has 5 openings from both colors.
                self.assertEqual({p[1] for p in made}, {True, False})
                # Same game seed for a given opening/color across conditions.
                self.assertEqual(len({p[3] for p in made}), 10)
                trial.run(pairs=5, seed=31, max_plies=20)
                self.assertEqual(len(made), 40)  # Nothing replayed.
                with trial.RESULTS.open(newline='', encoding='utf-8') as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 40)
                self.assertEqual(len(trial.read_games(fake_manifest)), 40)
                gains = trial.matched_comparisons(trial.read_games(fake_manifest), 5, 31)
                self.assertEqual(gains['common_completed_opening_pairs'], 5)
                self.assertEqual(gains['comparisons']['sampled_gain_minus_greedy_gain']['mean_score_difference'], 0)

    def test_play_uses_correct_sampling_and_never_search(self):
        # The artificial one-ply cutoff isolates one move request.
        board = chess.Board()
        with patch.object(trial, 'board_from_opening', return_value=board), \
             patch.object(trial, 'one_move', return_value=(chess.Move.from_uci('e2e4'), None)) as mover:
            out = trial.game(DummyModel(), DummyModel(), (), chess.WHITE,
                             'cpu', max_plies=1, seed=19, candidate_mode='sampled')
        self.assertTrue(out['truncated'])
        self.assertEqual(mover.call_count, 1)
        self.assertTrue(mover.call_args_list[0].kwargs['sample'])

        board = chess.Board()
        with patch.object(trial, 'board_from_opening', return_value=board), \
             patch.object(trial, 'one_move', return_value=(chess.Move.from_uci('e2e4'), None)) as mover:
            out = trial.game(DummyModel(), DummyModel(), (), chess.WHITE,
                             'cpu', max_plies=1, seed=19, candidate_mode='greedy')
        self.assertTrue(out['truncated'])
        self.assertFalse(mover.call_args_list[0].kwargs['sample'])


if __name__ == '__main__':
    unittest.main()
