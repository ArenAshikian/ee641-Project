"""Small regression tests for search-free sampling-temperature evaluation."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import numpy as np
import torch

from chess_rl import ppo_temperature_test as trial


class DummyModel:
    def eval(self):
        return self


class TestTemperatureEvaluation(unittest.TestCase):
    def test_entropy_and_calibration(self):
        logits = [np.array([2.0, 0.0, -1.0]),
                  np.array([1.5, .3, .0, -2.0]),
                  np.array([.4, .3, .2, .1])]
        self.assertGreater(trial.mean_entropy(logits, 1.0),
                           trial.mean_entropy(logits, .5))
        target = trial.mean_entropy(logits, .63)
        matched, status = trial.solve_entropy_temperature(logits, target)
        self.assertEqual(status, 'matched')
        self.assertAlmostEqual(matched, .63, places=4)
        with self.assertRaises(ValueError):
            trial.entropy(logits[0], 0.0)

    def test_condition_list_includes_matching_temperature(self):
        conditions = trial.condition_specs(.62345)
        self.assertEqual(len(conditions), 7)
        self.assertEqual(conditions[-1]['id'], 'v0_entropy_matched')
        self.assertAlmostEqual(conditions[-1]['temperature'], .62345)
        self.assertEqual(sum(c['candidate'] == 'v100' for c in conditions), 3)

    def test_only_legal_actions_are_sampled_one_forward(self):
        class Net:
            def __init__(self):
                self.calls = 0
            def __call__(self, x):
                self.calls += 1
                logits = torch.full((1, 4672), -20.0)
                logits[0, 0] = 1000.0  # illegal, must never be chosen
                logits[0, 7] = 2.0
                logits[0, 11] = 1.0
                return logits, torch.zeros(1)

        model = Net()
        fake_board = object()
        with patch.object(trial, 'legal_actions', return_value={7: 'e2e4', 11: 'd2d4'}), \
             patch.object(trial, 'encode_board', return_value=np.zeros((61, 8, 8), np.float32)):
            choice = trial.temperature_move(model, fake_board, 'cpu', .7)
        self.assertIn(choice, ('e2e4', 'd2d4'))
        self.assertEqual(model.calls, 1)

    def test_truncation_is_not_draw(self):
        with patch.object(trial, 'board_from_opening', return_value=chess.Board()), \
             patch.object(trial, 'temperature_move', return_value=chess.Move.from_uci('e2e4')) as pick:
            result = trial.play_game(DummyModel(), DummyModel(), (), chess.WHITE,
                                     'cpu', max_plies=1, seed=77, temp=.7)
        self.assertTrue(result['truncated'])
        self.assertIsNone(result['score'])
        self.assertEqual(result['termination'], 'MAX_PLIES')
        pick.assert_called_once()

    def test_pairs_drop_truncated_games(self):
        rows = {}
        for i in range(6):
            for color in ('white', 'black'):
                rows[('v100_t1', i, color)] = {'score': '1.0', 'truncated': '0'}
                rows[('v0_t1', i, color)] = {'score': '0.0', 'truncated': '0'}
        rows[('v100_t1', 0, 'white')] = {'score': '', 'truncated': '1'}
        self.assertEqual(len(trial.pair_scores(rows, 'v100_t1', 6)), 5)
        comparison = trial.compare(rows, 6, 42)['v100_t1_minus_v0_t1']
        self.assertEqual(comparison['opening_pairs'], 5)
        self.assertAlmostEqual(comparison['score_difference'], 1.0)

    def test_resume_uses_same_openings_without_rerunning(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            conditions = trial.condition_specs(.64321)
            openings = [{'uci': ['e2e4', 'e7e5']} for _ in range(5)]
            manifest = {
                'seed': 41, 'pairs': 5, 'conditions': conditions,
                'openings': openings, 'calibration': {'matched_temperature': .64321},
            }
            called = []
            def fake_play(model, opponent, opening, candidate_color, device,
                          max_plies, seed, temp):
                called.append((tuple(opening), candidate_color, seed, temp))
                return {'score': .5, 'winner': 'draw', 'termination': 'STALEMATE',
                        'truncated': False, 'plies': 20}

            with patch.object(trial, 'make_manifest', return_value=manifest), \
                 patch.object(trial, 'read_historical', return_value=DummyModel()), \
                 patch.object(trial, 'play_game', side_effect=fake_play), \
                 patch.object(trial, 'RESULTS', folder / 'games.csv'), \
                 patch.object(trial, 'SUMMARY_JSON', folder / 'summary.json'), \
                 patch.object(trial, 'SUMMARY_CSV', folder / 'summary.csv'), \
                 patch.object(trial, 'CALIBRATION_JSON', folder / 'calibration.json'):
                trial.run(pairs=5, seed=41)
                self.assertEqual(len(called), 70)
                self.assertEqual(len({(row[0], row[1], row[2]) for row in called}), 10)
                trial.run(pairs=5, seed=41)
                self.assertEqual(len(called), 70)
                with (folder / 'games.csv').open(newline='') as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 70)


if __name__ == '__main__':
    unittest.main()
