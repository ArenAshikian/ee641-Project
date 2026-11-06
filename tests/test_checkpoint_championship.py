"""Small tests for the frozen, no-search checkpoint championship."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess

from chess_rl import checkpoint_championship as ch


class TestCheckpointChampionship(unittest.TestCase):
    def test_five_matchups_and_paths(self):
        self.assertEqual(ch.MATCHUPS, [(100, 0), (100, 10), (100, 30),
                                      (100, 50), (30, 50)])
        self.assertEqual(ch.checkpoint(100).name, 'ppo_v0100.pt')
        self.assertEqual(ch.checkpoint(30).name, 'ppo_v0030.pt')
        self.assertEqual(ch.checkpoint(0).name, 'v0_supervised.pt')

    def test_new_positions_are_legal_unique_and_reproducible(self):
        a = ch.fresh_openings(6, 932, set())
        b = ch.fresh_openings(6, 932, set())
        self.assertEqual(a, b)
        self.assertEqual(len({entry['fen'] for entry in a}), 6)
        self.assertTrue(all(len(entry['uci']) in (10, 12, 14) for entry in a))
        for entry in a:
            board = ch.board_from_opening(entry['uci'])
            self.assertEqual(board.fen(), entry['fen'])
            self.assertFalse(board.is_game_over(claim_draw=True))
        old = ch.position_signature(ch.board_from_opening(a[0]['uci']))
        c = ch.fresh_openings(6, 932, {old})
        self.assertNotIn(old, [ch.position_signature(ch.board_from_opening(entry['uci'])) for entry in c])

    def test_score_ci_and_truncation(self):
        self.assertEqual(ch.paired_ci([1] * 30, 1), [1.0, 1.0])
        self.assertEqual(ch.paired_ci([0.5] * 30, 1), [0.5, 0.5])
        self.assertIsNone(ch.paired_ci([1] * 4, 1))
        label = ch.match_name(100, 0)
        records = {
            (label, 0, 'white'): {'score': '1.0', 'termination': 'CHECKMATE', 'truncated': '0'},
            (label, 0, 'black'): {'score': '0.5', 'termination': 'STALEMATE', 'truncated': '0'},
            (label, 1, 'white'): {'score': '', 'termination': 'MAX_PLIES', 'truncated': '1'},
            (label, 1, 'black'): {'score': '0.0', 'termination': 'CHECKMATE', 'truncated': '0'},
        }
        res = ch.summarize_match(100, 0, records, pairs=2, seed=4)
        self.assertEqual([res[k] for k in ('wins','draws','losses','truncated')], [1,1,1,1])
        self.assertEqual(res['completed_game_score'], 0.5)
        self.assertEqual(res['fully_completed_pairs'], 1)
        self.assertEqual(res['paired_score'], 0.75)
        self.assertIsNone(res['paired_bootstrap_95pct'])

    def test_resume_does_not_repeat_games(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            openings = ch.fresh_openings(5, 319, set())
            manifest = {'pairs': 5, 'seed': 319, 'max_plies': 40,
                        'openings': openings, 'matchups': [list(m) for m in ch.MATCHUPS]}
            count = [0]

            def fake_match(a, b, opening, color, device, limit, seed):
                count[0] += 1
                return {'score': 1.0 if color == chess.WHITE else 0.0,
                        'truncated': False, 'winner': 'white',
                        'termination': 'CHECKMATE', 'plies': 38}

            with patch.object(ch, 'make_manifest', return_value=manifest), \
                 patch.object(ch, 'read_historical', return_value=DummyModel()), \
                 patch.object(ch, 'play_eval_game', side_effect=fake_match), \
                 patch.object(ch, 'RESULTS', tmp / 'games.csv'), \
                 patch.object(ch, 'SUMMARY_CSV', tmp / 'summary.csv'), \
                 patch.object(ch, 'SUMMARY_JSON', tmp / 'summary.json'):
                ch.run_championship(5, 319, 40)
                self.assertEqual(count[0], 50)
                ch.run_championship(5, 319, 40)
                self.assertEqual(count[0], 50)
                rows = ch.read_games(manifest)
                self.assertEqual(len(rows), 50)
                self.assertEqual(ch.summarize_match(30, 50, rows, 5, 319)['paired_score'], 0.5)


class DummyModel:
    def eval(self):
        return self


if __name__ == '__main__':
    unittest.main()
