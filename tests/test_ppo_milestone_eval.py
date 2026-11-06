"""No external model files or running chess engine required."""
import unittest
from unittest.mock import patch

from chess_rl import ppo_milestone_eval as m


class MilestoneEvalTests(unittest.TestCase):
    def test_milestone_paths_and_counts(self):
        self.assertEqual([p.name for p, _, _ in m.NEW.values()],
                         ['s0026.pt', 's0075.pt', 's0124.pt'])
        self.assertEqual([steps for _, steps, _ in m.NEW.values()], [26, 75, 124])
        self.assertEqual([games for _, _, games in m.NEW.values()],
                         [50240, 75328, 100416])
        self.assertEqual(m.COUNTS['latest_s49'], 36928)
        self.assertNotEqual(m.paths('eval'), m.paths('smoke'))

    def test_synthetic_comparison_matched_by_pair_and_color(self):
        # Ten paired openings, five frozen models; the 100k policy wins more.
        records = {}
        for label in m.ORDER:
            for i in range(10):
                for color in ('white', 'black'):
                    score = 1.0 if label == 'ppo_100k' else 0.5
                    key = (label, '1900', i, color)
                    records[key] = {'model': label, 'maia': '1900',
                                    'opening_id': i, 'color': color,
                                    'score': score, 'truncated': False}
        with patch.object(m.old, 'TEST_PAIRS', 10):
            references = {k: v for k, v in records.items() if k[0] in m.REFERENCES}
            fresh = {k: v for k, v in records.items() if k[0] in m.NEW}
            summary = m.summarize(references, fresh, 'eval')
        self.assertTrue(summary['completed'])
        comparison = summary['comparisons']['ppo_100k_minus_latest_s49']
        self.assertEqual(comparison['matched_opening_pairs'], 10)
        self.assertEqual(comparison['score_delta'], 0.5)
        self.assertEqual(summary['models']['ppo_100k']['complete_pairs'], 10)

    def test_truncation_excluded_from_matched_scores(self):
        records = {}
        for label in m.ORDER:
            for i in range(10):
                for color in ('white', 'black'):
                    truncated = label == 'ppo_100k' and i == 0 and color == 'white'
                    records[(label, '1900', i, color)] = {
                        'model': label, 'maia': '1900', 'opening_id': i,
                        'color': color, 'truncated': truncated,
                        'score': None if truncated else 0.5,
                    }
        with patch.object(m.old, 'TEST_PAIRS', 10):
            references = {k: v for k, v in records.items() if k[0] in m.REFERENCES}
            fresh = {k: v for k, v in records.items() if k[0] in m.NEW}
            summary = m.summarize(references, fresh, 'eval')
        self.assertEqual(summary['comparisons']['ppo_100k_minus_latest_s49']['matched_opening_pairs'], 9)
        self.assertEqual(summary['models']['ppo_100k']['truncated'], 1)


if __name__ == '__main__':
    unittest.main()
