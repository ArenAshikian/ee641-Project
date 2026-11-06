"""No models, Lc0 process or internet needed for confirmation statistics tests."""
import unittest

from chess_rl import maia_confirm as c


def rows_for(model, scores, level='1900'):
    rows = []
    for i, (white, black) in enumerate(scores):
        rows.append({'model': model, 'maia': level, 'opening_id': i,
                     'color': 'white', 'score': white, 'truncated': False})
        rows.append({'model': model, 'maia': level, 'opening_id': i,
                     'color': 'black', 'score': black, 'truncated': False})
    return rows


class TestOctoberConfirmation(unittest.TestCase):
    def test_total_games(self):
        self.assertEqual(c.TEST_PAIRS * 2 * len(c.MODELS) * len(c.RATINGS), 3072)
        self.assertEqual(c.SMOKE_PAIRS * 2 * len(c.MODELS), 12)
        self.assertEqual(c.MODELS['v0'], c.V0)
        self.assertEqual(c.MODELS['latest_s49'].name, 's0049.pt')

    def test_manifest_detects_changes(self):
        doc = {'schema': 1, 'source_url': c.OCTOBER_URL, 'seed': c.SEED,
               'openings': [['e2e4', 'e7e5']]}
        doc['sha256'] = c.digest(doc)
        self.assertEqual(c.digest(doc), doc['sha256'])
        doc['seed'] += 1
        self.assertNotEqual(c.digest(doc), doc['sha256'])

    def test_primary_all_ties(self):
        same = [(0.5, 0.5)] * 12
        result = c.primary_test(rows_for('latest_s49', same) + rows_for('v0', same), draws=1000)
        self.assertEqual(result['matched_pairs'], 12)
        self.assertEqual(result['score_difference'], 0)
        self.assertEqual(result['p_two_sided'], 1.0)

    def test_primary_large_clear_improvement(self):
        result = c.primary_test(rows_for('latest_s49', [(1., 1.)] * 20) +
                                rows_for('v0', [(0., 0.)] * 20), draws=4000)
        self.assertEqual(result['matched_pairs'], 20)
        self.assertEqual(result['score_difference'], 1.0)
        self.assertEqual(result['ci95'], [1., 1.])
        self.assertLess(result['p_two_sided'], .01)

    def test_partial_and_truncated_pairs_excluded(self):
        a = rows_for('latest_s49', [(1., 1.)] * 8)
        b = rows_for('v0', [(0., 0.)] * 8)
        b[1]['truncated'] = True
        b[1]['score'] = None
        a.pop()
        result = c.primary_test(a + b, draws=200)
        self.assertEqual(result['matched_pairs'], 6)

    def test_smoke_not_overlapping_final(self):
        self.assertFalse(set(range(c.TEST_PAIRS)) & set(range(c.TEST_PAIRS, c.TOTAL_STARTS)))


if __name__ == '__main__':
    unittest.main()
