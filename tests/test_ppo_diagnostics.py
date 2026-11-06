"""Checks for plateau diagnostic scoring and PPO gradient direction."""
import unittest

import torch

from chess_rl.ppo_diagnostics import regression_summary, scored_game_summary
from chess_rl.ppo_math import masked_distribution, ppo_objective


class TestPPODiagnostics(unittest.TestCase):
    def test_value_fit_beats_constant_when_predictions_are_good(self):
        report = regression_summary([-.8, .8, -.5, .5], [-1, 1, -1, 1])
        self.assertLess(report['value_mse_vs_terminal_result'], report['best_constant_mse'])
        self.assertGreater(report['value_r2_vs_constant'], 0)
        self.assertGreater(report['value_outcome_correlation'], 0)

    def test_bad_value_head_scores_negative_r2(self):
        report = regression_summary([1, -1, 1, -1], [-1, 1, -1, 1])
        self.assertLess(report['value_r2_vs_constant'], 0)

    def test_truncations_do_not_count_as_draws(self):
        sample = [
            {'truncated': False, 'outcome_for_learner': 1},
            {'truncated': False, 'outcome_for_learner': 0},
            {'truncated': False, 'outcome_for_learner': -1},
            {'truncated': True, 'outcome_for_learner': None},
        ]
        result = scored_game_summary(sample)
        self.assertEqual((result['wins'], result['draws'], result['losses'], result['truncated']),
                         (1, 1, 1, 1))
        self.assertEqual(result['score_on_completed_games'], 0.5)

    def test_ppo_gradient_sign_on_positive_and_negative_advantages(self):
        for advantage, expected_sign in [(1.0, -1), (-1.0, 1)]:
            logits = torch.zeros((1, 4672), requires_grad=True)
            legal = torch.zeros_like(logits, dtype=torch.bool)
            legal[0, :2] = True
            old_lp = masked_distribution(logits.detach(), legal).log_prob(torch.tensor([0]))
            values = torch.zeros(1)
            loss, _ = ppo_objective(logits, values, legal,
                                    torch.tensor([0]), old_lp,
                                    values, torch.tensor([advantage]),
                                    torch.zeros(1), value_weight=0., entropy_weight=0.)
            loss.backward()
            self.assertEqual(int(torch.sign(logits.grad[0, 0]).item()), expected_sign)


if __name__ == '__main__':
    unittest.main()
