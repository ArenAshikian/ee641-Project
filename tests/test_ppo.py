"""Small RL correctness tests. Run with: run.cmd tests"""
import unittest

import chess
import numpy as np
import torch

from chess_rl.network import ChessNetwork
from chess_rl.ppo_math import compute_gae, masked_distribution, ppo_objective
from chess_rl.ppo_train import (board_from_opening, final_score, one_move,
                                opening_sequences, rollout_game, ppo_update)


class TestPPO(unittest.TestCase):
    def test_terminal_rewards(self):
        self.assertEqual(final_score(chess.WHITE, chess.WHITE), 1)
        self.assertEqual(final_score(chess.BLACK, chess.WHITE), -1)
        self.assertEqual(final_score(None, chess.WHITE), 0)
        self.assertEqual(final_score(chess.WHITE, chess.BLACK), -1)

    def test_gae_terminal_and_truncation(self):
        a, targets = compute_gae([0, 1], [0.2, 0.3], 0, gamma=1, gae_lambda=1)
        np.testing.assert_allclose(a, [0.8, 0.7], atol=1e-6)
        np.testing.assert_allclose(targets, [1, 1], atol=1e-6)
        a, targets = compute_gae([0, 0], [0.2, 0.3], 0.6, gamma=1, gae_lambda=1)
        np.testing.assert_allclose(targets, [0.6, 0.6], atol=1e-6)
        self.assertNotEqual(float(targets[0]), 0.0)  # Cutoff not scored as a draw.

    def test_legal_mask_zeroes_illegal_probability(self):
        logits = torch.randn(2, 4672)
        legal = torch.zeros_like(logits, dtype=torch.bool)
        legal[:, [8, 100, 222]] = True
        dist = masked_distribution(logits, legal)
        self.assertAlmostEqual(float(dist.probs[0, ~legal[0]].sum()), 0, places=6)
        self.assertAlmostEqual(float(dist.probs[0].sum()), 1, places=6)
        for _ in range(10):
            self.assertIn(int(dist.sample()[0]), (8, 100, 222))

    def test_clipped_ppo_objective_is_finite(self):
        logits = torch.zeros(3, 4672, requires_grad=True)
        legal = torch.zeros(3, 4672, dtype=torch.bool)
        legal[:, :3] = True
        acts = torch.tensor([0, 1, 2])
        old = masked_distribution(logits.detach(), legal).log_prob(acts)
        values = torch.tensor([.1, -.2, .3], requires_grad=True)
        loss, stats = ppo_objective(logits, values, legal, acts, old,
                                    values.detach(), torch.tensor([1., 0., -1.]),
                                    torch.zeros(3))
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertAlmostEqual(stats['approx_kl'], 0, places=5)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(values.grad)

    def test_opening_history_reconstructed(self):
        openings = opening_sequences(5, 19)
        self.assertEqual(openings, opening_sequences(5, 19))
        board = board_from_opening(openings[0])
        self.assertEqual(len(board.move_stack), len(openings[0]))
        self.assertTrue(board.is_valid())

    def test_truncation_bootstraps_at_next_learner_turn(self):
        tiny = ChessNetwork(channels=8, blocks=1).eval()
        # Two actual decisions, no artificial terminal game result.
        sequences = opening_sequences(1, 33)
        experiences, result = rollout_game(tiny, tiny, sequences[0], chess.WHITE,
                                           'cpu', max_plies=2)
        self.assertTrue(result['truncated'])
        self.assertFalse(result['terminated'])
        self.assertEqual(result['reason'], 'MAX_PLIES')
        self.assertGreaterEqual(len(experiences), 1)
        self.assertTrue(all(np.isfinite(x['return']) for x in experiences))

    def test_one_forward_pass_and_initial_policy_consistency(self):
        # The same eval-mode logits must give the old log probability.
        tiny = ChessNetwork(channels=8, blocks=1).eval()
        board = chess.Board()
        move, sample = one_move(tiny, board, 'cpu', sample=True, keep_record=True)
        self.assertIn(move, board.legal_moves)
        with torch.no_grad():
            logits, _ = tiny(torch.tensor(sample['board'].astype(np.float32)).unsqueeze(0))
            legal = torch.tensor(sample['mask']).unsqueeze(0)
            lp = masked_distribution(logits, legal).log_prob(torch.tensor([sample['action']]))
        self.assertAlmostEqual(float(lp), sample['old_log_prob'], places=3)


if __name__ == '__main__':
    unittest.main()
