"""Batched, search-free PPO self-play collection.

All boards receive one current-position network inference per move.
No lookahead, engine evaluation, MCTS, or move-sequence planning.
The only purpose of batching is to reduce per-inference GPU overhead.
"""

import numpy as np
import torch
import chess

from chess_rl.encoding import ACTION_COUNT, encode_board, legal_actions
from chess_rl.ppo_math import compute_gae, masked_distribution
from chess_rl.ppo_train import board_from_opening, final_score


def batched_moves(model, boards, device, sample=True, record_flags=None):
    """Select one legal action per board with ONE batched model call.

    record_flags determines whose transitions need PPO records. For sampling,
    each position has the same categorical distribution as ppo_train.one_move.
    Batching changes the order in which RNG draws are consumed, not the policy.
    """
    if not boards:
        return []
    if model.training:
        raise RuntimeError('Rollout model must be in eval mode (BatchNorm frozen).')
    if record_flags is None:
        record_flags = [False] * len(boards)
    if len(record_flags) != len(boards):
        raise ValueError('record_flags length must match boards.')

    observations = []
    masks = np.zeros((len(boards), ACTION_COUNT), dtype=np.bool_)
    choices = []
    for row, board in enumerate(boards):
        actions = legal_actions(board)
        if not actions:
            raise ValueError('No legal move in a batched position.')
        masks[row, list(actions)] = True
        observations.append(encode_board(board))
        choices.append(actions)

    states = np.stack(observations)
    with torch.no_grad():
        x = torch.from_numpy(states).to(device)
        legal = torch.from_numpy(masks).to(device)
        logits, values = model(x)
        dist = masked_distribution(logits, legal)
        actions = (dist.sample() if sample else
                   torch.argmax(logits.masked_fill(~legal, -1e9), dim=1))
        log_probs = dist.log_prob(actions)
    ids = actions.cpu().tolist()
    old_log_probs = log_probs.cpu().tolist()
    old_values = values.cpu().tolist()
    results = []
    for row, action in enumerate(ids):
        record = None
        if record_flags[row]:
            record = {
                'board': states[row].astype(np.float16),
                'mask': masks[row].copy(),
                'action': int(action),
                'old_log_prob': float(old_log_probs[row]),
                'value': float(old_values[row]),
            }
        results.append((choices[row][action], record))
    return results


def batch_bootstrap_values(model, boards, device):
    if not boards:
        return []
    with torch.no_grad():
        x = torch.from_numpy(np.stack([encode_board(board) for board in boards])).to(device)
        _, values = model(x)
    return [float(x) for x in values.cpu().tolist()]


def rollout_latest_batched(model, openings, learner_colors, device, max_plies=240):
    """Run independent learner-vs-frozen-latest games in parallel.

    Returns [(PPO experiences, game-result), ...] in the original game order.
    Terminal rewards, learner-turn cutoff and GAE match rollout_game semantics.
    This function is ONLY for latest-vs-latest, not the mixed historical pool.
    """
    if not openings:
        return []
    if len(openings) != len(learner_colors):
        raise ValueError('One learner color required per opening.')
    if max_plies < 1:
        raise ValueError('max_plies must be positive.')
    if model.training:
        raise RuntimeError('Frozen rollout model must be in eval mode.')

    games = []
    for opening, color in zip(openings, learner_colors):
        if color not in (chess.WHITE, chess.BLACK):
            raise ValueError('Invalid learner color.')
        games.append({'board': board_from_opening(opening), 'color': color,
                      'plies': 0, 'records': [], 'ended': False,
                      'terminated': False, 'truncated': False,
                      'winner': None, 'reason': '', 'bootstrap': 0.0})

    remaining = len(games)
    while remaining:
        to_move, move_indices = [], []
        to_bootstrap, bootstrap_indices = [], []
        for i, game in enumerate(games):
            if game['ended']:
                continue
            board = game['board']
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                game.update(ended=True, terminated=True, winner=outcome.winner,
                            reason=outcome.termination.name)
                remaining -= 1
            elif game['plies'] >= max_plies and board.turn == game['color']:
                game.update(ended=True, truncated=True, reason='MAX_PLIES')
                to_bootstrap.append(board)
                bootstrap_indices.append(i)
                remaining -= 1
            else:
                to_move.append(board)
                move_indices.append(i)

        for i, value in zip(bootstrap_indices,
                            batch_bootstrap_values(model, to_bootstrap, device)):
            games[i]['bootstrap'] = value

        if to_move:
            record_flags = [games[i]['board'].turn == games[i]['color'] for i in move_indices]
            selected = batched_moves(model, to_move, device, sample=True,
                                     record_flags=record_flags)
            for i, (move, record) in zip(move_indices, selected):
                game = games[i]
                if record is not None:
                    game['records'].append(record)
                game['board'].push(move)
                game['plies'] += 1

    results = []
    for game in games:
        records = game['records']
        if records:
            rewards = [0.0] * len(records)
            if game['terminated']:
                rewards[-1] = final_score(game['winner'], game['color'])
            adv, returns = compute_gae(rewards, [x['value'] for x in records],
                                       game['bootstrap'])
            for item, a, r in zip(records, adv, returns):
                item['advantage'] = float(a)
                item['return'] = float(r)
        result = {
            'terminated': game['terminated'], 'truncated': game['truncated'],
            'winner': game['winner'], 'reason': game['reason'],
            'plies': game['plies'],
            'outcome_for_learner': (final_score(game['winner'], game['color'])
                                    if game['terminated'] else None),
        }
        results.append((records, result))
    return results
