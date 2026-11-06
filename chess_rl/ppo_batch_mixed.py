"""Batched chess games against a fixed mix of frozen neural opponents.

Only the learner's moves enter PPO. Every move uses one current-position
inference, a legal action mask, and categorical sampling. No search or engine.
"""
from collections import defaultdict

import chess

from chess_rl.ppo_batch import batched_moves, batch_bootstrap_values
from chess_rl.ppo_math import compute_gae
from chess_rl.ppo_train import board_from_opening, final_score


def rollout_mixed_batched(learner, rivals, openings, learner_colors,
                          opponent_names, device, max_plies=240):
    """Return [(learner PPO transitions, game result)] in input order.

    rivals maps opponent labels to frozen models; 'current' selects learner.
    Opponent model may change *between* games, never *during* a game.
    """
    if not (len(openings) == len(learner_colors) == len(opponent_names)):
        raise ValueError('Require an opening, learner color, and opponent for every game.')
    if max_plies < 1:
        raise ValueError('max_plies must be positive.')
    if learner.training or any(model.training for model in rivals.values()):
        raise ValueError('All rollout networks must be in evaluation mode.')
    if any(name != 'current' and name not in rivals for name in opponent_names):
        raise ValueError('Unknown frozen opponent selected.')

    games = []
    for opening, color, rival in zip(openings, learner_colors, opponent_names):
        if color not in (chess.WHITE, chess.BLACK):
            raise ValueError('Invalid learner color.')
        games.append({'board': board_from_opening(opening), 'color': color,
                      'opponent': rival, 'plies': 0, 'records': [],
                      'ended': False, 'terminated': False, 'truncated': False,
                      'winner': None, 'reason': '', 'bootstrap': 0.0})

    remaining = len(games)
    while remaining:
        groups = defaultdict(list)
        bootstrap_boards, bootstrap_ids = [], []
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
                bootstrap_boards.append(board)
                bootstrap_ids.append(i)
                remaining -= 1
            else:
                # Group all positions requiring the current frozen policy,
                # including its learner moves and current-self opponent moves.
                group_name = ('current' if board.turn == game['color']
                              or game['opponent'] == 'current'
                              else game['opponent'])
                groups[group_name].append(i)

        if bootstrap_boards:
            for i, val in zip(bootstrap_ids,
                              batch_bootstrap_values(learner, bootstrap_boards, device)):
                games[i]['bootstrap'] = val

        # Do not update either player until every game in this batch is complete.
        # One network inference per position, grouped by which frozen model moves.
        for name, indices in groups.items():
            network = learner if name == 'current' else rivals[name]
            boards = [games[i]['board'] for i in indices]
            record_flags = [games[i]['board'].turn == games[i]['color']
                            for i in indices]
            chosen = batched_moves(network, boards, device, sample=True,
                                   record_flags=record_flags)
            for i, (move, sample) in zip(indices, chosen):
                game = games[i]
                if sample is not None:
                    game['records'].append(sample)
                game['board'].push(move)
                game['plies'] += 1

    out = []
    for game in games:
        records = game['records']
        if records:
            rewards = [0.0] * len(records)
            if game['terminated']:
                rewards[-1] = final_score(game['winner'], game['color'])
            advantages, returns = compute_gae(
                rewards, [record['value'] for record in records], game['bootstrap'])
            for sample, advantage, value in zip(records, advantages, returns):
                sample['advantage'] = float(advantage)
                sample['return'] = float(value)
        result = {
            'terminated': game['terminated'], 'truncated': game['truncated'],
            'winner': game['winner'], 'reason': game['reason'],
            'plies': game['plies'],
            'outcome_for_learner': (final_score(game['winner'], game['color'])
                                    if game['terminated'] else None),
        }
        out.append((records, result))
    return out
