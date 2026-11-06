"""Search-free chess PPO pilot and training.

Every PLAYED move requires just one policy/value network forward pass.
No Stockfish, minimax, alpha-beta, MCTS, board evaluation or engine labels.
python-chess is used ONLY to enforce chess rules and list legal moves.

This is a small first PPO experiment, not proof of improved playing strength.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chess
import numpy as np
import torch

from chess_rl.encoding import ACTION_COUNT, encode_board, legal_actions
from chess_rl.network import ChessNetwork
from chess_rl.ppo_math import compute_gae, masked_distribution, ppo_objective

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / 'checkpoints'
OUTPUTS = ROOT / 'outputs'
V0 = CHECKPOINTS / 'v0_supervised.pt'
V1M = CHECKPOINTS / 'supervised_1000000_balanced.pt'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_supervised(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = ChessNetwork(channels=int(saved['model_channels']),
                         blocks=int(saved['model_blocks'])).to(device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval()
    return model


def opening_sequences(count, seed):
    """Random legal moves, not a search or an opening book. Preserve move history."""
    rng = random.Random(seed)
    first_moves = ['e2e4', 'd2d4', 'c2c4', 'g1f3']
    openings = []
    seen = set()
    attempts = 0
    while len(openings) < count and attempts < count * 100:
        attempts += 1
        board = chess.Board()
        board.push_uci(first_moves[(attempts - 1) % len(first_moves)])
        for _ in range(3 + 2 * (len(openings) % 3)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over(claim_draw=True):
            continue
        moves = tuple(move.uci() for move in board.move_stack)
        if moves not in seen:
            openings.append(moves)
            seen.add(moves)
    if len(openings) < count:
        raise RuntimeError('Could not generate enough unique openings.')
    return openings


def board_from_opening(opening):
    board = chess.Board()
    for uci in opening:
        board.push_uci(uci)
    return board


def one_move(model, board, device, sample=False, keep_record=False):
    """One forward pass. Sample for PPO, argmax for evaluation."""
    actions = legal_actions(board)
    if not actions:
        raise ValueError('No legal moves.')
    ids = sorted(actions)
    legal = torch.zeros((1, ACTION_COUNT), dtype=torch.bool, device=device)
    legal[0, ids] = True
    observation = encode_board(board)
    with torch.no_grad():
        logits, value = model(torch.from_numpy(observation).unsqueeze(0).to(device))
        dist = masked_distribution(logits, legal)
        action = dist.sample() if sample else torch.argmax(logits.masked_fill(~legal, -1e9), dim=1)
        log_prob = dist.log_prob(action)
    index = int(action.item())
    record = None
    if keep_record:
        record = {
            'board': observation.astype(np.float16),
            'mask': legal[0].cpu().numpy(),
            'action': index,
            'old_log_prob': float(log_prob.item()),
            'value': float(value.item()),
        }
    return actions[index], record


def value_at_decision(model, board, device):
    """Value bootstrap at a cutoff, always from the learner's perspective."""
    with torch.no_grad():
        _, value = model(torch.from_numpy(encode_board(board)).unsqueeze(0).to(device))
    return float(value.item())


def final_score(winner, learner_color):
    if winner is None:
        return 0.0
    return 1.0 if winner == learner_color else -1.0


def rollout_game(learner, opponent, opening, learner_color, device, max_plies):
    """Complete game, or cutoff at a learner turn so bootstrapping is valid.

    One transition spans a learner decision AND the opponent's reply.
    The value head always sees a position where the learner is to move.
    """
    board = board_from_opening(opening)
    samples = []
    plies = 0
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            terminated, truncated = True, False
            winner = outcome.winner
            reason = outcome.termination.name
            break
        # Stop only on a learner decision boundary, not mid-opponent turn.
        if plies >= max_plies and board.turn == learner_color:
            terminated, truncated = False, True
            winner, reason = None, 'MAX_PLIES'
            break
        if board.turn == learner_color:
            move, record = one_move(learner, board, device, sample=True, keep_record=True)
            samples.append(record)
        else:
            move, _ = one_move(opponent, board, device, sample=True)
        board.push(move)
        plies += 1

    # A game can end before the learner moves, such as an opponent checkmate.
    # Record the outcome even though there are no transitions to train on.
    if not samples:
        return [], {'terminated': terminated, 'truncated': truncated,
                    'winner': winner, 'reason': reason, 'plies': plies,
                    'outcome_for_learner': final_score(winner, learner_color)
                    if terminated else None}
    rewards = [0.0] * len(samples)
    if terminated:
        rewards[-1] = final_score(winner, learner_color)
        bootstrap = 0.0
    else:
        if board.turn != learner_color:
            raise RuntimeError('Truncated without a learner decision state.')
        bootstrap = value_at_decision(learner, board, device)
    advantages, returns = compute_gae(rewards, [s['value'] for s in samples], bootstrap)
    for item, advantage, ret in zip(samples, advantages, returns):
        item['advantage'] = float(advantage)
        item['return'] = float(ret)
    return samples, {'terminated': terminated, 'truncated': truncated,
                     'winner': winner, 'reason': reason, 'plies': plies,
                     'outcome_for_learner': final_score(winner, learner_color) if terminated else None}


def ppo_update(model, optimizer, experiences, device, epochs, batch_size,
               clip, value_weight, entropy_weight, seed):
    if not experiences:
        raise ValueError('No learner transitions collected.')
    model.train()
    # Critical for PPO: use the same supervised BatchNorm running statistics
    # during rollout and optimization. Otherwise old/new log-probs are not comparable.
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.eval()
    states = np.stack([s['board'] for s in experiences])
    masks = np.stack([s['mask'] for s in experiences])
    actions = torch.tensor([s['action'] for s in experiences], device=device, dtype=torch.long)
    old_log_probs = torch.tensor([s['old_log_prob'] for s in experiences], device=device)
    old_values = torch.tensor([s['value'] for s in experiences], device=device)
    advantages = torch.tensor([s['advantage'] for s in experiences], device=device)
    returns = torch.tensor([s['return'] for s in experiences], device=device)
    # This normalizes across real player decision transitions, not turns of both sides.
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    rng = np.random.default_rng(seed)
    last = {}
    updates = 0
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(len(experiences)),
                                      max(1, (len(experiences) + batch_size - 1) // batch_size)):
            if len(indices) < 2:
                continue  # BatchNorm needs at least two positions per train batch.
            x = torch.from_numpy(states[indices].astype(np.float32)).to(device)
            legal = torch.from_numpy(masks[indices]).to(device)
            idx = torch.from_numpy(indices).to(device)
            logits, new_values = model(x)
            loss, last = ppo_objective(
                logits, new_values, legal, actions[idx],
                old_log_probs[idx], old_values[idx],
                advantages[idx], returns[idx],
                clip=clip, value_weight=value_weight, entropy_weight=entropy_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            updates += 1
    model.eval()
    if updates == 0:
        raise RuntimeError('No PPO updates: collect more learner decisions.')
    return {**last, 'optimizer_steps': updates, 'transitions': len(experiences)}


def read_pool(model, mode, iteration, device):
    """Frozen opponents. 50% recent snapshot, 50% permanent historical pool."""
    recent = copy.deepcopy(model).to(device).eval()
    historical = [('v0', V0)]
    if mode == 'train':
        for version in range(10, iteration, 10):
            file = CHECKPOINTS / f'ppo_v{version:04d}.pt'
            if not file.is_file():
                raise FileNotFoundError(f'Missing permanent checkpoint {file}')
            historical.append((f'v{version}', file))
    return recent, historical


def read_historical(path, device):
    if path == V0:
        return load_supervised(path, device)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = ChessNetwork(channels=saved['model_channels'], blocks=saved['model_blocks']).to(device)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()
    return model


def atomic_save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, path)


def write_history(rows, file):
    if not rows:
        return
    with file.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(mode, iterations, games_per_iter, seed, max_plies, device,
          ppo_epochs=2, batch_size=64, learning_rate=1e-5):
    CHECKPOINTS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    if not V0.is_file():
        raise FileNotFoundError(f'Missing permanent supervised model: {V0}')
    if mode not in ('pilot', 'train'):
        raise ValueError('Mode must be pilot or train.')
    # Pilot and full training use separate files so the pilot cannot contaminate an experiment.
    progress_path = CHECKPOINTS / f'ppo_{mode}_latest.pt'
    history_path = OUTPUTS / f'ppo_{mode}_metrics.csv'
    if mode == 'pilot':
        archive_folder = CHECKPOINTS / 'ppo_pilot_archives'
    else:
        archive_folder = CHECKPOINTS
    args_config = {
        'seed': seed, 'games_per_iter': games_per_iter, 'max_plies': max_plies,
        'ppo_epochs': ppo_epochs, 'batch_size': batch_size,
        'learning_rate': learning_rate, 'v0_sha256': sha256(V0),
        'mode': mode, 'clip': 0.1, 'gae_lambda': 0.95, 'gamma': 1.0,
        'policy_temperature': 1.0, 'value_weight': 0.5, 'entropy_weight': 0.01,
    }
    if progress_path.exists():
        saved = torch.load(progress_path, map_location='cpu', weights_only=True)
        if saved['config'] != args_config:
            raise RuntimeError('Existing PPO run settings or V0 file differ. Original run is preserved.')
        model = ChessNetwork(channels=saved['model_channels'], blocks=saved['model_blocks']).to(device)
        model.load_state_dict(saved['model_state_dict'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        first = int(saved['iteration']) + 1
        rows = saved['history']
        # Recover an archive after a crash between the two checkpoint writes.
        if (mode == 'train' and saved['iteration'] % 10 == 0):
            archive = CHECKPOINTS / f'ppo_v{saved["iteration"]:04d}.pt'
            if not archive.is_file():
                atomic_save(saved, archive)
        print(f'Resuming {mode} from iteration {first}.', flush=True)
    else:
        model = load_supervised(V0, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        first, rows = 1, []
        print(f'Starting {mode} from untouched V0.', flush=True)
    if first > iterations:
        print(f'Already completed iteration {first-1}. Nothing overwritten.', flush=True)
        return model
    # Deterministic per iteration. It is safe to resume at committed checkpoints.
    for i in range(first, iterations + 1):
        start = time.perf_counter()
        rng = random.Random(seed + i * 9127)
        torch.manual_seed(seed + i * 167)
        learner_snapshot, historical = read_pool(model, mode, i, device)
        historical_models = {name: read_historical(path, device) for name, path in historical}
        openings = opening_sequences(games_per_iter, seed + i * 17)
        experiences = []
        counts = Counter()
        lengths = []
        for g in range(games_per_iter):
            use_recent = rng.random() < 0.5
            if use_recent:
                name, rival = 'recent_frozen', learner_snapshot
            else:
                name = rng.choice(list(historical_models))
                rival = historical_models[name]
            candidate_color = chess.WHITE if g % 2 == 0 else chess.BLACK
            batch, result = rollout_game(learner_snapshot, rival, openings[g],
                                         candidate_color, device, max_plies)
            experiences.extend(batch)
            counts[name] += 1
            counts['games'] += 1
            if result['truncated']:
                counts['truncated'] += 1
            else:
                counts['wins' if result['outcome_for_learner'] > 0 else
                       'losses' if result['outcome_for_learner'] < 0 else 'draws'] += 1
            lengths.append(result['plies'])
        metrics = ppo_update(model, optimizer, experiences, device, ppo_epochs,
                             batch_size, clip=0.1, value_weight=0.5,
                             entropy_weight=0.01, seed=seed + i)
        row = {'iteration': i, 'games': counts['games'], 'wins': counts['wins'],
               'draws': counts['draws'], 'losses': counts['losses'],
               'truncated': counts['truncated'],
               'transitions': len(experiences),
               'mean_plies': round(float(np.mean(lengths)), 1),
               'recent_opponent_games': counts['recent_frozen'],
               'historical_opponent_games': counts['games'] - counts['recent_frozen'],
               **{k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()},
               'elapsed_seconds': round(time.perf_counter() - start, 1)}
        rows.append(row)
        checkpoint = {
            'iteration': i, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_blocks': 10, 'model_channels': 128,
            'config': args_config, 'history': rows,
            'description': 'Search-free PPO policy. V0 remains unchanged.',
            'saved_utc': datetime.now(timezone.utc).isoformat(),
        }
        # Historical archive has NEVER-overwrite semantics.
        if i % 10 == 0:
            archive_folder.mkdir(exist_ok=True)
            archive = archive_folder / f'ppo_v{i:04d}.pt'
            if archive.exists():
                raise FileExistsError(f'Permanent archive already exists: {archive}')
        atomic_save(checkpoint, progress_path)
        if i % 10 == 0:
            atomic_save(checkpoint, archive)
        write_history(rows, history_path)
        print(f'PPO {mode} {i}/{iterations}: W={counts["wins"]} D={counts["draws"]} '
              f'L={counts["losses"]} truncated={counts["truncated"]}; '
              f'{len(experiences)} learner moves; '
              f'policy_loss={metrics["policy_loss"]:.4f}, '
              f'value_loss={metrics["value_loss"]:.4f}, '
              f'KL={metrics["approx_kl"]:.4f}; '
              f'{row["elapsed_seconds"]:.0f}s', flush=True)
    print('Latest checkpoint:', progress_path, flush=True)
    print('Training metrics:', history_path, flush=True)
    return model


def evaluate(model, device, opening_count, seed, max_plies, name):
    """No-search, paired-color tournament against frozen V0 and 1M."""
    opponents = [('v0_supervised', V0), ('random_legal', None)]
    if V1M.exists():
        opponents.append(('supervised_1m', V1M))
    outputs = {}
    openings = opening_sequences(opening_count, seed)
    # Use same openings and colors for all checkpoints.
    for label, path in opponents:
        opponent = load_supervised(path, device) if path is not None else None
        tally = Counter()
        reasons = Counter()
        for opening in openings:
            for color in (chess.WHITE, chess.BLACK):
                board = board_from_opening(opening)
                rng = random.Random(seed + len(opening)*1009 + (1 if color else 2))
                plies = 0
                while True:
                    outcome = board.outcome(claim_draw=True)
                    if outcome is not None:
                        tally['wins' if outcome.winner == color else
                              'draws' if outcome.winner is None else 'losses'] += 1
                        reasons[outcome.termination.name] += 1
                        break
                    if plies >= max_plies:
                        tally['truncated'] += 1
                        reasons['MAX_PLIES'] += 1
                        break
                    if board.turn != color and opponent is None:
                        move = rng.choice(list(board.legal_moves))
                    else:
                        mover = model if board.turn == color else opponent
                        move, _ = one_move(mover, board, device, sample=False)
                    board.push(move)
                    plies += 1
        done = tally['wins'] + tally['draws'] + tally['losses']
        outputs[label] = {'wins': tally['wins'], 'draws': tally['draws'],
                          'losses': tally['losses'], 'truncated': tally['truncated'],
                          'score_completed': round((tally['wins'] + 0.5*tally['draws']) / done, 4) if done else None,
                          'termination_reasons': dict(reasons), 'games': opening_count * 2}
        print(f'{name} vs {label}: W={tally["wins"]} D={tally["draws"]} '
              f'L={tally["losses"]} truncated={tally["truncated"]}', flush=True)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('pilot', 'train', 'eval'), default='pilot')
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--games', type=int, default=None)
    parser.add_argument('--max-plies', type=int, default=240)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if min(args.max_plies, args.seed + 1) <= 0:
        parser.error('Max plies must be positive and seed nonnegative.')
    if not V0.is_file():
        parser.error(f'Missing {V0}. Run train-v0 first.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; strict no-search training and evaluation.', flush=True)
    if args.mode == 'pilot':
        print('Baseline before any PPO: no-search V0 matches.', flush=True)
        baseline = evaluate(load_supervised(V0, device), device, 4, 42, args.max_plies, 'V0')
        filename = OUTPUTS / 'ppo_pilot_v0_baseline.json'
        OUTPUTS.mkdir(exist_ok=True)
        filename.write_text(json.dumps(baseline, indent=2), encoding='utf-8')
        model = train('pilot', args.iterations or 2, args.games or 8, args.seed,
                      args.max_plies, device)
        after = evaluate(model, device, 4, 42, args.max_plies, 'PPO pilot')
        (OUTPUTS / 'ppo_pilot_after.json').write_text(json.dumps(after, indent=2), encoding='utf-8')
        print('Pilot done. This only tests correctness; it is NOT evidence of chess improvement.', flush=True)
    elif args.mode == 'train':
        train('train', args.iterations or 10, args.games or 32,
              args.seed, args.max_plies, device)
    else:
        file = CHECKPOINTS / 'ppo_train_latest.pt'
        if not file.exists():
            parser.error(f'Missing trained PPO checkpoint: {file}')
        saved = torch.load(file, map_location='cpu', weights_only=True)
        model = ChessNetwork(channels=saved['model_channels'], blocks=saved['model_blocks']).to(device)
        model.load_state_dict(saved['model_state_dict'])
        model.eval()
        evaluation = evaluate(model, device, 20, 42, args.max_plies, 'PPO trained')
        OUTPUTS.mkdir(exist_ok=True)
        (OUTPUTS / 'ppo_trained_eval.json').write_text(json.dumps(evaluation, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
