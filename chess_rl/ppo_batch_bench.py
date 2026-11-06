"""Read-only CPU/GPU benchmark for sequential versus batched PPO rollouts."""

import argparse
import json
import time

import chess
import numpy as np
import torch

from chess_rl.human_openings import MANIFEST, digest_openings, training_openings
from chess_rl.ppo_batch import batched_moves, rollout_latest_batched
from chess_rl.ppo_train import CHECKPOINTS, OUTPUTS, board_from_opening, one_move, read_historical, rollout_game, sha256

SOURCE = CHECKPOINTS / 'ppo_scale_latest' / 's0049.pt'
EXPECTED_HASH = '75bec742766a4c9709211ade12719064bc0eb6405a4931314cd7d2178f04a748'


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def prepare():
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Missing saved frozen Latest S49: {SOURCE}')
    got_hash = sha256(SOURCE)
    if got_hash != EXPECTED_HASH:
        raise RuntimeError('Latest S49 hash changed. Expected the frozen live bot checkpoint.')
    if not MANIFEST.is_file():
        raise FileNotFoundError(f'Run PPO opening preparation first: {MANIFEST}')
    opening_doc = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if digest_openings(opening_doc) != opening_doc.get('sha256'):
        raise RuntimeError('Saved human opening manifest changed.')
    return opening_doc


def compare_greedy(model, openings, device):
    """Check exact decisions/inputs and bounded GPU floating-point differences.

    Different CUDA convolution algorithms can produce slightly different float32
    outputs for batch=1 versus batch>1. A difference below 1e-4 was too strict
    for the full 10-block network, even when the selected action matched.
    """
    boards = [board_from_opening(o) for o in openings]
    singles = [one_move(model, b, device, sample=False, keep_record=True)
               for b in boards]
    batches = batched_moves(model, boards, device, sample=False,
                            record_flags=[True] * len(boards))

    # Exact checks must always pass. Numeric checks below are deliberately
    # bounded; we never simply suppress a failed policy comparison.
    max_value_difference = 0.0
    max_log_prob_difference = 0.0
    value_tolerance = 0.005
    log_prob_tolerance = 0.02
    for idx, (a, b) in enumerate(zip(singles, batches)):
        if a[0] != b[0]:
            raise RuntimeError(f'Position {idx}: batched policy selected a different greedy move: '
                               f'serial={a[0]}, batched={b[0]}')
        if a[1]['action'] != b[1]['action']:
            raise RuntimeError(f'Position {idx}: batched action ID changed.')
        if not np.array_equal(a[1]['mask'], b[1]['mask']):
            raise RuntimeError(f'Position {idx}: batched legal-action mask changed.')
        if not np.array_equal(a[1]['board'], b[1]['board']):
            raise RuntimeError(f'Position {idx}: batched board encoder changed.')

        value_diff = abs(a[1]['value'] - b[1]['value'])
        logp_diff = abs(a[1]['old_log_prob'] - b[1]['old_log_prob'])
        max_value_difference = max(max_value_difference, value_diff)
        max_log_prob_difference = max(max_log_prob_difference, logp_diff)
        if not np.isfinite(value_diff) or value_diff > value_tolerance:
            raise RuntimeError(
                f'Position {idx}: critic value difference {value_diff:.7g} '
                f'exceeded {value_tolerance}. '
                f'serial={a[1]["value"]:.7g}, batched={b[1]["value"]:.7g}')
        if not np.isfinite(logp_diff) or logp_diff > log_prob_tolerance:
            raise RuntimeError(
                f'Position {idx}: policy log-prob difference {logp_diff:.7g} '
                f'exceeded {log_prob_tolerance}. '
                f'serial={a[1]["old_log_prob"]:.7g}, '
                f'batched={b[1]["old_log_prob"]:.7g}')

    # A close log-probability for the best move alone does not prove that the
    # complete distribution stayed close. Audit *all* legal moves too.
    from chess_rl.encoding import encode_board
    from chess_rl.ppo_math import masked_distribution

    states = np.stack([encode_board(board) for board in boards])
    masks = np.stack([item[1]['mask'] for item in singles])
    with torch.no_grad():
        batch_x = torch.from_numpy(states).to(device)
        batch_mask = torch.from_numpy(masks).to(device)
        batch_logits, _ = model(batch_x)
        batch_probs = masked_distribution(batch_logits, batch_mask).probs.cpu().numpy()
        serial_probs = []
        for state, mask in zip(states, masks):
            one_x = torch.from_numpy(state).unsqueeze(0).to(device)
            one_mask = torch.from_numpy(mask).unsqueeze(0).to(device)
            one_logits, _ = model(one_x)
            serial_probs.append(masked_distribution(one_logits, one_mask).probs[0].cpu().numpy())

    tv_differences = [0.5 * float(np.abs(a - b).sum())
                      for a, b in zip(serial_probs, batch_probs)]
    max_tv = max(tv_differences, default=0.0)
    print(f'GPU numerical audit: max |value difference|={max_value_difference:.7g}, '
          f'max |log-prob difference|={max_log_prob_difference:.7g}, '
          f'max policy total-variation distance={max_tv:.7g}', flush=True)
    if not np.isfinite(max_tv) or max_tv > 0.02:
        raise RuntimeError('Full policy distributions differed too much between '
                           f'one-at-a-time and batched inference: max TV={max_tv:.7g} > 0.02')
    return len(boards)


def run_bench(games=32, max_plies=80, batch_size=32, smoke=False):
    doc = prepare()
    if games < 2 or games % 2 or batch_size < 2 or max_plies < 2:
        raise ValueError('Use positive even games, batch size >=2 and max plies >=2.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = read_historical(SOURCE, device).eval()
    openings = training_openings(doc, games, 641999)
    colors = [chess.WHITE if i % 2 == 0 else chess.BLACK for i in range(games)]

    # Warmup: avoid counting first-time CUDA/kernel initialization in the timings.
    small = [board_from_opening(o) for o in openings[:min(games, 8)]]
    batched_moves(model, small, device, sample=False)
    synchronize(device)
    checked = compare_greedy(model, openings[:min(games, 16)], device)
    print(f'Checkpoint hash OK. Matched {checked} greedy moves, legality masks, value and log-probs.', flush=True)
    print(f'Device={device}, games={games}, max_plies={max_plies}, batch_size={batch_size}', flush=True)

    torch.manual_seed(641)
    synchronize(device)
    start = time.perf_counter()
    sequential = [rollout_game(model, model, opening, color, device, max_plies)
                  for opening, color in zip(openings, colors)]
    synchronize(device)
    sequential_seconds = time.perf_counter() - start

    torch.manual_seed(641)
    synchronize(device)
    start = time.perf_counter()
    batched = []
    for offset in range(0, games, batch_size):
        batched.extend(rollout_latest_batched(model, openings[offset:offset+batch_size],
                                              colors[offset:offset+batch_size], device,
                                              max_plies=max_plies))
    synchronize(device)
    batch_seconds = time.perf_counter() - start

    def totals(result):
        return {'plies': sum(r['plies'] for _, r in result),
                'transitions': sum(len(samples) for samples, _ in result),
                'truncated': sum(bool(r['truncated']) for _, r in result)}
    serial_totals, batch_totals = totals(sequential), totals(batched)
    if len(batched) != games:
        raise RuntimeError('Lost games in batch implementation.')
    for samples, r in batched:
        if not r['terminated'] and not r['truncated']:
            raise RuntimeError('Game did not reach a termination or safe cutoff.')
        if samples and (not np.isfinite(samples[-1]['advantage']) or
                        not np.isfinite(samples[-1]['return'])):
            raise RuntimeError('Invalid PPO return or advantage.')
    report = {
        'checkpoint_sha256': EXPECTED_HASH, 'device': str(device), 'games': games,
        'max_plies': max_plies, 'batch_size': batch_size,
        'sequential_seconds': round(sequential_seconds, 3),
        'batch_seconds': round(batch_seconds, 3),
        'sequential': serial_totals, 'batched': batch_totals,
        'sequential_plies_per_sec': round(serial_totals['plies']/sequential_seconds, 2),
        'batch_plies_per_sec': round(batch_totals['plies']/batch_seconds, 2),
        'observed_throughput_speedup': round((batch_totals['plies']/batch_seconds) /
                                             (serial_totals['plies']/sequential_seconds), 3),
        'note': ('Trajectories differ because batched sampling changes random draw order. '
                 'This compares throughput, not exact identical games. No weights changed.'),
    }
    print('Serial:  ', report['sequential_plies_per_sec'], 'plies/sec,',
          report['sequential_seconds'], 'seconds', flush=True)
    print('Batched: ', report['batch_plies_per_sec'], 'plies/sec,',
          report['batch_seconds'], 'seconds', flush=True)
    print('Observed speedup:', report['observed_throughput_speedup'], 'x', flush=True)
    print('No checkpoint, optimizer, or training data was modified.', flush=True)
    if not smoke:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        path = OUTPUTS / 'ppo_batch_benchmark.json'
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Saved:', path, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--games', type=int, default=32)
    parser.add_argument('--max-plies', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.smoke:
        args.games, args.max_plies, args.batch_size = 4, 16, 4
    run_bench(args.games, args.max_plies, args.batch_size, smoke=args.smoke)
