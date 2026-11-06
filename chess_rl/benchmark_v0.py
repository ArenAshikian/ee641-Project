"""Matched GPU training benchmarks using REAL preprocessed Lichess positions.

This is a throughput test, not a training experiment or a model checkpoint.
Both networks see the same batches, loss, and optimizer settings. The
positions are loaded from existing September 1M shards, not synthetic input.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from chess_rl.network import ChessNetwork

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / 'data' / 'processed' / 'supervised_1000000_balanced'
DEFAULT_OUTPUT = ROOT / 'outputs' / 'v0_architecture_benchmark.json'


def load_real_batches(folder, batch_size, device, batch_count=4):
    """Read existing training shards once; reuse identical batches for both runs."""
    paths = sorted(folder.glob('train_*.npz'))
    if not paths:
        raise FileNotFoundError(
            f'No 1M dataset shards found in {folder}. Run run.cmd train-1m first.')
    boards, actions, outcomes = [], [], []
    needed = batch_size * batch_count
    for path in paths:
        with np.load(path) as data:
            boards.append(data['boards'].astype(np.float32))
            actions.append(data['actions'].astype(np.int64))
            outcomes.append(data['outcomes'].astype(np.float32))
        if sum(len(a) for a in actions) >= needed:
            break
    x = torch.from_numpy(np.concatenate(boards)[:needed]).to(device)
    a = torch.from_numpy(np.concatenate(actions)[:needed]).to(device)
    y = torch.from_numpy(np.concatenate(outcomes)[:needed]).to(device)
    if len(x) < needed:
        raise RuntimeError(f'Only {len(x)} real positions available; expected {needed}.')
    return [(x[i:i + batch_size], a[i:i + batch_size], y[i:i + batch_size])
            for i in range(0, needed, batch_size)]


def benchmark_architecture(channels, blocks, batches, device, steps=32, warmup=6):
    torch.manual_seed(1729)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = ChessNetwork(channels=channels, blocks=blocks).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def update(i):
        x, actions, targets = batches[i % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        logits, values = model(x)
        loss = F.cross_entropy(logits, actions) + 0.25 * F.mse_loss(values, targets)
        loss.backward()
        optimizer.step()
        return float(loss.detach().item())

    for i in range(warmup):
        update(i)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    last_loss = 0.0
    for i in range(steps):
        last_loss = update(warmup + i)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    count = steps * len(batches[0][0])
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    result = {
        'blocks': blocks,
        'channels': channels,
        'batch_size': len(batches[0][0]),
        'timed_steps': steps,
        'warmup_steps': warmup,
        'seconds': round(elapsed, 3),
        'positions_per_second': round(count / elapsed, 1),
        'peak_gpu_allocated_gib': round(peak, 3),
        'last_training_loss': round(last_loss, 5),
        'parameter_count': sum(p.numel() for p in model.parameters()),
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def run_benchmark(data_folder=DEFAULT_DATA, output=DEFAULT_OUTPUT, batch_size=128,
                  steps=32, warmup=6):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not available; check your PyTorch installation.')
    device = torch.device('cuda')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Loading matched real Lichess batches:', data_folder, flush=True)
    batches = load_real_batches(Path(data_folder), batch_size, device)
    results = []
    for blocks, channels in ((6, 96), (10, 128)):
        print(f'Benchmarking {blocks} blocks x {channels} channels...', flush=True)
        result = benchmark_architecture(channels, blocks, batches, device, steps, warmup)
        results.append(result)
        print(f'  {result["positions_per_second"]:,.0f} positions/sec; '
              f'{result["peak_gpu_allocated_gib"]:.2f} GiB peak allocated; '
              f'{result["parameter_count"]:,} parameters', flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'gpu': torch.cuda.get_device_name(0),
        'torch_version': torch.__version__,
        'data_folder': str(Path(data_folder)),
        'data_kind': 'real saved Lichess training positions, 4 cached batches',
        'test_description': 'Forward, loss, backward, optimizer step; float32, AdamW; identical batches for both models.',
        'limitations': ('Batches are cached on GPU. Excludes PGN extraction, shard loading, '
                        'validation, and game generation. Speed is not a PPO throughput estimate.'),
        'architectures': results,
    }
    output.write_text(json.dumps(data, indent=2), encoding='utf-8')
    for r in results:
        full = (20_000_000 * 3) / r['positions_per_second'] / 3600
        print(f'  Rough 20M x 3 epochs pure training-step time ({r["blocks"]}x{r["channels"]}): '
              f'{full:.1f} GPU-hours. Actual wall time will be longer.', flush=True)
    print('Saved benchmark:', output, flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(description='EE641 real-data architecture benchmark')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=6)
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.warmup) < 1:
        parser.error('All sizes must be positive.')
    run_benchmark(args.data, batch_size=args.batch_size, steps=args.steps,
                  warmup=args.warmup)


if __name__ == '__main__':
    main()
