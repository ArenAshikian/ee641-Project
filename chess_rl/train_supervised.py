"""Short supervised Phase 1 sanity check on Lichess games.

This is a pilot run using 3,000 sampled positions and a small CNN by default.
It does not produce the final V0 model or claim a meaningful chess strength.
"""

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from chess_rl.lichess_data import DEFAULT_PGN, PROJECT_ROOT, build_pilot_dataset
from chess_rl.network import ChessNetwork


def make_loader(path, batch_size, shuffle):
    with np.load(path) as data:
        x = torch.from_numpy(data['boards'].astype(np.float32))
        actions = torch.from_numpy(data['actions'].astype(np.int64))
        values = torch.from_numpy(data['outcomes'].astype(np.float32))
    return DataLoader(TensorDataset(x, actions, values), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss, total_policy, total_value, total_correct, count = 0., 0., 0., 0, 0
    for boards, actions, outcomes in loader:
        boards = boards.to(device)
        actions = actions.to(device)
        outcomes = outcomes.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits, predictions = model(boards)
            policy_loss = nn.functional.cross_entropy(logits, actions)
            value_loss = nn.functional.mse_loss(predictions, outcomes)
            loss = policy_loss + 0.25 * value_loss
            if training:
                loss.backward()
                optimizer.step()
        n = boards.size(0)
        total_loss += loss.item() * n
        total_policy += policy_loss.item() * n
        total_value += value_loss.item() * n
        total_correct += (logits.argmax(dim=1) == actions).sum().item()
        count += n
    return {'loss': total_loss / count, 'policy_loss': total_policy / count,
            'value_mse': total_value / count,
            'human_move_accuracy': total_correct / count}


def main():
    parser = argparse.ArgumentParser(description='EE641 supervised chess pilot')
    parser.add_argument('--source', type=Path, default=DEFAULT_PGN)
    parser.add_argument('--positions', type=int, default=3000)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    if args.positions < 100 or args.epochs < 1 or args.batch_size < 1:
        parser.error('Need at least 100 positions, one epoch, and a positive batch size.')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('PyTorch:', torch.__version__, '| Device:', device, flush=True)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0), flush=True)

    processed = PROJECT_ROOT / 'data' / 'processed'
    train_file = processed / 'pilot_train.npz'
    val_file = processed / 'pilot_val.npz'
    stats_file = processed / 'pilot_stats.json'
    stale = True
    if train_file.exists() and val_file.exists() and stats_file.exists() and not args.rebuild:
        stats = json.loads(stats_file.read_text(encoding='utf-8'))
        stale = stats['source'] != args.source.name or stats['seed'] != args.seed or (
            stats['train_positions'] + stats['validation_positions'] != args.positions)
    if stale:
        build_pilot_dataset(args.source, max_positions=args.positions, seed=args.seed, folder=processed)
    else:
        print('Reusing existing pilot data. Use --rebuild to re-extract.', flush=True)

    train_loader = make_loader(train_file, args.batch_size, shuffle=True)
    val_loader = make_loader(val_file, args.batch_size, shuffle=False)
    model = ChessNetwork(channels=args.channels, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    out = PROJECT_ROOT / 'outputs'
    out.mkdir(exist_ok=True)
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        val_metrics = run_epoch(model, val_loader, None, device)
        rows.append({'epoch': epoch, **{'train_' + k: v for k, v in train_metrics.items()},
                     **{'val_' + k: v for k, v in val_metrics.items()}})
        print(f"Epoch {epoch}: train loss {train_metrics['loss']:.4f}, "
              f"train accuracy {train_metrics['human_move_accuracy']:.1%} | "
              f"val loss {val_metrics['loss']:.4f}, "
              f"val accuracy {val_metrics['human_move_accuracy']:.1%}", flush=True)

    with (out / 'supervised_pilot_metrics.csv').open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checkpoints = PROJECT_ROOT / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    path = checkpoints / 'supervised_pilot.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_channels': args.channels,
        'model_blocks': args.blocks,
        'seed': args.seed,
        'epochs': args.epochs,
        'positions': args.positions,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'note': 'Small supervised pilot. Not final supervised V0.',
        'last_validation_metrics': val_metrics,
    }, path)
    print('Saved model:', path, flush=True)
    print('Saved metrics:', out / 'supervised_pilot_metrics.csv', flush=True)
    print('This is a supervised PILOT, not a complete model or PPO training.', flush=True)


if __name__ == '__main__':
    main()
