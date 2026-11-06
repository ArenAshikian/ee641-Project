"""Compact, lossless storage for the existing 61-plane chess encoding.

Planes 0..59 contain only 0 or 1. Plane 60 is halfmove_clock / 150,
constant over each square. Store 3840 bits + one byte instead of 3904 floats.
No change is made to chess_rl.encoding.encode_board().
"""

from pathlib import Path
import os

import numpy as np


PIECE_BITS = 60 * 8 * 8
PACKED_BYTES = PIECE_BITS // 8


def pack_boards(boards):
    boards = np.asarray(boards, dtype=np.float32)
    if boards.ndim != 4 or boards.shape[1:] != (61, 8, 8):
        raise ValueError('Expected board tensor shape (N, 61, 8, 8)')
    binary = boards[:, :60].reshape(len(boards), -1)
    if not np.all((binary == 0) | (binary == 1)):
        raise ValueError('Expected binary board planes 0..59')
    if not np.allclose(boards[:, 60], boards[:, 60, :1, :1], atol=1e-6):
        raise ValueError('Expected uniform halfmove plane')
    counters = np.rint(boards[:, 60, 0, 0] * 150).clip(0, 150).astype(np.uint8)
    packed = np.packbits(binary.astype(np.uint8), axis=1, bitorder='little')
    return packed, counters


def unpack_boards(packed, halfmove):
    packed = np.asarray(packed, dtype=np.uint8)
    halfmove = np.asarray(halfmove, dtype=np.uint8)
    if packed.ndim != 2 or packed.shape[1] != PACKED_BYTES or len(packed) != len(halfmove):
        raise ValueError('Invalid packed board dimensions')
    bits = np.unpackbits(packed, axis=1, bitorder='little').reshape(-1, 60, 8, 8)
    boards = np.empty((len(bits), 61, 8, 8), dtype=np.float16)
    boards[:, :60] = bits
    boards[:, 60] = (halfmove.astype(np.float32) / 150.0)[:, None, None]
    return boards


def write_shard(folder, split, index, rows):
    """Atomically write one NPZ shard, using memory bounded by rows size."""
    if not rows:
        return index
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    boards, halfmove = pack_boards(np.asarray([r[0] for r in rows], dtype=np.float32))
    final = folder / f'{split}_{index:05d}.npz'
    partial = folder / f'{split}_{index:05d}.partial.npz'
    np.savez_compressed(
        partial, packed=boards, halfmove=halfmove,
        actions=np.asarray([r[1] for r in rows], dtype=np.int32),
        outcomes=np.asarray([r[2] for r in rows], dtype=np.float32),
        legal_counts=np.asarray([r[3] for r in rows], dtype=np.int16),
        thirds=np.asarray([r[4] for r in rows], dtype=np.uint8),
    )
    os.replace(partial, final)
    rows.clear()
    return index + 1


def load_shard(file):
    with np.load(file) as obj:
        boards = unpack_boards(obj['packed'], obj['halfmove'])
        actions = obj['actions'].astype(np.int64)
        outcomes = obj['outcomes'].astype(np.float32)
        thirds = obj['thirds'].astype(np.int64)
        legal_counts = obj['legal_counts'].astype(np.int16)
    return boards, actions, outcomes, thirds, legal_counts
