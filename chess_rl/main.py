"""Small command-line entry points for the first project milestone."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import chess
import torch

from chess_rl.encoding import ACTION_COUNT, BOARD_CHANNELS, encode_board
from chess_rl.environment import ChessEnvironment
from chess_rl.network import ChessNetwork


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def demo(seed, max_plies):
    rng = random.Random(seed)
    env = ChessEnvironment(max_plies=max_plies, claim_draw=True)

    while True:
        status = env.status()
        if status.terminated or status.truncated:
            break
        env.step(rng.choice(list(env.board.legal_moves)))

    output = PROJECT_ROOT / "outputs"
    output.mkdir(exist_ok=True)
    (output / "sample_game.pgn").write_text(str(env.to_pgn()), encoding="utf-8")
    report = {
        "seed": seed,
        "plies": env.plies,
        "result": status.result,
        "terminated": status.terminated,
        "truncated": status.truncated,
        "reason": status.termination,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "sample_game.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Game demo result:", report)
    print("Saved outputs/sample_game.pgn and outputs/sample_game.json")
    print("This game used RANDOM legal moves, not an AI trained to play chess.")


def model_check():
    # Use small dimensions for a fast smoke test. Real experiment settings are in the proposal.
    model = ChessNetwork(channels=16, blocks=1).eval()
    board = chess.Board()
    position = torch.from_numpy(encode_board(board)).unsqueeze(0)
    with torch.no_grad():
        logits, values = model(position)
    move = model.choose_move(board)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Board shape:", tuple(position.shape))
    print("Policy shape:", tuple(logits.shape))
    print("Value shape:", tuple(values.shape))
    print("Example legal move from UNTRAINED model:", move.uci())
    assert position.shape == (1, BOARD_CHANNELS, 8, 8)
    assert logits.shape == (1, ACTION_COUNT)
    assert values.shape == (1,)


def main():
    parser = argparse.ArgumentParser(description="EE 641 Chess RL starter")
    parser.add_argument("command", choices=["demo", "model"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-plies", type=int, default=160)
    args = parser.parse_args()

    if args.command == "demo":
        demo(args.seed, args.max_plies)
    else:
        model_check()


if __name__ == "__main__":
    main()
