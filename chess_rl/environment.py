"""Chess rules and episode termination. No engine evaluation or search."""

from dataclasses import dataclass

import chess
import chess.pgn


@dataclass
class StepInfo:
    terminated: bool
    truncated: bool
    result: str
    termination: str | None
    winner: bool | None


class ChessEnvironment:
    def __init__(self, max_plies=300, claim_draw=True):
        self.max_plies = max_plies
        self.claim_draw = claim_draw
        self.reset()

    def reset(self, fen=None):
        self.board = chess.Board(fen) if fen else chess.Board()
        self.plies = 0
        self.initial_fen = self.board.fen()
        return self.board

    def status(self):
        outcome = self.board.outcome(claim_draw=self.claim_draw)
        if outcome is not None:
            return StepInfo(
                terminated=True,
                truncated=False,
                result=outcome.result(),
                termination=outcome.termination.name,
                winner=outcome.winner,
            )
        if self.plies >= self.max_plies:
            # An artificial cutoff is NOT a chess draw.
            return StepInfo(False, True, "*", "MAX_PLIES", None)
        return StepInfo(False, False, "*", None, None)

    def step(self, move):
        if self.status().terminated or self.status().truncated:
            raise ValueError("The game has ended. Call reset() first.")
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move.uci()}")

        self.board.push(move)
        self.plies += 1
        return self.status()

    def terminal_reward(self, color):
        """+1 win, 0 draw, -1 loss, from the specified color's perspective.

        This function must only be used after a real chess termination.
        Truncated episodes are bootstrapped later in PPO, not labeled draws.
        """
        status = self.status()
        if not status.terminated:
            raise ValueError("Terminal reward is undefined before real game termination.")
        if status.winner is None:
            return 0.0
        return 1.0 if status.winner == color else -1.0

    def to_pgn(self, event="EE641 smoke test"):
        game = chess.pgn.Game.from_board(self.board)
        game.headers["Event"] = event
        game.headers["Result"] = self.status().result
        game.headers["Termination"] = self.status().termination or "UNKNOWN"
        return game
