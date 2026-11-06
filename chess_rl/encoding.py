"""61-channel board encoding and 4,672-action AlphaZero-style move encoding.

No search or move evaluation is performed. python-chess supplies legal moves.
"""

import chess
import numpy as np

BOARD_CHANNELS = 61
MOVES_PER_SQUARE = 73
ACTION_COUNT = 64 * MOVES_PER_SQUARE

# Directions use the side-to-move's perspective (north means forward).
RAYS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1),
        (-1, -1), (-1, 0), (-1, 1)]
KNIGHTS = [(1, 2), (2, 1), (2, -1), (1, -2),
           (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def oriented_square(square, side_to_move):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    if side_to_move == chess.BLACK:
        file = 7 - file
        rank = 7 - rank
    return file, rank


def encode_board(board):
    """Return (61, 8, 8) float32, corresponding to the proposal's 8x8x61.

    Four snapshots: 12 piece planes + 2 repetition planes each = 56.
    Current castling rights = 4 planes. Half-move clock = 1 plane.
    """
    x = np.zeros((BOARD_CHANNELS, 8, 8), dtype=np.float32)
    perspective = board.turn
    past = board.copy(stack=True)

    for history_index in range(4):
        offset = 14 * history_index
        for square, piece in past.piece_map().items():
            file, rank = oriented_square(square, perspective)
            is_ours = piece.color == perspective
            channel = (0 if is_ours else 6) + piece.piece_type - 1
            x[offset + channel, 7 - rank, file] = 1.0

        x[offset + 12, :, :] = float(past.is_repetition(2))
        x[offset + 13, :, :] = float(past.is_repetition(3))
        if not past.move_stack:
            break
        past.pop()

    # Rights are ordered: our kingside, our queenside, their kingside, their queenside.
    enemy = not perspective
    rights = [board.has_kingside_castling_rights(perspective),
              board.has_queenside_castling_rights(perspective),
              board.has_kingside_castling_rights(enemy),
              board.has_queenside_castling_rights(enemy)]
    for i, right in enumerate(rights):
        x[56 + i, :, :] = float(right)

    x[60, :, :] = min(board.halfmove_clock / 150.0, 1.0)
    return x


def action_for_move(board, move):
    """Encode a move in a 0..4671 action ID, using the current side's view."""
    ff, fr = oriented_square(move.from_square, board.turn)
    tf, tr = oriented_square(move.to_square, board.turn)
    df, dr = tf - ff, tr - fr

    if move.promotion in UNDERPROMOTIONS:
        if dr != 1 or df not in (-1, 0, 1):
            raise ValueError("Unexpected underpromotion move")
        direction_index = df + 1
        piece_index = UNDERPROMOTIONS.index(move.promotion)
        move_type = 64 + 3 * piece_index + direction_index
    elif (df, dr) in KNIGHTS:
        move_type = 56 + KNIGHTS.index((df, dr))
    else:
        distance = max(abs(df), abs(dr))
        if distance < 1 or distance > 7:
            raise ValueError("Invalid move distance")
        direction = (df // distance, dr // distance)
        if direction not in RAYS:
            raise ValueError("Invalid move direction")
        move_type = RAYS.index(direction) * 7 + distance - 1

    # Row-major square order is aligned with the policy head's tensor layout.
    from_index = (7 - fr) * 8 + ff
    return from_index * MOVES_PER_SQUARE + move_type


def legal_actions(board):
    return {action_for_move(board, move): move for move in board.legal_moves}


def legal_mask(board):
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    for action in legal_actions(board):
        mask[action] = True
    return mask


def move_for_action(board, action):
    move = legal_actions(board).get(int(action))
    if move is None:
        raise ValueError(f"Action {action} is not legal for this board")
    return move
