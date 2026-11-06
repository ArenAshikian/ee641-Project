"""One-forward-pass neural chess policy and value network.

This module is the untrained foundation. Supervised learning and PPO come next.
"""

import torch
from torch import nn

from chess_rl.encoding import ACTION_COUNT, BOARD_CHANNELS, encode_board, legal_mask, move_for_action


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(x + y)


class ChessNetwork(nn.Module):
    def __init__(self, channels=128, blocks=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy = nn.Conv2d(channels, 73, 1)
        self.value = nn.Sequential(
            nn.Conv2d(channels, 16, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, positions):
        x = self.blocks(self.stem(positions))
        # Each board square has 73 logits. Rearrange to square-major action order.
        policy_logits = self.policy(x).permute(0, 2, 3, 1).reshape(-1, ACTION_COUNT)
        values = self.value(x).squeeze(1)
        return policy_logits, values

    @torch.no_grad()
    def choose_move(self, board, device="cpu"):
        """One network forward pass, then a legal-move mask and argmax."""
        if board.is_game_over(claim_draw=True):
            raise ValueError("Cannot choose a move in a completed game")
        self.eval()
        x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
        mask = torch.from_numpy(legal_mask(board)).to(device)
        logits, _ = self(x)
        action = logits[0].masked_fill(~mask, -torch.inf).argmax().item()
        return move_for_action(board, action)
