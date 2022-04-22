import torch
from torch import nn

from vision_transformer.relu_squared import ReluSquared


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super(FeedForward, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            ReluSquared(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
