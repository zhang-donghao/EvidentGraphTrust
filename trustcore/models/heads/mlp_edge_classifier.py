import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEdgeClassifier(nn.Module):
    """Simple MLP edge classifier producing logits."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, h_u: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_u, h_v], dim=-1)
        return self.mlp(x)
