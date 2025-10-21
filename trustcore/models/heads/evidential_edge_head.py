import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialEdgeHead(nn.Module):
    """
    A readout head for edge classification that predicts Dirichlet parameters α.
    Inputs are the pairwise node embeddings (h_u, h_v).
    """

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, fusion: str = "none"):
        super().__init__()
        assert n_classes >= 2
        assert fusion in ("none", "cumulative")
        self.n_classes = n_classes
        self.fusion = fusion
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

    @torch.no_grad()
    def _dirichlet_stats(self, alpha: torch.Tensor):
        S = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / S
        uncertainty = self.n_classes / S.squeeze(-1)
        return probs, uncertainty

    def _evidence(self, x: torch.Tensor):
        return F.softplus(x)

    def forward(
        self,
        h_u: torch.Tensor,
        h_v: torch.Tensor,
        aux_alphas: list | None = None,
    ) -> dict:
        x = torch.cat([h_u, h_v], dim=-1)
        evidence = self._evidence(self.mlp(x))
        alpha = evidence + 1.0

        if self.fusion == "cumulative" and aux_alphas:
            from trustcore.models.efg_components.dirichlet import cbf_fuse

            alpha = cbf_fuse([alpha] + aux_alphas)

        probs, uncertainty = self._dirichlet_stats(alpha)
        return {"alpha": alpha, "probs": probs, "uncertainty": uncertainty}
