import torch
import torch.nn.functional as F

from .dirichlet import dirichlet_nll, dirichlet_expected_probs


def evidence_cross_entropy(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Expectation of cross-entropy under Dirichlet: E[-log p_y].
    See dirichlet_nll. Returns mean across batch.
    """
    return dirichlet_nll(alpha, target).mean()


def dissonance_penalty(alpha: torch.Tensor) -> torch.Tensor:
    """
    Penalize conflicting strong beliefs across classes.
    A simple choice: average pairwise product of normalized beliefs.
    """
    p = dirichlet_expected_probs(alpha)
    conflict = p.unsqueeze(-1) * p.unsqueeze(-2)
    off_diag = conflict - torch.diag_embed(torch.diagonal(conflict, dim1=-2, dim2=-1))
    return off_diag.mean()


def false_confidence_penalty(alpha: torch.Tensor, target: torch.Tensor, margin: float = 0.75) -> torch.Tensor:
    """
    Penalize overconfident wrong beliefs by pushing max prob below margin when wrong.
    """
    p = dirichlet_expected_probs(alpha)
    max_p, pred = p.max(dim=-1)
    wrong = pred != target
    if wrong.any():
        return F.relu(max_p[wrong] - margin).mean()
    return torch.tensor(0.0, device=alpha.device)


def compose_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
    lambda_disson: float = 0.1,
    lambda_falseconf: float = 0.01,
) -> dict:
    L_ece = evidence_cross_entropy(alpha, target)
    L_dis = dissonance_penalty(alpha)
    L_fc = false_confidence_penalty(alpha, target)
    total = L_ece + lambda_disson * L_dis + lambda_falseconf * L_fc
    return {
        "loss": total,
        "L_ece": L_ece.detach(),
        "L_disson": L_dis.detach(),
        "L_falseconf": L_fc.detach(),
    }
