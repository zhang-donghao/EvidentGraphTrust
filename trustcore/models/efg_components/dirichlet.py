import torch
import torch.nn.functional as F


def dirichlet_expected_probs(alpha: torch.Tensor) -> torch.Tensor:
    return alpha / alpha.sum(dim=-1, keepdim=True)


def dirichlet_nll(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood under Dirichlet-multinomial surrogate:
    For class y, E[-log p_y] = ψ(α_0) - ψ(α_y), where α_0 = sum_c α_c.
    """
    import torch.special as sps

    alpha0 = alpha.sum(dim=-1)
    alpha_y = alpha.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    return sps.digamma(alpha0) - sps.digamma(alpha_y)


def cbf_fuse(alpha_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Cumulative Belief Fusion (parameter-free): add evidences then +1.
    α_out = (Σ_k (α_k - 1)) + 1 = Σ_k α_k - (K - 1)
    All alphas must be >= 1.
    """
    assert len(alpha_list) >= 1
    fused = torch.stack(alpha_list, dim=0).sum(dim=0) - (len(alpha_list) - 1)
    return torch.clamp(fused, min=1.0001)
