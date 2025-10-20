"""Reproducibility helpers."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = None) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
