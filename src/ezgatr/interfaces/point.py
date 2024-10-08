from typing import Literal

import torch


def encode(kind: Literal["pga", "ega"], points: torch.Tensor) -> torch.Tensor:
    pass


def decode(
    kind: Literal["pga", "ega"], mvs: torch.Tensor, threshold: float = 1e-3
) -> torch.Tensor:
    pass
