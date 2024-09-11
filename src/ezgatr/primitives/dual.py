import functools

import torch

from ezgatr.primitives.bilinear import geometric_product


def dual(x: torch.Tensor) -> torch.Tensor:
    pass


def equi_join(
    x: torch.Tensor, y: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Compute the equivariant join of two multi-vectors given the reference.

    Parameters
    ----------
    x : torch.Tensor
        Multi-vectors with shape (..., 16).
    y : torch.Tensor
        Multi-vectors with shape (..., 16).
    reference : torch.Tensor
        Multi-vectors with shape (..., 16).

    Returns
    -------
    torch.Tensor
        Equivariant join result multi-vectors with shape (..., 16).
    """
