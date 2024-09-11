import functools

import torch

from ezgatr.primitives.bilinear import geometric_product


@functools.lru_cache(maxsize=None, typed=True)
def _compute_dualization(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the reversion index and sign flip for dualization."""

    perm = torch.tensor(range(15, -1, -1), device=device)
    sign = torch.tensor(
        [1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1],
        device=device,
        dtype=dtype,
    )
    return perm, sign


def dual(x: torch.Tensor) -> torch.Tensor:
    """Compute the dual of the input multi-vector.

    Parameters
    ----------
    x : torch.Tensor
        Multi-vectors with shape (..., 16).

    Returns
    -------
    torch.Tensor
        Corresponding dual with shape (..., 16).
    """
    perm, sign = _compute_dualization(x.device, x.dtype)
    return sign * torch.index_select(x, -1, perm)


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