from __future__ import annotations

import functools
from itertools import product

import torch

from ezgatr.nn.functional.linear import outer_product


@functools.lru_cache(maxsize=None, typed=True)
def _compute_dualization(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the reversion index and sign flip for dualization."""

    perm = torch.tensor(range(15, -1, -1), device=device)
    sign = torch.tensor(
        [1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1],
        device=device,
        dtype=dtype,
    )
    return perm, sign


@functools.lru_cache(maxsize=None, typed=True)
def _compute_efficient_join_kernel(
    device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    r"""Compute the kernel for efficient equivariant join computation."""

    kernel = torch.zeros((16, 16, 16), device=device, dtype=dtype)
    for i, j in product(range(16), repeat=2):
        x = torch.zeros(16, device=device, dtype=dtype)
        y = torch.zeros(16, device=device, dtype=dtype)

        x[i] = y[j] = 1.0
        kernel[:, i, j] = equi_dual(outer_product(equi_dual(x), equi_dual(y)))

    return kernel


def equi_dual(x: torch.Tensor) -> torch.Tensor:
    r"""Compute the dual of the input multi-vector.

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
    x: torch.Tensor, y: torch.Tensor, reference: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Compute the equivariant join of two multi-vectors given the reference.

    Parameters
    ----------
    x : torch.Tensor
        Multi-vectors with shape (..., 16).
    y : torch.Tensor
        Multi-vectors with shape (..., 16).
    reference : torch.Tensor, optional
        Multi-vectors with shape (..., 16).

    Returns
    -------
    torch.Tensor
        Equivariant join result multi-vectors with shape (..., 16).
    """
    kernel = _compute_efficient_join_kernel(x.device, x.dtype)
    ret = torch.einsum("ijk, ...j, ...k -> ...i", kernel, x, y)

    if reference is not None:
        ret *= reference[..., [14]]
    return ret
