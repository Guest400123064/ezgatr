import torch

from ezgatr.interfaces import plane


def encode(normals: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    r"""Encode reflection into multi-vectors with PGA.

    Plane serves as both the geometric object and the reflection operation in
    PGA. Therefore, we encode reflection over a plane as a plane itself.

    Parameters
    ----------
    normals : torch.Tensor
        Normal vectors of the reflection planes with shape (..., 3).
    positions : torch.Tensor
        End positions of the translation vectors with shape (..., 3).

    Returns
    -------
    torch.Tensor
        Multi-vectors with PGA representation; tensor of shape (..., 16).
    """
    return plane.encode(normals, positions)


def decode(mvs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Extract normal and position vectors of the reflection plane with PGA.

    Parameters
    ----------
    mvs : torch.Tensor
        Multi-vectors with PGA representation; tensor of shape (..., 16).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Normal vectors and end positions of the translation vectors;
        tensors of shape (..., 3).
    """
    return plane.decode(mvs)
