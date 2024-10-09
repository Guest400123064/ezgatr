import torch

from ezgatr.interfaces import translation
from ezgatr.nn.functional.linear import geometric_product


def encode_pga(normals: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    r"""Encode planes into multi-vectors with PGA.

    Parameters
    ----------
    normals : torch.Tensor
        Normal vectors of the planes with shape (..., 3).
    positions : torch.Tensor
        End positions of the translation vectors with shape (..., 3).

    Returns
    -------
    torch.Tensor
        Multi-vectors with PGA representation; tensor of shape (..., 16).
    """
    ret = torch.zeros(
        *normals.shape[:-1], 16, dtype=normals.dtype, device=normals.device
    )
    ret[..., 2:5] = normals[..., :]
    ret = geometric_product(
        geometric_product(translation.encode_pga(positions), ret),
        translation.encode_pga(-positions),
    )
    return ret


def decode_pga(mvs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Extract normal and translation vectors from multi-vectors with PGA.

    Note that the translation vectors associated with each plane is not unique.
    In this case, we use the PGA convention that ``e_0`` coefficients are the
    distances from the origin to the plane along the normal direction to determine
    the translation vectors.

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
    ret = mvs[..., 2:5]
    unit_normals = ret / ret.norm(dim=-1, keepdim=True)

    return ret, mvs[..., [1]] * unit_normals
