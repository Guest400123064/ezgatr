import torch

from ezgatr.interfaces import translation
from ezgatr.nn.functional import geometric_product


def encode_pga(normals: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Encode planes into multi-vectors with PGA.

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

    ret = geometric_product(translation.encode_pga(positions), ret)
    return geometric_product(ret, translation.encode_pga(-positions))


def decode_pga(mvs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass
