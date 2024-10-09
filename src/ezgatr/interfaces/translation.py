import torch


def encode_pga(positions: torch.Tensor) -> torch.Tensor:
    r"""Encode translations into multi-vectors with PGA.

    We use the convention that translations are represented by the ``xyz`` coordinates
    of the translation vectors in 3D Euclidean space, i.e., end positions.

    Parameters
    ----------
    positions : torch.Tensor
        End positions of the translation vectors with shape (..., 3).

    Returns
    -------
    torch.Tensor
        Multi-vectors with PGA representation; tensor of shape (..., 16).
    """
    ret = torch.zeros(
        *positions.shape[:-1], 16, dtype=positions.dtype, device=positions.device
    )
    ret[..., 0] = 1.0
    ret[..., 5:8] = -0.5 * positions[..., :]

    return ret


def decode_pga(_: torch.Tensor) -> torch.Tensor:
    msg = "Translation interface is not equivariant and decode is not used."
    raise NotImplementedError(msg)
