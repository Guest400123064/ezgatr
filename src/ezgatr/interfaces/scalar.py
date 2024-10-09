import torch


def encode_pga(scalars: torch.Tensor) -> torch.Tensor:
    r"""Encode scalars into multi-vectors with PGA.

    This function **assumes that the scalar tensor has shape (..., 1)**
    and will pad the remaining 15 components with zeros.

    Parameters
    ----------
    scalars : torch.Tensor
        Scalars with shape (..., 1).

    Returns
    -------
    torch.Tensor
        Multi-vectors with PGA representation; tensor of shape (..., 16).
    """
    pad = torch.zeros(
        *scalars.shape[:-1], 15, dtype=scalars.dtype, device=scalars.device
    )
    return torch.cat([scalars, pad], dim=-1)


def decode_pga(mvs: torch.Tensor) -> torch.Tensor:
    r"""Extract scalars from multi-vectors with PGA.

    This function do not automatically squeeze the last dimension when returning
    the scalar tensor.

    Parameters
    ----------
    mvs : torch.Tensor
        Multi-vectors with PGA representation; tensor of shape (..., 16).

    Returns
    -------
    torch.Tensor
        Scalars with shape (..., 1).
    """
    return mvs[..., [0]]
