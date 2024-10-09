import torch


def encode_pga(pseudoscalars: torch.Tensor) -> torch.Tensor:
    r"""Encode scalars into **the pseudoscalar dimension** of multi-vectors with PGA.

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
        *pseudoscalars.shape[:-1],
        15,
        dtype=pseudoscalars.dtype,
        device=pseudoscalars.device,
    )
    return torch.cat([pad, pseudoscalars], dim=-1)


def decode_pga(mvs: torch.Tensor) -> torch.Tensor:
    r"""Extract the pseudoscalar values from multi-vectors with PGA.

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
    return mvs[..., [15]]
