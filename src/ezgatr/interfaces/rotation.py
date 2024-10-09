import torch


def encode_pga(quaternions: torch.Tensor) -> torch.Tensor:
    r"""Encode 3D rotation (as quaternions) into multi-vectors with PGA.

    Parameters
    ----------
    quaternions : torch.Tensor with shape (..., 4)
        Quaternions in ``ijkw`` order and Hamilton convention (``ijk = -1``)

    Returns
    -------
    torch.Tensor with shape (..., 16)
        Embedding into multivector.
    """
    ret = torch.zeros(
        *quaternions.shape[:-1], 16, dtype=quaternions.dtype, device=quaternions.device
    )

    # To represent a quaternion with bi-vectors:
    #   k: -e_12
    #   j: e_13
    #   i: -e_23
    ret[..., 0] = quaternions[..., 3]
    ret[..., 8] = -quaternions[..., 2]
    ret[..., 9] = quaternions[..., 1]
    ret[..., 10] = -quaternions[..., 0]

    return ret


def decode_pga(mvs: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    r"""Extract quaternions from multi-vectors with PGA.

    Parameters
    ----------
    mvs : torch.Tensor with shape (..., 16)
        Multivector.
    normalize : bool, default to False
        Whether to normalize the quaternion to unit norm.

    Returns
    -------
    torch.Tensor with shape (..., 4)
        Quaternions in ``ijkw`` order and Hamilton convention (``ijk = -1``)
    """
    ret = torch.cat(
        [
            -mvs[..., [10]],
            mvs[..., [9]],
            -mvs[..., [8]],
            mvs[..., [0]],
        ],
        dim=-1,
    )

    if normalize:
        ret = ret / ret.norm(dim=-1, keepdim=True)
    return ret
