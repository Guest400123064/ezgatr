import torch


def encode_pga(points: torch.Tensor) -> torch.Tensor:
    r"""Encode 3D points into multi-vectors with PGA.

    Parameters
    ----------
    points : torch.Tensor
        3D points with ``xyz`` coordinates; tensor of shape (..., 3).

    Returns
    -------
    torch.Tensor
        Multi-vectors with PGA representation; tensor of shape (..., 16).
    """
    ret = torch.zeros(*points.shape[:-1], 16, dtype=points.dtype, device=points.device)
    ret[..., 14] = 1.0
    ret[..., 13] = -points[..., 0]
    ret[..., 12] = points[..., 1]
    ret[..., 11] = -points[..., 2]

    return ret


def decode_pga(mvs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""Extract 3D points from multi-vectors with PGA.

    One needs to divide the coordinates by the homogeneous coordinate to obtain
    the 3D points. In this case, to prevent numerical instability, we set a threshold
    to the homogeneous coordinate, so that the division will only be performed when
    the absolute value of the homogeneous coordinate is above the threshold.

    Parameters
    ----------
    mvs : torch.Tensor
        Multi-vectors with PGA representation; tensor of shape (..., 16).
    eps : float, default to 1e-6
        Minimum value of the additional, un-physical component. Necessary to avoid
        exploding values or NaNs when this un-physical component of the homogeneous
        coordinates becomes too close to zero.

    Returns
    -------
    torch.Tensor
        3D points with ``xyz`` coordinates; tensor of shape (..., 3).
    """
    ret = torch.cat(
        [
            -mvs[..., [13]],
            mvs[..., [12]],
            -mvs[..., [11]],
        ],
        dim=-1,
    )
    hom = mvs[..., [14]]
    hom = torch.where(hom.abs() > eps, hom, eps * hom.sign())

    return ret / hom
