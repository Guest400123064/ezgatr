from __future__ import annotations

import torch

from ezgatr.nn.functional import inner_product


def equi_rms_norm(
    x: torch.Tensor, weight: torch.Tensor | None = None, eps: float | None = None
) -> torch.Tensor:
    r"""Compute PGA inner-induced RMS norm of multi-vectors.

    Although the original GATr paper [2]_ named the normalization operation
    as E(3)-equivariant LayerNorm, we find the actual implementation more
    similar to RMSNorm [1]_ by substituting the scalar squaring with the inner
    product of multi-vectors, i.e., PGA inner-induced RMSNorm.

    We find the adaptation more intuitive by viewing each channel of the input
    multi-vector as an "imaginary number", so that the inner product of the
    multi-vector with itself is the squared "modulus". And, in turn, instead of
    thinking the input tensor as channels-of-multi-vectors, we can think of it
    as feature-vector-of-imaginary-numbers. And the normalization operation is
    the same as the RMSNorm operation on the scalar-valued feature vectors.

    NOTE: THIS DESIGN IS PROBABLY QUESTIONABLE. The inner product do not
    consider coefficients associated with blades containing ``e_0`` while
    the scaling is also applied to these blades.

    Parameters
    ----------
    x : torch.Tensor
        Input multi-vectors with shape (..., n_channels, 16).
    weight : torch.Tensor, optional
        weight for re-scaling the normalized input. It can be both
        static or learnable, depending on how ``weight`` are initialized
        outside of the function.
    eps : float, optional
        Small value to prevent division by zero.

    Returns
    -------
    torch.Tensor
        Normalized input with shape (..., n_channels, 16).

    Reference
    ---------
    .. [1] `"Root Mean Square Layer Normalization", Biao Zhang and Rico Sennrich,
            <https://ar5iv.labs.arxiv.org/html/1910.07467>`_
    .. [2] `"Geometric Algebra Transformer", Johann Brehmer, et al.,
            <https://ar5iv.labs.arxiv.org/html/2305.18415>`_
    """
    eps = eps or torch.finfo(x.dtype).eps
    norm = torch.mean(inner_product(x, x), dim=-2, keepdim=True)

    x = x / torch.sqrt(torch.clamp(norm, min=eps))
    if weight is not None:
        x = x * weight.view(-1, 1)
    return x
