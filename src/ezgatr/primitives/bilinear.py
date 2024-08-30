import functools
import os
import pathlib
from typing import Literal

import torch

_BASIS_FNAME: dict[str, str] = {
    "gp": os.path.join("basis", "geometric_product.pt"),
    "op": os.path.join("basis", "outer_product.pt"),
}
_BASIS_CACHE: dict[str, dict[tuple[torch.device, torch.dtype], torch.Tensor]] = {
    "gp": {},
    "op": {},
}


def _load_bilinear_basis(
    kind: Literal["gp", "op"], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Load the bilinear basis for geometric product or outer product.

    The bilinear basis is a 3D tensor with shape (16, 16, 16). One can
    understand the basis tensor by thinking of how the calculation of
    the bilinear maps between two multi-vectors expressed as coefficient
    vectors w.r.t. k-blades. When two multi-vectors are multiplied, each
    new coefficient corresponding to the k-blade comes from the "cartesian
    product" of the coefficients of the two multi-vectors. For example,
    the coefficient of the 1-blade ``e_1`` comes from multiple sources
    including ``gp(e_1, 1)`` and ``gp(e_0, e_01)``. So, the basis tensor
    actually defines a "computation graph" for the bilinear maps.

    The source basis are stored under the ``basis`` directory in `torch.float32`.
    The basis loader use the ``torch.float32`` tensor loaded to CPU as the prototype
    for all other devices and data types.

    Parameters
    ----------
    kind : Literal["gp", "op"]
        Kind of the bilinear basis, ``"gp"`` for geometric product and
        ``"op"`` for outer (wedge) product.
    device : torch.device
        Device for the basis.
    dtype : torch.dtype
        Data type for the basis.

    Returns
    -------
    torch.Tensor
        Bilinear basis with shape (16, 16, 16).
    """
    cache = _BASIS_CACHE[kind]
    basis = cache.get((device, dtype))
    if basis is not None:
        return basis

    proto_key = (torch.device("cpu"), torch.float32)
    try:
        cache[(device, dtype)] = cache[proto_key].detach().to(device, dtype)
    except KeyError:
        cache[proto_key] = torch.load(
            pathlib.Path(__file__).parent.resolve() / _BASIS_FNAME[kind]
        )

    return _load_bilinear_basis(kind, device, dtype)


@functools.lru_cache(maxsize=None, typed=True)
def _compute_inner_product_selector(device: torch.device) -> torch.Tensor:
    """Load the indices for PGA inner product to the device.

    PGA inner product operation exclude the coefficients corresponding to
    basis containing ``e_0``. The indices are hard-coded here. The reason
    to have this cached function is to avoid repeated copying from CPU to
    target multi-vector device using the ``torch.Tensor.to`` method.

    Parameters
    ----------
    device : torch.device
        Device for the indices.

    Returns
    -------
        torch.Tensor
    """
    return torch.tensor([0, 2, 3, 4, 8, 9, 10, 14], device=device)


def geometric_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Geometric product between two batches of multi-vectors.

    The input tensors ``x`` and ``y`` are multi-vectors with shape (..., 16).
    where ``...`` dimensions can denote batches or batches plus channels. When
    channel dimensions are present, the geometric product is calculated
    channel-wise (and batch-wise). For instance, the first channel of ``x[0]``
    is multiplied with the first channel of ``y[0]``, and so on. No channel-mixing
    here.

    Parameters
    ----------
    x : torch.Tensor
        Multi-vectors with shape (..., 16).
    y : torch.Tensor
        Multi-vectors with shape (..., 16).

    Returns
    -------
    torch.Tensor
        Multi-vectors with shape (..., 16).
    """
    basis = _load_bilinear_basis("gp", x.device, x.dtype)
    return torch.einsum("ijk, ...j, ...k -> ...i", basis, x, y)


def outer_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Outer product between two batches of multi-vectors.

    The input tensors ``x`` and ``y`` are multi-vectors with shape (..., 16).
    where ``...`` dimensions can denote batches or batches plus channels. When
    channel dimensions are present, the outer product is calculated channel-wise
    (and batch-wise). For instance, the first channel of ``x[0]`` is multiplied with
    the first channel of ``y[0]``, and so on. No channel-mixing here.

    Parameters
    ----------
    x : torch.Tensor
        Multi-vectors with shape (..., 16).
    y : torch.Tensor
        Multi-vectors with shape (..., 16).

    Returns
    -------
    torch.Tensor
        Multi-vectors with shape (..., 16).
    """
    basis = _load_bilinear_basis("op", x.device, x.dtype)
    return torch.einsum("ijk, ...j, ...k -> ...i", basis, x, y)


def inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the PGA inner product between two multi-vectors.

    Parameters
    ----------
    x : torch.Tensor
        Multi-vectors with shape (..., 16).
    y : torch.Tensor
        Multi-vectors with shape (..., 16).

    Returns
    -------
    torch.Tensor
        Inner product of the multi-vectors with shape (..., 1).
    """
    selector = _compute_inner_product_selector(x.device)
    return torch.einsum(
        "...i, ...i -> ...",
        torch.index_select(x, -1, selector),
        torch.index_select(y, -1, selector),
    ).unsqueeze(-1)
