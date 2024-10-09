import functools
from typing import Any, Literal

import torch
import torch.nn.functional as F
from einops import rearrange

from ezgatr.nn.functional.linear import _compute_inner_product_selector


@functools.lru_cache(maxsize=None, typed=True)
def _compute_tri_vector_selector(device: torch.device) -> torch.Tensor:
    r"""Load the indices corresponds to tri-vectors to the device.

    Parameters
    ----------
    device : torch.device
        Device for the indices.

    Returns
    -------
        torch.Tensor
    """
    return torch.tensor([11, 12, 13, 14], device=device)


@functools.lru_cache(maxsize=None, typed=True)
def _compute_daa_qk_basis(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute basis queries and keys in the distance-aware attention.

    Parameters
    ----------
    device: torch.device
        Device for the basis.
    dtype: torch.dtype
        Data type for the basis.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Basis tensor for the queries and keys in the distance-aware attention.
        Both with shape (4, 4, 5).
    """
    bq = torch.zeros((4, 4, 5), device=device, dtype=dtype)
    bk = torch.zeros((4, 4, 5), device=device, dtype=dtype)
    r3 = torch.arange(3, device=device)

    bq[r3, r3, 0] = 1.0
    bk[3, 3, 0] = -1.0

    bq[3, 3, 1] = 1.0
    bk[r3, r3, 1] = -1.0

    bq[r3, 3, r3 + 2] = 1.0
    bk[r3, 3, r3 + 2] = 2.0

    return bq, bk


def _linear_square_normalizer(
    e123: torch.Tensor, eps: float | None = None
) -> torch.Tensor:
    r"""Apply linear square normalization to the input tensor.

    Parameters
    ----------
    e123 : torch.Tensor
        Coefficients corresponds to the ``e_123`` blade.
    eps : float, optional
        Small value to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Normalized multi-vector tensor.
    """
    eps = eps or torch.finfo(e123.dtype).eps
    return e123 / (e123.pow(2) + eps)


def compute_qk_for_daa(
    query: torch.Tensor,
    key: torch.Tensor,
    eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the query and key tensors for the distance-aware attention.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor with shape (B, H, T, qk_channels, 16).
    key : torch.Tensor
        Key tensor with shape (B, H, T, qk_channels, 16).
    eps : float, optional
        Small value to avoid division by zero used in the linear square normalizer.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Query and key tensors for the equivariant distance-aware attention.
    """

    def _build_dist_vec(q_or_k, basis):
        sel = _compute_tri_vector_selector(query.device)
        tri = torch.index_select(q_or_k, -1, sel)
        ret = tri * _linear_square_normalizer(tri[..., [3]], eps=eps)
        return torch.einsum("ijk, ...i, ...j -> ...k", basis, ret, ret)

    bq, bk = _compute_daa_qk_basis(query.device, query.dtype)
    return _build_dist_vec(query, bq), _build_dist_vec(key, bk)


def compute_qk_for_ipa(
    query: torch.Tensor, key: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the query and key tensors for the inner product attention.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor with shape (B, H, T, qk_channels, 16).
    key : torch.Tensor
        Key tensor with shape (B, H, T, qk_channels, 16).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Query and key tensors for the equivariant inner product attention
        with the channel-blade dimensions flattened.
    """

    def _build_inner_vec(q_or_k):
        sel = _compute_inner_product_selector(q_or_k.device)
        return torch.index_select(q_or_k, -1, sel)

    return _build_inner_vec(query), _build_inner_vec(key)


_ATTENTION_KIND_DISPATCH = {
    "ipa": compute_qk_for_ipa,
    "daa": compute_qk_for_daa,
}


def equi_geometric_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kinds: dict[Literal["ipa", "daa"], dict[str, Any] | None],
    weight: list[torch.Tensor | float] | None = None,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    r"""Equivariant geometric attention.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor with shape (B, H, T, qk_channels, 16).
    key : torch.Tensor
        Key tensor with shape (B, H, T, qk_channels, 16).
    value : torch.Tensor
        Value tensor with shape (B, H, T, v_channels, 16).
    kinds : dict[Literal["ipa", "daa"], dict[str, Any] | None]
        Kinds of similarity measures to consider in the attention calculation
        along with additional configuration/parameters sent to the corresponding
        query-key generating function. One should supply a dictionary mapping
        from kind to parameters in addition to query and key tensors. Use ``None``
        to denote no additional parameters supplied. Available options are:
            - "ipa": Inner product attention
            - "daa": Distance-aware attention
    weight : list[torch.Tensor | float], optional
        Weight tensor for the attention kinds. If not provided, the weights are
        set to 1.0 for all kinds to represent equal importance.
    attn_mask : torch.Tensor, optional
        Attention mask tensor.
    dropout_p : float, default to 0.0
        Dropout probability.
    is_causal : bool, default to False
        Whether to apply causal masking.
    scale : float, optional
        Scaling factor for the attention scores, overwriting the default scale
        determined by the hidden dimension.

    Returns
    -------
    torch.Tensor
        Output tensor with shape (B, H, T, v_channels, 16).
    """

    def _flatten_ck(mv):
        return rearrange(mv, "... c k -> ... (c k)")

    weight = weight or [1.0] * len(kinds)
    if len(kinds.keys()) != len(weight):
        msg = (
            "The length of the kinds and weight must be the same. "
            f"Got {len(kinds)} kinds and {len(weight)} weights."
        )
        raise ValueError(msg)

    qs, ks = [], []
    for (kind, kwargs), w in zip(kinds.items(), weight):
        q, k = _ATTENTION_KIND_DISPATCH[kind](query, key, **(kwargs or {}))  # type: ignore[operator]
        qs.append(_flatten_ck(q * w))
        ks.append(_flatten_ck(k))

    ret = F.scaled_dot_product_attention(
        torch.cat(qs, dim=-1),
        torch.cat(ks, dim=-1),
        _flatten_ck(value),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    return rearrange(ret, "... (c k) -> ... c k", k=16)
