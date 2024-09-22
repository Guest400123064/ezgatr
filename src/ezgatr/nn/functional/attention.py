import functools
from typing import Optional, Literal

import torch
import torch.nn.functional as F
from einops import rearrange

from ezgatr.nn.functional.linear import _compute_inner_product_selector


@functools.lru_cache(maxsize=None, typed=True)
def _compute_tri_vector_selector(device: torch.device) -> torch.Tensor:
    """Load the indices corresponds to tri-vectors to the device.

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
def _compute_daa_qk_basis():
    pass


def compute_qk_for_daa(
    query: torch.Tensor, key: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the query and key tensors for the distance-aware attention.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor with shape (B, H, T, qk_channels, 16).
    key : torch.Tensor
        Key tensor with shape (B, H, T, qk_channels, 16).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Query and key tensors for the equivariant distance-aware attention.
        with the channel-blade dimensions flattened.
    """
    raise NotImplementedError


def compute_qk_for_ipa(
    query: torch.Tensor, key: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the query and key tensors for the inner product attention.

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

    def _select_rearrange(mv):
        sel = _compute_inner_product_selector(mv.device)
        exp = "... c k -> ... (c k)"
        return rearrange(torch.index_select(mv, -1, sel), exp)

    return _select_rearrange(query), _select_rearrange(key)


_ATTENTION_KIND_DISPATCH = {
    "ipa": compute_qk_for_ipa,
    "daa": compute_qk_for_daa,
}


def equi_geometric_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kinds: list[Literal["ipa", "daa"]],
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Equivariant geometric attention.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor with shape (B, H, T, qk_channels, 16).
    key : torch.Tensor
        Key tensor with shape (B, H, T, qk_channels, 16).
    value : torch.Tensor
        Value tensor with shape (B, H, T, v_channels, 16).
    kinds : list[Literal["ipa", "daa"]]
        Kinds of similarity measures to consider in the attention calculation.
        One should supply a list of attention kinds. Available options are:
            - "ipa": Inner product attention.
            - "daa": Distance-aware attention
    attn_mask : Optional[torch.Tensor], default to None
        Attention mask tensor.
    dropout_p : float, default to 0.0
        Dropout probability.
    is_causal : bool, default to False
        Whether to apply causal masking.
    scale : Optional[float], default to None
        Scaling factor for the attention scores, overwriting the default scale
        determined by the hidden dimension.

    Returns
    -------
    torch.Tensor
        Output tensor with shape (B, H, T, v_channels, 16).
    """
    qs, ks = zip(_ATTENTION_KIND_DISPATCH[kind](query, key) for kind in kinds)
    ret = F.scaled_dot_product_attention(
        torch.cat(qs, dim=-1),
        torch.cat(ks, dim=-1),
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    return rearrange(ret, "... (v c) -> ... v c", c=16)
