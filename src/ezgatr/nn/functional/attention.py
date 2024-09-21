from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from ezgatr.nn.functional.linear import _compute_inner_product_selector


def equi_scaled_inner_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute the most basic SDP attention induced by PGA inner product.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor with shape (..., seq_len, q_channels, 16).
    key : torch.Tensor
        Key tensor with shape (..., seq_len, k_channels, 16).
    value : torch.Tensor
        Value tensor with shape (..., seq_len, v_channels, 16).
    attn_mask : Optional[torch.Tensor], default to None
        Attention mask tensor with shape (..., seq_len, seq_len) or None.
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
        Output tensor with shape (..., seq_len, q_channels, 16).
    """
    selector = _compute_inner_product_selector(query.device)  # Assuming same QKV dev
    ein_expr = "... c k -> ... (c k)"

    return F.scaled_dot_product_attention(
        rearrange(torch.index_select(query, -1, selector), ein_expr),
        rearrange(torch.index_select(key, -1, selector), ein_expr),
        rearrange(torch.index_select(value, -1, selector), ein_expr),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
