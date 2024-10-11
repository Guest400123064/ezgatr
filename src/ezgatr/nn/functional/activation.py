from __future__ import annotations

from typing import Literal

import torch


def scaler_gated_gelu(
    x: torch.Tensor, approximate: Literal["none", "tanh"] = "tanh"
) -> torch.Tensor:
    r"""Compute scaler-gated GeLU activation function.

    Parameters
    ----------
    x : torch.Tensor
        Input batch of multi-vectors.
    approximate : Literal["none", "tanh"], default to "tanh"
        Approximation method for the GeLU function. Default to "tanh".

    Returns
    -------
    torch.Tensor
        Output batch of multi-vectors gated by scalar blade.
    """
    gates = torch.nn.functional.gelu(x[..., [0]], approximate=approximate)
    return x * gates
