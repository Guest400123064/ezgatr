import math

import torch
import torch.nn as nn

from ezgatr.nn.functional.linear import equi_linear


class EquiLinear(nn.Module):
    """Pin(3, 0, 1)-equivariant linear map.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    normalize_basis : bool
        Whether to normalize the basis elements according to the number of
        "inflow paths" for each blade.
    """
    __constants__ = ["in_channels", "out_channels", "normalize_basis"]
    in_channels: int
    out_channels: int
    normalize_basis: bool
    weight: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize_basis: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize_basis = normalize_basis
        self.weight = nn.Parameter(
            torch.empty((out_channels, in_channels), **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return equi_linear(x, self.weight, self.normalize_basis)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"normalize_basis={self.normalize_basis}"
        )
