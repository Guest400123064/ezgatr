import torch
import torch.nn as nn

from ezgatr.nn import EquiLinear, EquiRMSNorm
from ezgatr.nn.functional.dual import equi_join
from ezgatr.nn.functional.linear import geometric_product
from ezgatr.nn.functional.activation import scaler_gated_gelu


class GeometricBilinear(nn.Module):
    """Implements the geometric bilinear sub-layer of the geometric MLP.

    Geometric bilinear operation consists of geometric product and equivariant
    join operations. The results of two operations are concatenated along the
    hidden channel axis and passed through a final equivariant linear projection
    before being passed to the next layer, block, or module.

    In both geometric product and equivariant join operations, the input multi-vectors
    are first projected to a hidden space with the same number of channels, i.e., left
    and right. Then, the results of each operation are derived from the interaction of
    left and right hidden representations, each with half number of ``hidden_channels``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels. This refers to the hidden size throughout
        the network, i.e., the input number of channels of the next layer,
        block, or module.
    hidden_channels : int
        Number of hidden channels. Must be even. This hidden size should not
        be confused with the size of hidden representations throughout the
        network. It only refers to the hidden sizes used for the equivariant
        join and geometric product operations.
    """

    __constants__ = ["in_channels", "out_channels", "hidden_channels"]

    in_channels: int
    out_channels: int
    hidden_channels: int
    proj_bilinear: EquiLinear
    proj_to_next: EquiLinear

    def __init__(
        self, in_channels: int, out_channels: int, hidden_channels: int
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        if hidden_channels % 2 != 0:
            raise ValueError("Hidden channels must be even.")

        self.proj_bilinear = EquiLinear(in_channels, hidden_channels * 2)
        self.proj_to_next = EquiLinear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        l_geo, r_geo, l_join, r_join = torch.split(
            self.proj_bilinear(x), self.hidden_channels // 2, dim=-2
        )
        x = torch.cat(
            [geometric_product(l_geo, r_geo), equi_join(l_join, r_join, reference)],
            dim=-2,
        )
        return self.proj_to_next(x)


class GeometricMLP(nn.Module):
    """Implements Geometric MLP block described in the GATr paper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        channelwise_rescale: bool = True,
    ) -> None:
        super().__init__()

        self.layer_norm = EquiRMSNorm(
            in_channels,
            channelwise_rescale=channelwise_rescale,
        )
        self.geo_bilinear = GeometricBilinear(
            in_channels, out_channels, hidden_channels
        )
        self.proj_to_next = EquiLinear(out_channels, out_channels)

    def forward(self, x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        x_res = x

        x = self.layer_norm(x)
        x = self.geo_bilinear(x, reference)
        x = self.proj_to_next(scaler_gated_gelu(x))

        return x + x_res
