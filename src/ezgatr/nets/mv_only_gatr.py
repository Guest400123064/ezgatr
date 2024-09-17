from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ezgatr.nn import EquiLinear, EquiRMSNorm
from ezgatr.nn.functional.dual import equi_join
from ezgatr.nn.functional.linear import geometric_product
from ezgatr.nn.functional.activation import scaler_gated_gelu


@dataclass
class ModelConfig:
    """Configuration class for the ``MVOnlyGATr`` model.

    Parameters
    ----------
    size_context : int, default to 2048
        Number of elements, e.g., number of points in a point cloud,
        in the input sequence.
    size_channels_in : int, default to 1
        Number of input channels.
    size_channels_out : int, default to 1
        Number of output channels.
    size_channels_hidden : int, default to 32
        Number of hidden representation channels throughout the network, i.e.,
        the input/output number of channels of the next layer, block, or module.
    size_channels_intermediate : int, default to 32
        Number of intermediate channels for the geometric bilinear operation.
        Must be even. This intermediate size should not be confused with the size
        of hidden representations throughout the network. It only refers to the
        hidden sizes used for the equivariant join and geometric product operations.
    norm_eps : Optional[float], default to None
        Small value to prevent division by zero in the normalization layer.
    norm_channelwise_rescale : bool, default to True
        Apply learnable channel-wise rescaling weights to the normalized multi-vector
        inputs. Initialized to ones if set to ``True``.
    """

    size_context: int = 2048

    size_channels_in: int = 1
    size_channels_out: int = 1
    size_channels_hidden: int = 32
    size_channels_intermediate: int = 32

    norm_eps: Optional[float] = None
    norm_channelwise_rescale: bool = True


class Bilinear(nn.Module):
    """Implements the geometric bilinear sub-layer of the geometric MLP.

    Geometric bilinear operation consists of geometric product and equivariant
    join operations. The results of two operations are concatenated along the
    hidden channel axis and passed through a final equivariant linear projection
    before being passed to the next layer, block, or module.

    In both geometric product and equivariant join operations, the input
    multi-vectors are first projected to a hidden space with the same number of
    channels, i.e., left and right. Then, the results of each operation are
    derived from the interaction of left and right hidden representations, each
    with half number of ``size_channels_intermediate``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config
        if config.size_channels_intermediate % 2 != 0:
            raise ValueError("Number of hidden channels must be even.")

        self.proj_bilinear = EquiLinear(
            config.size_channels_in, config.size_channels_intermediate * 2
        )
        self.proj_to_next = EquiLinear(
            config.size_channels_intermediate, config.size_channels_out
        )

    def forward(self, x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        l_geo, r_geo, l_join, r_join = torch.split(
            self.proj_bilinear(x), self.hidden_channels // 2, dim=-2
        )
        x = torch.cat(
            [geometric_product(l_geo, r_geo), equi_join(l_join, r_join, reference)],
            dim=-2,
        )
        return self.proj_to_next(x)


class MLP(nn.Module):
    """Geometric MLP block without scaler channels."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.layer_norm = EquiRMSNorm(
            config.size_channels_hidden,
            eps=config.norm_eps,
            channelwise_rescale=config.norm_channelwise_rescale,
        )
        self.geo_bilinear = Bilinear(
            config.size_channels_hidden,
            config.size_channels_hidden,
            config.size_channels_intermediate,
        )
        self.proj_to_next = EquiLinear(
            config.size_channels_hidden, config.size_channels_hidden
        )

    def forward(self, x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        x_res = x

        x = self.layer_norm(x)
        x = self.geo_bilinear(x, reference)
        x = self.proj_to_next(scaler_gated_gelu(x))

        return x + x_res


class Attention(nn.Module):
    """Geometric attention block with scaler channels."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = x

        return x + x_res


class TransformerBlock(nn.Module):
    pass


class Transformer(nn.Module):
    pass
