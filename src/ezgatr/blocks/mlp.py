import torch
import torch.nn as nn

from ezgatr.nn import EquiLinear
from ezgatr.nn.functional.dual import equi_join
from ezgatr.nn.functional.linear import geometric_product


class GeometricBilinear(nn.Module):
    """ """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
    ) -> None:
        pass
