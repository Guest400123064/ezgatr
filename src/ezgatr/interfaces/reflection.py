import torch

from ezgatr.interfaces import plane


def encode_pga(normals: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return plane.encode_pga(normals, positions)


def decode_pga(mvs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return plane.decode_pga(mvs)
