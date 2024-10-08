import torch


def encode_pga() -> torch.Tensor:
    pass


def decode_pga(_: torch.Tensor) -> torch.Tensor:
    msg = "Translation interface is not equivariant and decode is not used."
    raise NotImplementedError(msg)
