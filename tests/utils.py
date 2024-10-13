from __future__ import annotations

import torch
import numpy as np
import clifford as cf
from clifford import pga


def mv_to_tensor(
    mvs: list[cf.MultiVector], batch_dims: tuple[int] | None = None
) -> torch.Tensor:
    r"""Convert a list of multi-vectors to a batched tensor.

    Parameters
    ----------
    mvs : list[cf.MultiVector]
        List of clifford PGA multi-vectors.
    batch_dims: tuple[int] | None
        The dimensions of the batch. Note that this is a general 'batch',
        i.e., different objects within a sequence or different channels
        are also considered batch dimension. If None, a single multi-vector
        is generated.

    Returns
    -------
    torch.Tensor with shape (*batch_dims, 16)
        Batched tensor of multi-vectors.
    """
    ret = (
        torch.from_numpy(np.array([mv.value for mv in mvs]))
            .to(torch.float32)
            .reshape(*(batch_dims or (1,)), 16)
    )
    return ret


def make_random_clifford_mvs(
    batch_dims: tuple[int] | None = None, rng: int | None = None
) -> list[cf.MultiVector]:
    r"""Generate batches of random multi-vectors for testing.

    Parameters
    ----------
    batch_dims : tuple[int] | None
        The dimensions of the batch. Note that this is a general 'batch',
        i.e., different objects within a sequence or different channels
        are also considered batch dimension. If None, a single multi-vector
        is generated.
    rng : int | None
        Random seed for reproducibility.

    Returns
    -------
    list[cf.MultiVector]
        List of random multi-vectors.
    """
    batch_dims = batch_dims or (1,)
    ret = cf.randomMV(
        layout=pga.layout, n=np.prod(batch_dims), rng=rng
    )
    if isinstance(ret, cf.MultiVector):
        return [ret]
    return ret
