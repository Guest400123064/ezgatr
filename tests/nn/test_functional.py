from __future__ import annotations

import pytest
import torch

from ezgatr.nn.functional import (
    compute_qk_for_daa,
    compute_qk_for_ipa,
    equi_dual,
    equi_geometric_attention,
    equi_join,
    equi_linear,
    equi_rms_norm,
    geometric_product,
    inner_product,
    outer_product,
    scaler_gated_gelu,
)
from tests.utils import make_random_clifford_mvs, mv_to_tensor


class TestRegression:
    """Outputs from different implementations should align.

    This collection of tests primarily ensures that the outputs from the
    ``ezgatr`` are consistent internally and externally. For instance,
    the operators leveraging kernels for computational efficiency should
    agree with their slower, explicit counter parts. Or outputs of basic
    geometric algebra operations should agree with the implementation of
    ``clifford`` library.
    """


class TestPinEquivariance:
    """Certain operators should be pin-equivariant."""


def test_geometric_product():
    batch_dims = (3, 4, 5)
    rng = 42

    xs = make_random_clifford_mvs(batch_dims, rng)
    ys = make_random_clifford_mvs(batch_dims, rng)
    ts = mv_to_tensor([(x * y) for x, y in zip(xs, ys)], batch_dims)

    ts_torch = geometric_product(
        mv_to_tensor(xs, batch_dims), mv_to_tensor(ys, batch_dims)
    )
    assert torch.allclose(ts, ts_torch, rtol=1e-3)
