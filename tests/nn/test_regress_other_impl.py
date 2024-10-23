"""Outputs from different implementations should align.

This collection of tests primarily ensures that the outputs from the
``ezgatr`` are consistent internally and externally. For instance,
the operators leveraging kernels for computational efficiency should
agree with their slower, explicit counter parts. Or outputs of basic
geometric algebra operations should agree with the implementation of
``clifford`` library.
"""

from __future__ import annotations

import torch
import numpy as np
from hypothesis import given, settings

from ezgatr.nn.functional import (
    equi_dual,
    equi_join,
    geometric_product,
    outer_product,
)
from tests.utils import (
    make_random_pga_mvs,
    mv_to_tensor,
    strategy_batch_dims,
)


@given(batch_dims=strategy_batch_dims(max_size=8))
@settings(deadline=None)
def test_vary_shape_geometric_product(batch_dims):
    xs = make_random_pga_mvs(batch_dims)
    ys = make_random_pga_mvs(batch_dims)
    ps = geometric_product(
        mv_to_tensor(xs, batch_dims), mv_to_tensor(ys, batch_dims)
    )
    torch.testing.assert_close(
        ps,
        mv_to_tensor([x * y for x, y in zip(xs, ys)], batch_dims),
        rtol=1e-3,
        atol=1e-3,
    )


@given(batch_dims=strategy_batch_dims(max_size=8))
@settings(deadline=None)
def test_vary_shape_outer_product(batch_dims):
    xs = make_random_pga_mvs(batch_dims)
    ys = make_random_pga_mvs(batch_dims)
    ps = outer_product(
        mv_to_tensor(xs, batch_dims), mv_to_tensor(ys, batch_dims)
    )
    torch.testing.assert_close(
        ps,
        mv_to_tensor([x ^ y for x, y in zip(xs, ys)], batch_dims),
        rtol=1e-3,
        atol=1e-3,
    )
