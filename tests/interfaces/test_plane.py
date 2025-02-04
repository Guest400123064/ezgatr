from __future__ import annotations

import torch
from hypothesis import given, settings

from ezgatr.interfaces.plane import encode_pga
from ezgatr.nn.functional import geometric_product, inner_product
from tests.utils import make_random_3d_vectors, strategy_batch_dims


@given(batch_dims=strategy_batch_dims(max_size=8))
@settings(deadline=None)
def test_pga_plane_inner_eq_sq_norm(batch_dims):
    r"""Test inner of plane encoded in PGA is equivalent to squared Euclidean norm.

    The inner product of a plane with itself should be equivalent to the squared
    norm of the normal vector in the Euclidean space.
    """
    normals = make_random_3d_vectors(batch_dims)
    translations = make_random_3d_vectors(batch_dims)
    norms = (torch.norm(normals, dim=-1) ** 2).squeeze()
    enc = encode_pga(normals, translations)

    torch.testing.assert_close(
        inner_product(enc, enc).squeeze(),
        norms,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        geometric_product(enc, enc).sum(-1).squeeze(),
        norms,
        rtol=1e-4,
        atol=1e-4,
    )
