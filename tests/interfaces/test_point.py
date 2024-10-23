from __future__ import annotations

import torch
from hypothesis import given
from hypothesis.strategies import floats

from ezgatr.interfaces.point import encode_pga, decode_pga
from tests.utils import make_random_3d_vectors


def test_loss_less_encode_decode():
    pcs = make_random_3d_vectors((8, 1024))
    torch.testing.assert_close(pcs, decode_pga(encode_pga(pcs)))


@given(
    scale=floats(
        min_value=1e-4,
        max_value=1e4,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_mv_scale_equivariant(scale: float):
    """Test ``pts = Decode(a * Encode(pts))``.

    The multi-vector representation should be scale equivariant because
    of the existence of the homogeneous coordinate.
    """
    pcs = make_random_3d_vectors((8, 1024))
    enc = encode_pga(pcs)

    torch.testing.assert_close(pcs, decode_pga(enc * scale))
    torch.testing.assert_close(pcs, decode_pga(enc * scale * -1.0))
