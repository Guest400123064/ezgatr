import torch

from ezgatr.nn.functional import geometric_product
from tests.helpers import make_random_clifford_mvs, mv_to_tensor


def test_geometric_product():
    batch_dims = (3, 4, 5)
    rng = 42

    xs = make_random_clifford_mvs(batch_dims, rng)
    ys = make_random_clifford_mvs(batch_dims, rng)
    ts = mv_to_tensor([(x * y) for x, y in zip(xs, ys)], batch_dims)

    ts_torch = geometric_product(
        mv_to_tensor(xs, batch_dims), mv_to_tensor(ys, batch_dims)
    )
    assert torch.allclose(ts, ts_torch, rtol=1e-4)
