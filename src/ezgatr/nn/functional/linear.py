import functools

import torch


@functools.lru_cache(maxsize=None, typed=True)
def _compute_pin_equi_linear_basis(
    device: torch.device, dtype: torch.dtype, normalize: bool = True
) -> torch.Tensor:
    """Constructs basis elements for Pin(3, 0, 1)-equivariant linear maps
    between multi-vectors.

    Parameters
    ----------
    device : torch.device
        Device for the basis.
    dtype : torch.dtype
        Data type for the basis.
    normalize : bool
        Whether to normalize the basis elements according to
        the number of "inflow paths".

    Returns
    -------
    torch.Tensor
        Basis with shape (9, 16, 16) for equivariant linear maps.
    """
    basis_elements = [
        [0],
        [1, 2, 3, 4],
        [5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14],
        [15],
        [(1, 0)],
        [(5, 2), (6, 3), (7, 4)],
        [(11, 8), (12, 9), (13, 10)],
        [(15, 14)],
    ]

    basis = []
    for elements in basis_elements:
        w = torch.zeros((16, 16))
        for element in elements:
            try:
                i, j = element
                w[i, j] = 1.0
            except TypeError:
                w[element, element] = 1.0

        if normalize:
            w /= torch.linalg.norm(w)

        w = w.unsqueeze(0)
        basis.append(w)

    return torch.cat(basis, dim=0).to(device, dtype)


def equi_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, normalize_basis: bool = True
) -> torch.Tensor:
    """Perform Pin-equivariant linear map defined by weight on input x.

    One way to think of the equivariant linear map is a channel-wise
    "map-reduce", where the same weight (of one neuron) are applied to
    all channels of the input multi-vector and the results are summed up
    along the basis/blade axis. In other words, the map is a channel-mixing
    operation. Using a parallel analogy with a regular ``nn.Linear`` layer,
    each channel of a input multi-vector corresponds to a "feature value"
    of a simple hidden representation, and the number of output channels
    is the number of neurons in the hidden layer.

    Within each channel, similar to the geometric product implementation,
    the linear operation starts with a matrix multiplication between the
    (weighted, if normalized) 16-by-16 computation graph, i.e., the basis,
    and the input multi-vector to take into account the effect that blades
    containing ``e_0`` will be "downgraded" to a lower grade after the map.
    Again, a map from "source-to-destination" style. Note that we may want
    to optimize this operation as the basis is pretty sparse.

    Parameters
    ----------
    x : torch.Tensor with shape (..., in_channels, 16)
        Input multivector. Batch dimensions must be broadcastable between
        x and weight.
    weight : torch.Tensor with shape (out_channels, in_channels, 9)
        Coefficients for the 9 basis elements. Batch dimensions must be
        broadcastable between x and weight.
    bias : torch.Tensor with shape (out_channels,)
        Bias for the linear map. The bias values are only added to the
        scalar blades (i.e., index position 0) of each output channel.
    normalize_basis : bool
        Whether to normalize the basis elements according to
        the number of "inflow paths" of each blade.

    Returns
    -------
    torch.Tensor
        Output with shape (..., out_channels, 16).
    """
    basis = _compute_pin_equi_linear_basis(x.device, x.dtype, normalize_basis)

    x = torch.einsum("oiw, wds, ...is -> ...od", weight, basis, x)
    if bias is not None:
        x[..., [0]] += bias.view(-1, 1)
    return x
