import functools
import torch


@functools.lru_cache(maxsize=None)
def _compute_pin_equi_linear_basis(
    device: torch.device, dtype: torch.dtype, normalize: bool = True
) -> torch.Tensor:
    """Constructs basis elements for Pin(3,0,1)-equivariant linear maps between multi-vectors.

    This function is cached.

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
    basis : torch.Tensor with shape (9, 16, 16)
        Basis elements for equivariant linear maps.
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


def equi_linear(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant linear map f(x) = sum_{a,j} weights_a W^a_ij x_j.

    The W^a are 9 pre-defined basis elements.

    Parameters
    ----------
    x : torch.Tensor with shape (..., in_channels, 16)
        Input multivector. Batch dimensions must be broadcastable between x and weights.
    weights : torch.Tensor with shape (out_channels, in_channels, 9)
        Coefficients for the 9 basis elements. Batch dimensions must be broadcastable between x and weights.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result. Batch dimensions are result of broadcasting between x and weights.
    """
    basis = _compute_pin_equi_linear_basis(x.device, x.dtype)
    return torch.einsum("yxa, aij, ...xj -> ...yi", weights, basis, x)
