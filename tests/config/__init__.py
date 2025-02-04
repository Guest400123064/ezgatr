from dataclasses import dataclass


@dataclass
class BasicConfig:
    r"""Basic configurations shared across all tests.

    Parameters
    ----------
    rtol : float
        Relative tolerance for ``torch.testing.assert_close``.
    atol : float
        Absolute tolerance for ``torch.testing.assert_close``.
    equal_nan : bool
        If ``True``, then two ``NaN`` s will be considered equal.
    check_device : bool
        If ``True``, check that ``input`` and ``other`` are on the same device.
    check_dtype : bool
        If ``True``, check that ``input`` and ``other`` have the same dtype.
    device : str
        The device on which tests are executed (e.g., 'cpu', 'cuda').
    max_batch_size : int
        Maximum number of sequences within a batch of multi-vectors when
        generating random testing data points.
    max_context_size : int
        Maximum number of multi-vectors within a single sequence when generating
        random testing data points.
    max_channel_size : int
        Maximum number of channels for each multi-vector within a sequence
        when generating random testing data points.
    """
    rtol: float
    atol: float
    equal_nan: bool
    check_device: bool
    check_dtype: bool
    device: str
    max_batch_size: int
    max_context_size: int
    max_channel_size: int
