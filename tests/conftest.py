import pathlib
import pytest
import toml

from tests.helpers import generate_random_data
from tests.config import BasicConfig

DIR_TEST = pathlib.Path(".").resolve().parent


@pytest.fixture(scope="session")
def basic_config():
    r"""Load basic configurations for tests.

    This fixture reads the ``core.toml`` file and returns an instance of the `BasicConfig` dataclass
    containing the configuration parameters.

    Returns
    -------
    BasicConfig
        An instance containing basic test configurations.
    """
    with open(DIR_TEST / "config" / "core.toml") as f:
        config_data = toml.load(f)
    return BasicConfig(**config_data)


@pytest.fixture
def random_data(basic_config):
    r"""Generate random data for tests using the basic configuration.

    This fixture uses `generate_random_data` to create random data points based on the parameters
    specified in `basic_config`.

    Parameters
    ----------
    basic_config : BasicConfig
        Configuration object containing parameters for generating random data.

    Returns
    -------
    torch.Tensor
        Generated random data of shape (batch_size, num_points, input_channels, 16).
    """
    return generate_random_data(basic_config)
