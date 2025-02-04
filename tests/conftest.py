import pathlib

import pytest
import toml

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
