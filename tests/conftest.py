import pytest

from wagentdb.config import Settings
from wagentdb.store import Store


@pytest.fixture
def settings(tmp_path):
    return Settings(backend="local", local_dir=tmp_path / "wdb")


@pytest.fixture
def store(settings):
    s = Store(settings)
    yield s
    s.close()
