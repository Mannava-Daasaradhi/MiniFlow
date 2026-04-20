# tests/conftest.py

import pytest
from miniflow.db import DBUtils

@pytest.fixture
def tmp_db(tmp_path):
    """
    Creates an isolated, temporary SQLite database for a single test.
    tmp_path is a built-in pytest fixture providing a unique temporary directory.
    """
    db_path = str(tmp_path / "test.db")
    DBUtils.init_db(db_path)
    return db_path