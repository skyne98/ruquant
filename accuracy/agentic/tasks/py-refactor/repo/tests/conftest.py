import pytest

from library import db


@pytest.fixture(autouse=True)
def fresh_db():
    db.reset()
    yield
    db.reset()
