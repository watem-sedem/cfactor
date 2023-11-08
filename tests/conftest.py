"""
    Dummy conftest.py for cfactor.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""
import os
from pathlib import Path

import pandas as pd
import pytest

CURRENT_DIR = Path(os.path.dirname(__file__))


@pytest.fixture()
def load_calculated_dummy_data():
    """Calculated example subfactors"""
    f = CURRENT_DIR / "data" / "dummy_calculations.csv"
    df = pd.read_csv(f)
    return df
