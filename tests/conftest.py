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

CURRENT_DIR = Path(os.path.dirname(__file__))


def load_calculated_dummy_data():
    """Calculated example subfactors"""
    txt_dummy_data = CURRENT_DIR / "data" / "dummy_calculations.csv"
    df_dummy_data = pd.read_csv(txt_dummy_data)
    return df_dummy_data
