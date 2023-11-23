import pandas as pd

from cfactor import util


def test_celc_to_fahr():
    """Test the conversion from degrees celcius ot fahrenheit"""
    # Typical case float
    temperature_celcius = 24
    expected_temperature = 75.2

    temperature_fahrenheit = util.celc_to_fahr(temperature_celcius)
    assert expected_temperature == temperature_fahrenheit

    # Test with pandas series
    temperature_celcius_series = pd.Series(data=[0, 10, 24, -10])
    expected_temperature_series = pd.Series(data=[32.0, 50.0, 75.2, 14.0])

    temperature_fahrenheit_series = util.celc_to_fahr(temperature_celcius_series)
    pd.testing.assert_series_equal(
        expected_temperature_series, temperature_fahrenheit_series
    )
