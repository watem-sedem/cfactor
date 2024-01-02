import numpy as np

from cfactor import util


def test_celc_to_fahr_float():
    """Test the conversion from degrees celcius ot fahrenheit using floats"""
    # Typical case float
    temperature_celcius = 24
    expected_temperature = 75.2

    temperature_fahrenheit = util.celc_to_fahr(temperature_celcius)
    assert expected_temperature == temperature_fahrenheit


def test_celc_to_fahr_numpy():
    """Test the conversion from degrees celcius ot fahrenheit using numpy arrays"""
    temperature_celcius_series = np.array([0, 10, 24, -10])

    expected_temperature_series = np.array([32.0, 50.0, 75.2, 14.0])

    temperature_fahrenheit_series = util.celc_to_fahr(temperature_celcius_series)
    np.testing.assert_allclose(
        expected_temperature_series, temperature_fahrenheit_series
    )
