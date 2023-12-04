import numpy as np
import pytest
from conftest import load_calculated_dummy_data

from cfactor import cfactor


def test_compute_crop_cover():
    """Test calculation of crop cover"""

    # Typical case
    H = 0.49
    Fc = 0.77
    expected_cc = 0.34432154097916456

    cc = cfactor.compute_crop_cover(H, Fc)
    assert cc == expected_cc

    # Out of bound value H
    H = -0.49
    Fc = 0.77

    with pytest.raises(ValueError) as excinfo:
        cc = cfactor.compute_crop_cover(H, Fc)
    assert ("Effective drop height cannot be negative") in str(excinfo.value)

    # Out of bound value Fc - case 1
    H = 0.49
    Fc = 1.2

    with pytest.raises(ValueError) as excinfo:
        cc = cfactor.compute_crop_cover(H, Fc)
    assert ("Soil cover must be between 0 and 1") in str(excinfo.value)

    # Out of bound value Fc - case 2
    H = 0.49
    Fc = -9

    with pytest.raises(ValueError) as excinfo:
        cc = cfactor.compute_crop_cover(H, Fc)
    assert ("Soil cover must be between 0 and 1") in str(excinfo.value)

    # Typical case numpy
    df_dummy = load_calculated_dummy_data()
    H = df_dummy["H"].to_numpy()
    Fc = df_dummy["Fc"].to_numpy()
    expected_cc = df_dummy["CC"].to_numpy()
    cc = cfactor.compute_crop_cover(H, Fc)
    np.testing.assert_allclose(cc, expected_cc)


@pytest.mark.skip(reason="not yet implemented")
def test_compute_soil_roughness():
    """Test calculation of soil roughness"""
    # TO DO
    # cfactor.compute_soil_roughness()


def test_compute_surface_roughness():
    """Test calculation of surface roughness"""
    # Typical case - float
    ru = 8.918848956028448
    expected_SR = 0.9292345717420408
    SR = cfactor.compute_surface_roughness(ru)
    assert SR == expected_SR

    # Typical case - np.ndaraay
    df_dummy = load_calculated_dummy_data()
    ru = df_dummy["Ru"].to_numpy()
    expected_sr = df_dummy["SR"].to_numpy()
    sr = cfactor.compute_surface_roughness(ru)
    np.testing.assert_allclose(expected_sr, sr)


def test_calculate_number_of_days():
    """Test calculation of number of days"""
    # Typical case - string
    start_date = "2016-02-15"
    end_date = "2016-03-01"
    expected_days = 15
    calculated_days = cfactor.calculate_number_of_days(start_date, end_date)
    assert expected_days == calculated_days

    # Typical case - np.array
    df_dummy = load_calculated_dummy_data()
    start_dates = df_dummy["bdate"].to_numpy()
    end_dates = df_dummy["edate"].to_numpy()
    expected_days = df_dummy["D"].to_numpy()
    calculated_days = cfactor.calculate_number_of_days(start_dates, end_dates)
    np.testing.assert_allclose(expected_days, calculated_days)


def test_compute_crop_residu():
    """Test calculation of crop residu"""
    # Typical case
    days = 15
    initial_crop_residu = 5000
    a = 0.02518464958645108
    expected_residu = 3426.9414870271776
    calculated_residu = cfactor.compute_crop_residu(days, a, initial_crop_residu)
    assert expected_residu == calculated_residu


def test_compute_crop_residu_timeseries():
    """Test calculation of crop residu on timeseries"""
    # Typical case
    df_dummy = load_calculated_dummy_data()
    days = df_dummy["D"].to_numpy()
    a = df_dummy["a"].to_numpy()
    initial_crop_residu = 5000
    expected_residu_start = df_dummy["Bsi"].to_numpy()
    expected_residu_end = df_dummy["Bse"].to_numpy()
    expected_result = (expected_residu_start, expected_residu_end)
    calculated_result = cfactor.compute_crop_residu_timeseries(
        days, a, initial_crop_residu
    )
    np.testing.assert_allclose(expected_result, calculated_result)


def test_compute_harvest_residu_decay_rate():
    """Test calculation of harvest residu decay rate"""

    # Typical case - float
    rain = 73.56
    temperature = 4.5
    p = 0.05
    expected_W = 2.8555900621118013
    expected_F = 0.583248438971586
    expected_a = 0.0291624219485793

    result = cfactor.compute_harvest_residu_decay_rate(rain, temperature, p)
    assert result == (expected_W, expected_F, expected_a)

    # Out of bound value rain
    rain = -10

    with pytest.raises(ValueError) as excinfo:
        result = cfactor.compute_harvest_residu_decay_rate(rain, temperature, p)
    assert ("Halfmonthly rainfall cannot be negative") in str(excinfo.value)

    # Typical case - np.ndarray
    df_dummy = load_calculated_dummy_data()
    rain = df_dummy["rain"].to_numpy()
    temperature = df_dummy["temp"].to_numpy()
    p = df_dummy["p"].to_numpy()
    expected_W = df_dummy["W"].to_numpy()
    expected_F = df_dummy["F"].to_numpy()
    expected_a = df_dummy["a"].to_numpy()
    result = cfactor.compute_harvest_residu_decay_rate(rain, temperature, p)
    np.testing.assert_allclose(result, (expected_W, expected_F, expected_a))


@pytest.mark.skip(reason="not yet implemented")
def test_compute_soil_cover():
    """Test calculation of soil cover"""
    # TO DO
    # cfactor.compute_soil_cover()


def test_compute_soil_loss_ratio():
    """Test calculation of soil loss ration"""
    # Typical case float
    sc = 0.03768204712884102
    sr = 0.9999092973948398
    cc = 1
    expected_slr = 0.03767862926899866
    slr = cfactor.compute_soil_loss_ratio(sc, sr, cc)
    assert expected_slr == slr

    # Typical case np.ndarray
    df_dummy = load_calculated_dummy_data()
    sc = df_dummy["SC"].to_numpy()
    sr = df_dummy["SR"].to_numpy()
    cc = df_dummy["CC"].to_numpy()
    expected_slr = df_dummy["SLR"].to_numpy()
    slr = cfactor.compute_soil_loss_ratio(sc, sr, cc)
    np.testing.assert_allclose(expected_slr, slr)

    # Test error handling float
    sc = 0.03768204712884102
    sr = -10
    cc = 0.99
    with pytest.raises(ValueError) as excinfo:
        slr = cfactor.compute_soil_loss_ratio(sc, sr, cc)
    assert ("All SLR subfactors must lie between 0 and 1") in str(excinfo.value)

    # Test error handling np.ndarray
    sc = np.array([1, 0.03768204712884102, 1])
    sr = np.array([0.9999671539111016, -10, 0.9999037924066257])
    cc = np.array([1, 1, 0.99])
    with pytest.raises(ValueError) as excinfo:
        slr = cfactor.compute_soil_loss_ratio(sc, sr, cc)
    assert ("All SLR subfactors must lie between 0 and 1") in str(excinfo.value)


@pytest.mark.skip(reason="not yet implemented")
def test_aggregate_slr_to_crop_factor():
    """Test aggregation of SLR to C-factor"""
    # TO DO
    # cfactor.aggregate_slr_to_c_factor()
