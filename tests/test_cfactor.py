import numpy as np
import pytest
from conftest import load_calculated_dummy_data

from cfactor import cfactor


def test_compute_crop_cover():
    """Test calculation of crop cover"""

    # Typical case
    h = 0.49
    fc = 0.77
    expected_cc = 0.34432154097916456

    cc = cfactor.compute_crop_cover(h, fc)
    assert cc == expected_cc

    # Out of bound value h
    h = -0.49

    with pytest.raises(ValueError) as excinfo:
        cc = cfactor.compute_crop_cover(h, fc)
    assert ("Effective drop height cannot be negative") in str(excinfo.value)

    # Out of bound value fc - case 1
    h = 0.49
    fc = 1.2

    with pytest.raises(ValueError) as excinfo:
        cc = cfactor.compute_crop_cover(h, fc)
    assert ("Soil cover must be between 0 and 1") in str(excinfo.value)

    # Out of bound value fc - case 2
    fc = -9

    with pytest.raises(ValueError) as excinfo:
        cc = cfactor.compute_crop_cover(h, fc)
    assert ("Soil cover must be between 0 and 1") in str(excinfo.value)

    # Typical case numpy
    df_dummy = load_calculated_dummy_data()
    arr_h = df_dummy["h"].to_numpy()
    arr_fc = df_dummy["fc"].to_numpy()
    arr_expected_cc = df_dummy["CC"].to_numpy()
    arr_cc = cfactor.compute_crop_cover(arr_h, arr_fc)
    np.testing.assert_allclose(arr_cc, arr_expected_cc)


def test_compute_soil_roughness():
    """Test calculation of soil roughness"""
    # Typical case float
    ri = 10.2
    rain = 35.41
    rhm = 28.47
    expected_ru = 9.781252579741903
    expected_f1_n = -0.1951732283464567
    expected_f2_ei = -0.020072855464159812
    ru, f1_n, f2_ei = cfactor.compute_soil_roughness(ri, rain, rhm)
    assert (expected_ru, expected_f1_n, expected_f2_ei) == (ru, f1_n, f2_ei)

    rain = -10
    with pytest.raises(ValueError) as excinfo:
        cfactor.compute_soil_roughness(ri, rain, rhm)
    assert ("Amount of rain cannot be negative") in str(excinfo.value)

    # Typical case np.array
    df_dummy = load_calculated_dummy_data()
    arr_ri = df_dummy["Ri"].to_numpy()
    arr_rain = df_dummy["rain"].to_numpy()
    arr_rhm = df_dummy["Rhm"].to_numpy()
    arr_expected_ru = df_dummy["Ru"].to_numpy()
    arr_expected_f1_n = df_dummy["f1_N"].to_numpy()
    arr_expected_f2_ei = df_dummy["f2_EI"].to_numpy()
    expected_result = (arr_expected_ru, arr_expected_f1_n, arr_expected_f2_ei)
    calculated_result = cfactor.compute_soil_roughness(arr_ri, arr_rain, arr_rhm)
    np.testing.assert_allclose(expected_result, calculated_result)


def test_compute_surface_roughness():
    """Test calculation of surface roughness"""
    # Typical case - float
    ru = 8.918848956028448
    expected_SR = 0.9292345717420408
    SR = cfactor.compute_surface_roughness(ru)
    assert SR == expected_SR

    # Typical case - np.ndaraay
    df_dummy = load_calculated_dummy_data()
    arr_ru = df_dummy["Ru"].to_numpy()
    arr_expected_sr = df_dummy["SR"].to_numpy()
    arr_sr = cfactor.compute_surface_roughness(arr_ru)
    np.testing.assert_allclose(arr_expected_sr, arr_sr)


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
    arr_start_dates = df_dummy["bdate"].to_numpy()
    arr_end_dates = df_dummy["edate"].to_numpy()
    arr_expected_days = df_dummy["D"].to_numpy()
    arr_calculated_days = cfactor.calculate_number_of_days(
        arr_start_dates, arr_end_dates
    )
    np.testing.assert_allclose(arr_expected_days, arr_calculated_days)


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
    arr_days = df_dummy["D"].to_numpy()
    arr_a = df_dummy["a"].to_numpy()
    initial_crop_residu = 5000
    arr_expected_residu_start = df_dummy["Bsi"].to_numpy()
    arr_expected_residu_end = df_dummy["Bse"].to_numpy()
    expected_result = (arr_expected_residu_start, arr_expected_residu_end)
    calculated_result = cfactor.compute_crop_residu_timeseries(
        arr_days, arr_a, initial_crop_residu
    )
    np.testing.assert_allclose(expected_result, calculated_result)


def test_compute_harvest_residu_decay_rate():
    """Test calculation of harvest residu decay rate"""

    # Typical case - float
    rain = 73.56
    temperature = 4.5
    p = 0.05
    expected_w = 2.8555900621118013
    expected_f = 0.583248438971586
    expected_a = 0.0291624219485793

    result = cfactor.compute_harvest_residu_decay_rate(rain, temperature, p)
    assert result == (expected_w, expected_f, expected_a)

    # Out of bound value rain
    rain = -10

    with pytest.raises(ValueError) as excinfo:
        result = cfactor.compute_harvest_residu_decay_rate(rain, temperature, p)
    assert ("Halfmonthly rainfall cannot be negative") in str(excinfo.value)

    # Typical case - np.ndarray
    df_dummy = load_calculated_dummy_data()
    arr_rain = df_dummy["rain"].to_numpy()
    arr_temperature = df_dummy["temp"].to_numpy()
    arr_p = df_dummy["p"].to_numpy()
    arr_expected_w = df_dummy["W"].to_numpy()
    arr_expected_f = df_dummy["F"].to_numpy()
    arr_expected_a = df_dummy["a"].to_numpy()
    result = cfactor.compute_harvest_residu_decay_rate(arr_rain, arr_temperature, arr_p)
    np.testing.assert_allclose(result, (arr_expected_w, arr_expected_f, arr_expected_a))


def test_compute_soil_cover():
    """Test calculation of soil cover"""
    # Typical case float
    crop_residu = 5000
    alpha = 5.53
    ru = 6.096
    expected_sp = 93.7023900651985
    expected_sc = 0.037643926507827864
    sp, sc = cfactor.compute_soil_cover(crop_residu, alpha, ru)
    assert (expected_sp, expected_sc) == (sp, sc)

    # Typical case np.array
    df_dummy = load_calculated_dummy_data()
    arr_crop_residu = df_dummy["Bse"].to_numpy()
    arr_alpha = df_dummy["alpha"].to_numpy()
    arr_ru = df_dummy["Ru"].to_numpy()
    arr_expected_sp = df_dummy["Sp"].to_numpy()
    arr_expected_sc = df_dummy["SC"].to_numpy()
    expected_result = (arr_expected_sp, arr_expected_sc)
    calculated_result = cfactor.compute_soil_cover(arr_crop_residu, arr_alpha, arr_ru)
    np.testing.assert_allclose(expected_result, calculated_result)


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
    arr_sc = df_dummy["SC"].to_numpy()
    arr_sr = df_dummy["SR"].to_numpy()
    arr_cc = df_dummy["CC"].to_numpy()
    arr_expected_slr = df_dummy["soil_loss_ratio"].to_numpy()
    arr_slr = cfactor.compute_soil_loss_ratio(arr_sc, arr_sr, arr_cc)
    np.testing.assert_allclose(arr_expected_slr, arr_slr)

    # Test error handling float
    sc = 0.03768204712884102
    sr = -10
    cc = 0.99
    with pytest.raises(ValueError) as excinfo:
        slr = cfactor.compute_soil_loss_ratio(sc, sr, cc)
    assert ("All soil_loss_ratio subfactors must lie between 0 and 1") in str(
        excinfo.value
    )

    # Test error handling np.ndarray
    arr_sc = np.array([1, 0.03768204712884102, 1])
    arr_sr = np.array([0.9999671539111016, -10, 0.9999037924066257])
    arr_cc = np.array([1, 1, 0.99])
    with pytest.raises(ValueError) as excinfo:
        arr_slr = cfactor.compute_soil_loss_ratio(arr_sc, arr_sr, arr_cc)
    assert ("All soil_loss_ratio subfactors must lie between 0 and 1") in str(
        excinfo.value
    )


@pytest.mark.skip(reason="not yet implemented")
def test_aggregate_slr_to_c_factor():
    """Test aggregation of soil_loss_ratio to C-factor"""
    # TO DO
    # cfactor.aggregate_slr_to_c_factor()
