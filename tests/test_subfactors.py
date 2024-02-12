import numpy as np
import pytest
from conftest import load_calculated_dummy_data

from cfactor import subfactors


def test_compute_crop_cover_float():
    """Test expected result of calculation of crop cover using floats"""
    h = 0.49
    fc = 0.77
    expected_cc = 0.34432154097916456

    cc = subfactors.compute_crop_cover(h, fc)
    assert cc == expected_cc


def test_compute_crop_cover_float_negative_height():
    """Test error handling of crop cover using an out of bound drop height (float)"""

    h = -0.49
    fc = 0.77

    with pytest.raises(ValueError) as excinfo:
        subfactors.compute_crop_cover(h, fc)
    assert ("Effective drop height cannot be negative") in str(excinfo.value)


def test_compute_crop_cover_float_large_soil_cover():
    """Test error handling of crop cover
    using an out of bound soil cover (>1) (float)"""
    h = 0.49
    fc = 1.2

    with pytest.raises(ValueError) as excinfo:
        subfactors.compute_crop_cover(h, fc)
    assert ("Soil cover must be between 0 and 1") in str(excinfo.value)


def test_compute_crop_cover_float_negative_soil_cover():
    """Test error handling of crop cover
    using an out of bound soil cover (<0) (float)"""
    fc = -9
    h = 0.49

    with pytest.raises(ValueError) as excinfo:
        subfactors.compute_crop_cover(h, fc)
    assert ("Soil cover must be between 0 and 1") in str(excinfo.value)


def test_compute_crop_cover_numpy():
    """Test expected result of calculation of crop cover using numpy arrays"""
    df_dummy = load_calculated_dummy_data()
    arr_h = df_dummy["H"].to_numpy()
    arr_fc = df_dummy["Fc"].to_numpy()
    arr_expected_cc = df_dummy["CC"].to_numpy()
    arr_cc = subfactors.compute_crop_cover(arr_h, arr_fc)
    np.testing.assert_allclose(arr_cc, arr_expected_cc)


def test_compute_soil_roughness_float():
    """Test expected result of  calculation of soil roughness using floats"""

    ri = 10.2
    rain = 35.41
    rhm = 28.47
    expected_ru = 9.781252579741903
    expected_f1_n = -0.1951732283464567
    expected_f2_ei = -0.020072855464159812
    ru, f1_n, f2_ei = subfactors.compute_soil_roughness(ri, rain, rhm)
    assert (expected_ru, expected_f1_n, expected_f2_ei) == (ru, f1_n, f2_ei)


def test_compute_soil_roughness_float_negative_rain():
    """Test error handling of calculation of soil roughness with negative rain"""
    rain = -10
    ri = 10.2
    rhm = 28.47

    with pytest.raises(ValueError) as excinfo:
        subfactors.compute_soil_roughness(ri, rain, rhm)
    assert ("Amount of rain cannot be negative") in str(excinfo.value)


def test_compute_soil_roughness_numpy():
    """Test expected result of calculation of soil roughness using numpy arrays"""
    df_dummy = load_calculated_dummy_data()
    arr_ri = df_dummy["Ri"].to_numpy()
    arr_rain = df_dummy["rain"].to_numpy()
    arr_rhm = df_dummy["Rhm"].to_numpy()
    arr_expected_ru = df_dummy["Ru"].to_numpy()
    arr_expected_f1_n = df_dummy["f1_N"].to_numpy()
    arr_expected_f2_ei = df_dummy["f2_EI"].to_numpy()
    expected_result = (arr_expected_ru, arr_expected_f1_n, arr_expected_f2_ei)
    calculated_result = subfactors.compute_soil_roughness(arr_ri, arr_rain, arr_rhm)
    np.testing.assert_allclose(expected_result, calculated_result)


def test_compute_surface_roughness_float():
    """Test expected result of calculation of surface roughness using floats"""
    ru = 8.918848956028448
    expected_SR = 0.9292345717420408
    SR = subfactors.compute_surface_roughness(ru)
    assert SR == expected_SR


def test_compute_surface_roughness_numpy():
    """Test expected result of calculation of surface roughness using numpy arrays"""
    df_dummy = load_calculated_dummy_data()
    arr_ru = df_dummy["Ru"].to_numpy()
    arr_expected_sr = df_dummy["SR"].to_numpy()
    arr_sr = subfactors.compute_surface_roughness(arr_ru)
    np.testing.assert_allclose(arr_expected_sr, arr_sr)


def test_calculate_number_of_days():
    """Test calculation of number of days"""
    start_date = "2016-02-15"
    end_date = "2016-03-01"
    expected_days = 15
    calculated_days = subfactors.calculate_number_of_days(start_date, end_date)
    assert expected_days == calculated_days


def test_calculate_number_of_days_numpy():
    """Test calculation of number of days with numpy array"""
    df_dummy = load_calculated_dummy_data()
    arr_start_dates = df_dummy["bdate"].to_numpy()
    arr_end_dates = df_dummy["edate"].to_numpy()
    arr_expected_days = df_dummy["D"].to_numpy()
    arr_calculated_days = subfactors.calculate_number_of_days(
        arr_start_dates, arr_end_dates
    )
    np.testing.assert_allclose(arr_expected_days, arr_calculated_days)


def test_compute_crop_residu_single_moment_single_place():
    """Test expected result of compute_crop_residu"""
    days = 15
    initial_crop_residu = 5000
    a = 0.02518464958645108
    expected_residu = 3426.9414870271776
    calculated_residu = subfactors.compute_crop_residu(
        days, a, initial_crop_residu, mode="space"
    )
    assert expected_residu == calculated_residu


def test_compute_crop_residu_single_moment_multiple_places():
    """Test expected result of compute_crop_residu"""
    days = 15
    initial_crop_residu = 5000
    a = 0.02518464958645108
    expected_residu = 3426.9414870271776
    calculated_residu = subfactors.compute_crop_residu(
        days, a, initial_crop_residu, mode="space"
    )
    assert expected_residu == calculated_residu


def test_compute_crop_residu_multiple_times_single_place():
    """Test calculation of crop residu on timeseries"""

    df_dummy = load_calculated_dummy_data()
    arr_days = df_dummy["D"].to_numpy()
    arr_a = df_dummy["a"].to_numpy()
    initial_crop_residu = 5000
    arr_expected_residu_end = df_dummy["Bse"].to_numpy()
    calculated_result = subfactors.compute_crop_residu(
        arr_days, arr_a, initial_crop_residu, mode="time"
    )
    np.testing.assert_allclose(arr_expected_residu_end, calculated_result)


def test_compute_harvest_residu_decay_rate_float():
    """Test expected outcome of compute_harvest_residu_decay_rate using floats"""

    rain = 73.56
    temperature = 4.5
    p = 0.05
    expected_w = 2.8555900621118013
    expected_f = 0.583248438971586
    expected_a = 0.0291624219485793

    result = subfactors.compute_harvest_residu_decay_rate(rain, temperature, p)
    assert result == (expected_w, expected_f, expected_a)


def test_compute_harvest_residu_decay_rate_float_negative_rain():
    """Test error handling of a negative rain value input
    in compute_harvest_residu_decay_rate"""
    rain = -10
    temperature = 4.5
    p = 0.05

    with pytest.raises(ValueError) as excinfo:
        subfactors.compute_harvest_residu_decay_rate(rain, temperature, p)
    assert ("Halfmonthly rainfall cannot be negative") in str(excinfo.value)


def test_compute_harvest_residu_decay_rate_numpy():
    """Test expected outcome of compute_harvest_residu_decay_rate using np.array"""
    df_dummy = load_calculated_dummy_data()
    arr_rain = df_dummy["rain"].to_numpy()
    arr_temperature = df_dummy["temp"].to_numpy()
    arr_p = df_dummy["p"].to_numpy()
    arr_expected_w = df_dummy["W"].to_numpy()
    arr_expected_f = df_dummy["F"].to_numpy()
    arr_expected_a = df_dummy["a"].to_numpy()
    result = subfactors.compute_harvest_residu_decay_rate(
        arr_rain, arr_temperature, arr_p
    )
    np.testing.assert_allclose(result, (arr_expected_w, arr_expected_f, arr_expected_a))


def test_compute_soil_cover_float():
    """Test expected result of compute_soil_cover using floats"""
    crop_residu = 5000
    alpha = 5.53
    ru = 6.096
    b = 0.035
    expected_sp = 93.7023900651985
    expected_sc = 0.037643926507827864
    sp, sc = subfactors.compute_soil_cover(crop_residu, alpha, ru, b)
    assert (expected_sp, expected_sc) == (sp, sc)


def test_compute_soil_cover_numpy():
    """Test expected result of compute_soil_cover using np.array"""
    b = 0.035
    df_dummy = load_calculated_dummy_data()
    arr_crop_residu = df_dummy["Bse"].to_numpy()
    arr_alpha = df_dummy["alpha"].to_numpy()
    arr_ru = df_dummy["Ru"].to_numpy()
    arr_expected_sp = df_dummy["Sp"].to_numpy()
    arr_expected_sc = df_dummy["SC"].to_numpy()
    expected_result = (arr_expected_sp, arr_expected_sc)
    calculated_result = subfactors.compute_soil_cover(
        arr_crop_residu, arr_alpha, arr_ru, b
    )
    np.testing.assert_allclose(expected_result, calculated_result)


def test_compute_soil_loss_ratio_float():
    """Test expected result of soil loss ration using floats"""
    # Typical case float
    sc = 0.03768204712884102
    sr = 0.9999092973948398
    cc = 1
    expected_slr = 0.03767862926899866
    slr = subfactors.compute_soil_loss_ratio(sc, sr, cc)
    assert expected_slr == slr


def test_compute_soil_loss_ratio_numpy():
    """Test expected result of soil loss ration using np.array"""
    df_dummy = load_calculated_dummy_data()
    arr_sc = df_dummy["SC"].to_numpy()
    arr_sr = df_dummy["SR"].to_numpy()
    arr_cc = df_dummy["CC"].to_numpy()
    arr_expected_slr = df_dummy["SLR"].to_numpy()
    arr_slr = subfactors.compute_soil_loss_ratio(arr_sc, arr_sr, arr_cc)
    np.testing.assert_allclose(arr_expected_slr, arr_slr)


def test_compute_soil_loss_ratio_float_error_handling():
    """Test error handling of soil loss ratio using floats (negative subfactor)"""
    sc = 0.03768204712884102
    sr = -10
    cc = 0.99
    with pytest.raises(ValueError) as excinfo:
        subfactors.compute_soil_loss_ratio(sc, sr, cc)
    assert ("All soil_loss_ratio subfactors must lie between 0 and 1") in str(
        excinfo.value
    )


def test_compute_soil_loss_ratio_numpy_error_handling():
    """Test error handling of soil loss ratio using np.array (negative subfactor)"""
    arr_sc = np.array([1, 0.03768204712884102, 1])
    arr_sr = np.array([0.9999671539111016, -10, 0.9999037924066257])
    arr_cc = np.array([1, 1, 0.99])
    with pytest.raises(ValueError) as excinfo:
        subfactors.compute_soil_loss_ratio(arr_sc, arr_sr, arr_cc)
    assert ("All soil_loss_ratio subfactors must lie between 0 and 1") in str(
        excinfo.value
    )
