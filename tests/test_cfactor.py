import numpy as np
import pytest

from cfactor.cfactor import calculate_soil_loss_ratio


@pytest.mark.skip(reason="not yet implemented")
def test_aggregate_slr_to_c_factor():
    """Test aggregation of soil_loss_ratio to C-factor"""
    # TO DO
    # cfactor.aggregate_slr_to_c_factor()


def test_calculate_soil_loss_ratio_single_moment_single_place():
    """Test calculation of SLR based on basic data for
    a single moment and a single place"""
    begin_date = "2023-06-01"
    end_date = "2023-06-15"
    rain = 35.41
    temperature = 18.48
    rhm = 109.22
    ri = 6.1
    h = 0.15
    fc = 0.905
    p = 0.03
    alpha = 5.53
    initial_crop_residu = 5000
    mode = "space"
    expected_crop_residu = 3523.151343869222
    expected_harvest_decay_coefficient = 0.025005861085209136
    expected_days = 14
    expected_soil_roughness = 6.099491057885111
    expected_crop_cover = 0.13844840517399903
    expected_surface_roughness = 0.9999092366142325
    expected_soil_cover = 0.03764958126621877
    expected_soil_loss_ratio = 0.005212051375406498

    (
        crop_residu,
        harvest_decay_coefficient,
        days,
        soil_roughness,
        crop_cover,
        surface_roughness,
        soil_cover,
        soil_loss_ratio,
    ) = calculate_soil_loss_ratio(
        begin_date,
        end_date,
        rain,
        temperature,
        rhm,
        ri,
        h,
        fc,
        p,
        initial_crop_residu,
        alpha,
        mode,
    )

    assert crop_residu == expected_crop_residu
    assert harvest_decay_coefficient == expected_harvest_decay_coefficient
    assert days == expected_days
    assert soil_roughness == expected_soil_roughness
    assert crop_cover == expected_crop_cover
    assert surface_roughness == expected_surface_roughness
    assert soil_cover == expected_soil_cover
    assert soil_loss_ratio == expected_soil_loss_ratio


def test_calculate_soil_loss_ratio_single_moment_multiple_places():
    """Test calculation of SLR based on a single moment but for multiple places"""
    begin_date = "2023-06-01"
    end_date = "2023-06-15"
    rain = np.array([35.41, 33.95, 28.51, 26.76])
    temperature = np.array([18.48, 17.23, 18.86, 1.47])
    rhm = np.array([109.22, 145.195, 53.505, 28.47])
    ri = np.array([6.1, 10.2, 6.096, 6.1])
    h = np.array([0.15, 0.015, 0.13, 0])
    fc = np.array([0.905, 0.875, 0.725, 0.405])
    p = np.array([0.03, 0.01, 0.05, 0.03])
    alpha = np.array([5.53, 5.53, 9.21, 23.03])
    initial_crop_residu = np.array([5000, 4500, 150, 3500])
    mode = "space"
    expected_crop_residu = np.array(
        [3523.15134387, 4015.53444796, 83.34852713, 2807.18026399]
    )
    expected_harvest_decay_coefficient = np.array(
        [0.02500586, 0.00813621, 0.04197174, 0.01575589]
    )
    expected_days = 14
    expected_soil_roughness = np.array([6.09949106, 9.64694597, 6.096, 6.09967852])
    expected_crop_cover = np.array([0.13844841, 0.12929443, 0.30526418, 0.595])
    expected_surface_roughness = np.array([0.99990924, 0.91180913, 1.0, 0.99990436])
    expected_soil_cover = np.array([0.03764958, 0.04533543, 0.6366012, 0.03023588])
    expected_soil_loss_ratio = np.array(
        [0.00521205, 0.00534468, 0.19433155, 0.01798863]
    )

    (
        crop_residu,
        harvest_decay_coefficient,
        days,
        soil_roughness,
        crop_cover,
        surface_roughness,
        soil_cover,
        soil_loss_ratio,
    ) = calculate_soil_loss_ratio(
        begin_date,
        end_date,
        rain,
        temperature,
        rhm,
        ri,
        h,
        fc,
        p,
        initial_crop_residu,
        alpha,
        mode,
    )

    np.testing.assert_allclose(crop_residu, expected_crop_residu)
    np.testing.assert_allclose(
        harvest_decay_coefficient, expected_harvest_decay_coefficient
    )
    np.testing.assert_allclose(days, expected_days)
    np.testing.assert_allclose(soil_roughness, expected_soil_roughness)
    np.testing.assert_allclose(crop_cover, expected_crop_cover)
    np.testing.assert_allclose(surface_roughness, expected_surface_roughness)
    np.testing.assert_allclose(soil_cover, expected_soil_cover)
    np.testing.assert_allclose(soil_loss_ratio, expected_soil_loss_ratio)


def test_calculate_soil_loss_ratio_multiple_moments_single_place():
    """Test calculation of SLR based on a single moment but for multiple places"""
    begin_date = np.array(["2016-01-01", "2016-01-15", "2016-02-01", "2016-02-15"])
    end_date = np.array(["2016-01-15", "2016-02-01", "2016-02-15", "2016-03-01"])
    rain = np.array([35.41, 10.2, 28.51, 26.76])
    temperature = np.array([18.48, 17.23, 18.86, 1.47])
    rhm = np.array([109.22, 145.195, 53.505, 28.47])
    ri = np.array([6.1, 10.2, 6.096, 6.1])
    h = np.array([0.15, 0.015, 0.13, 0])
    fc = np.array([0.905, 0.875, 0.725, 0.405])
    p = np.array([0.03, 0.01, 0.05, 0.03])
    alpha = np.array([5.53, 5.53, 9.21, 23.03])
    initial_crop_residu = 5000
    mode = "time"
    expected_crop_residu = np.array(
        [3523.15134387, 3293.80091959, 1830.22303543, 1444.98579841]
    )
    expected_harvest_decay_coefficient = np.array(
        [0.02500586, 0.00395963, 0.04197174, 0.01575589]
    )
    expected_days = np.array([14, 17, 14, 15])
    expected_soil_roughness = np.array([6.09949106, 9.88714011, 6.096, 6.09967852])
    expected_crop_cover = np.array([0.13844841, 0.12929443, 0.30526418, 0.595])
    expected_surface_roughness = np.array([0.99990924, 0.90613258, 1.0, 0.99990436])
    expected_soil_cover = np.array([0.03764958, 0.04263305, 0.03127319, 0.03020354])
    expected_soil_loss_ratio = np.array([0.00521205, 0.0049948, 0.00954659, 0.01796939])

    (
        crop_residu,
        harvest_decay_coefficient,
        days,
        soil_roughness,
        crop_cover,
        surface_roughness,
        soil_cover,
        soil_loss_ratio,
    ) = calculate_soil_loss_ratio(
        begin_date,
        end_date,
        rain,
        temperature,
        rhm,
        ri,
        h,
        fc,
        p,
        initial_crop_residu,
        alpha,
        mode,
    )

    np.testing.assert_allclose(crop_residu, expected_crop_residu)
    np.testing.assert_allclose(
        harvest_decay_coefficient, expected_harvest_decay_coefficient
    )
    np.testing.assert_allclose(days, expected_days)
    np.testing.assert_allclose(soil_roughness, expected_soil_roughness)
    np.testing.assert_allclose(crop_cover, expected_crop_cover)
    np.testing.assert_allclose(surface_roughness, expected_surface_roughness)
    np.testing.assert_allclose(soil_cover, expected_soil_cover)
    np.testing.assert_allclose(soil_loss_ratio, expected_soil_loss_ratio)
