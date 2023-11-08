import numpy as np
import pytest

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


@pytest.mark.skip(reason="not yet implemented")
def test_compute_soil_roughness():
    """Test calculation of soil roughness"""
    # TO DO
    # cfactor.compute_soil_roughness()


def test_compute_surface_roughness():
    """Test calculation of surface roughness"""
    # Typical case
    ru = 8.918848956028448
    expected_SR = 0.9292345717420408
    SR = cfactor.compute_surface_roughness(ru)
    assert SR == expected_SR


@pytest.mark.skip(reason="not yet implemented")
def test_compute_crop_residu():
    """Test calculation of crop residu"""
    # TO DO
    # cfactor.compute_crop_residu()


def test_compute_harvest_residu_decay_rate():
    """Test calculation of harvest residu decay rate"""

    # Typical case
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
    sc = np.array([1, 0.03768204712884102, 1])
    sr = np.array([0.9999671539111016, 0.9999092973948398, 0.9999037924066257])
    cc = np.array([1, 1, 0.99])
    expected_slr = np.array(
        [0.9999671539111016, 0.03767862926899866, 0.9899047544825594]
    )
    slr = cfactor.compute_soil_loss_ratio(sc, sr, cc)
    np.testing.assert_array_equal(expected_slr, slr)


@pytest.mark.skip(reason="not yet implemented")
def test_aggregate_slr_to_crop_factor():
    """Test aggregation of SLR to C-factor"""
    # TO DO
    # cfactor.aggregate_slr_to_c_factor()
