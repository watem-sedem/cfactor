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


def test_compute_soil_roughness():
    """Test calculation of soil roughness"""
    # TO DO
    # cfactor.compute_soil_roughness()


def test_compute_surface_roughness():
    """Test calculation of surface roughness"""
    # TO DO
    # cfactor.compute_surface_roughness()


def test_compute_crop_residu():
    """Test calculation of crop residu"""
    # TO DO
    # cfactor.compute_crop_residu()


def test_compute_harvest_residu_decay_rate():
    """Test calculation of harvest residu decay rate"""
    # TO DO
    # cfactor.compute_harvest_residu_decay_rate()


def test_compute_soil_cover():
    """Test calculation of soil cover"""
    # TO DO
    # cfactor.compute_soil_cover()


def test_compute_soil_loss_ratio():
    """Test calculation of soil loss ration"""
    # TO DO
    # cfactor.compute_soil_loss_ratio()


def test_aggregate_slr_to_crop_factor():
    """Test aggregation of SLR to C-factor"""
    # TO DO
    # cfactor.aggregate_slr_to_c_factor()
