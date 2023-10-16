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
