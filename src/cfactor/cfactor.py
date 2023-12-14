import numpy as np

from cfactor import subfactors


def aggregate_slr_to_c_factor(soil_loss_ratio, ei30):
    """Aggregate  SLR according to erosivity

    Parameters
    ----------
    soil_loss_ratio: numpy.ndarray
        Soil loss ratio, see :func:`cfactor.subfactors.compute_soil_loss_ratio`
    ei30: numpy.ndarray
        # TODO: refer to R-factor package with intersphinx

    Returns
    -------
    c: float
        C-factor (-, [0,1])

    """
    sum_r = np.sum(ei30)
    product = soil_loss_ratio * ei30
    c = np.sum(product) / sum_r
    return c


def calculate_slr(
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
):
    """Calculate Soil loss ratio based on basic input parameters

    This function combines the calculations of all subfactors

    .. note::
        Prior landuse and soil moisture are assumed equal to one.

    Parameters
    ----------
    begin_date: str
        start date of period formatted as 'YYYY-MM-DD'
    end_date: str
        end date of period formatted as 'YYYY-MM-DD'
    rain: float or numpy.ndarray
        Summed rainfall (mm) over period defined by begin_date and end_date
    temperature: float or numpy.ndarray
        (Average) temperature (degree C) over period by begin_date and end_date
    rhm: float or numpy.ndarray
        Cumulative rainfall erosivity (in :math:`\\frac{MJ.mm}{ha.year}`)
    ri: float or numpy.ndarray
        # TO DO
    h: float or numpy.ndarray
        Effective drop height (m): estimate of average height between rainfall capture
        by crop and soil.
    fc: float or numpy.ndarray
        Soil cover by crop (in %)
    p: float or numpy.ndarray
        Maximum decay speed (-) #TODO: check unit
    initial_crop_residu: float or numpy.ndarray
        Initial amount of crop residu (kg dry matter / ha)
    alpha: float or numpy.ndarray
        Soil cover in comparison to weight residu (:math:`m^2/kg`)

    Returns
    -------

    """
    cc = subfactors.compute_crop_cover(h, fc)

    _, _, a = subfactors.compute_harvest_residu_decay_rate(rain, temperature, p)

    d = subfactors.calculate_number_of_days(begin_date, end_date)

    crop_residu = subfactors.compute_crop_residu(d, a, initial_crop_residu)

    ru, _, _ = subfactors.compute_soil_roughness(ri, rain, rhm)
    sr = subfactors.compute_surface_roughness(ru)

    _, sc = subfactors.compute_soil_cover(initial_crop_residu, alpha, ru)

    slr = subfactors.compute_soil_loss_ratio(sc, sr, cc)

    return crop_residu, a, d, ru, cc, sr, sc, slr


def calculate_slr_timeseries(
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
):
    """Calculate Soil loss ratio based on basic input parameters

    This function combines the calculations of all subfactors

    .. note::
        Prior landuse and soil moisture are assumed equal to one.

    Parameters
    ----------
    begin_date: numpy.ndarray
        start dates of period formatted as 'YYYY-MM-DD'
    end_date: numpy.ndarray
        end dates of period formatted as 'YYYY-MM-DD'
    rain: float or numpy.ndarray
        Summed rainfall (mm) over period defined by begin_date and end_date
    temperature: float or numpy.ndarray
        (Average) temperature (degree C) over period by begin_date and end_date
    rhm: numpy.ndarray
        Cumulative rainfall erosivity (in :math:`\\frac{MJ.mm}{ha.year}`)
    ri: numpy.ndarray
        # TO DO
    h: numpy.ndarray
        Effective drop height (m): estimate of average height between rainfall capture
        by crop and soil.
    fc: numpy.ndarray
        Soil cover by crop (in %)
    p: numpy.ndarray
        Maximum decay speed (-) #TODO: check unit
    initial_crop_residu: float
        Initial amount of crop residu (kg dry matter / ha)
    alpha: numpy.ndarray
        Soil cover in comparison to weight residu (:math:`m^2/kg`)

    Returns
    -------

    """
    cc = subfactors.compute_crop_cover(h, fc)

    _, _, a = subfactors.compute_harvest_residu_decay_rate(rain, temperature, p)

    d = subfactors.calculate_number_of_days(begin_date, end_date)

    initial_crop_residu, crop_residu = subfactors.compute_crop_residu_timeseries(
        d, a, initial_crop_residu
    )

    ru, _, _ = subfactors.compute_soil_roughness(ri, rain, rhm)
    sr = subfactors.compute_surface_roughness(ru)

    _, sc = subfactors.compute_soil_cover(crop_residu, alpha, ru)

    slr = subfactors.compute_soil_loss_ratio(sc, sr, cc)

    return crop_residu, a, d, ru, cc, sr, sc, slr
