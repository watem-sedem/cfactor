import numpy as np
import pandas as pd
from numba import jit

from cfactor.util import celc_to_fahr

b = 0.035
rii = 6.096

r0 = 25.76  # minimum gemiddelde halfmaandelijks neerslag nodig voor afbraak
# (opm. pagina 6?)
t0 = 37  # degree C
a = 7.76  # degree C


def compute_surface_roughness(soil_roughness):
    """Computes surface roughness subfactor SR

    Computes surface roughness :math:`SR` as

    .. math::

        SR = e(−0.026*(R_u-6.096))


    With :math:`R_u` is a measure for roughness of a parcell (mm). See
    :func:`cfactor.cfactor.compute_soil_roughness` for more information about this
    parameter.

    Parameters
    ----------
    soil_roughness: float or numpy.ndarray
        Soil roughness, output from :func:`cfactor.cfactor.compute_soil_roughness`

    Returns
    -------
    float or numpy.ndarray
        Surface roughness ([0,1])

    """
    return np.exp(-0.026 * (soil_roughness - 6.096))


def compute_soil_roughness(ri, rain, rhm):
    """Compute soil roughness subfactor

    This function computes the roughness for every roughness period id. The beginning
    of a period is defined by a new roughness condition (i.e. prep for a new crop by
    plowing):

    The :math:`R_u` (-) is computed by:

    .. math::

        R_u = 6.096+(D_r*(R_i-6.096))

    The final roughness is referred to as :math:`r_{ii}`, i.e. 6.096.
    The initial roughness is crop dependent (soil preparation dependent).

    The roughness decay function :math:`D_r` is defined as:

    .. math::

        D_r = exp{0.5*\\frac{-0.14}{25.4}P_t}+0.5*\\frac{-0.012}{17.02}EI_t))

    Under the influence of precipitation, the roughness of an agricultural field,
    left undisturbed, will systematically decrease until an (average) minimum roughness
    of 6.096 mm (0.24 inches) is reached. The decrease function Dr is defined to
    compute this decrease (see [1]_). # TODO: add refs to Reynard et al.

    Parameters
    ----------
    ri: float or numpy.ndarray
        #TODO
    rain: float or numpy.ndarray
        Amount of rainfall (in mm, during period).
    rhm: float or numpy.ndarray
        Cumulative rainfall erosivity (in :math:`\\frac{MJ.mm}{ha.year}`)

    Returns
    -------
    soil_roughness:  numpy.ndarray
        Soil roughness (mm)
    f1_n:  numpy.ndarray
        Factor rainfall
    f2_ei:  numpy.ndarray
        Factor erosivity

    Notes
    -----
    1. a slight different formulation is used from [1]_ where the parameter for EI is
       defined as -0.0705, whereas here we use -0.00070505287 (-0.012 / 17.02)
       #TODO: check.
    2. The rhm is equal to EI, for guidelines computation, see [2]_ and #TODO: refer to
       R-factor package via intersphinx.

    References
    ----------

    .. [1] Verbist, K., Schiettecatte, W., Gabriels, D., n.d. Eindrapport:
     “Computermodel RUSLE C-factor.”

    .. [2] Verstraeten, G., Poesen, J., Demarée, G. & Salles, C. (2006). Long-term
     (105 years) variability in rain erosivity as derived from 10-min rainfall depth
     data for Ukkel (Brussels, Belgium): Implications for assessing soil erosion rates.
     Journal of Geophysical Research Atmospheres, 111(22), 1–11.

    """
    if np.any(np.asarray(rain) < 0):
        raise ValueError("Amount of rain cannot be negative")

    f1_n = -0.14 / 25.4 * rain
    f2_ei = -0.012 / 17.02 * rhm

    dr = np.exp(0.5 * f1_n + 0.5 * f2_ei)

    soil_roughness = 6.096 + (dr * (ri - 6.096))

    return soil_roughness, f1_n, f2_ei


def compute_soil_cover(
    crop_residu,
    alpha,
    soil_roughness,
):
    """Computes soil cover (SC) subfactor

    This subfactor is defined as the erosion limiting influence of the ground cover
    by crop residues, stones and non-erodible material in direct contact with the soil
    [1]_


    .. math::

        SC = exp{-b.sp.{\\frac{6.096}{Ru}}^{0.08}}


    with sp being the amount of land being covered by residu

    .. math::

        sp = 100.(1-exp{-\\alpha.B_s})

    with

    - :math:`alpha`: soil cover in comparison to weight residu (:math:`m^2/kg`)

    - :math:`B_s`: amount of residu per unit of area (:math:`kg/m^2`), for definition,
       see :func:`cfactor.cfactor.compute_crop_residu`


    Parameters
    ----------
    crop_residu: float or numpy.ndarray
        crop residu (kg dry matter / ha) for harvest period,
        see :func:`cfactor.cfactor.compute_crop_residu`
    alpha: floar or numpy.ndarray
        Soil cover in comparison to weight residu (:math:`m^2/kg`)
    soil_roughness: float or numpy.ndarray
        Soil roughness (mm), see :func:`cfactor.cfactor.compute_soil_roughness`

    Returns
    -------
    sp: numpy.ndarray
        Percentage soil cover of harvest remains
    soil_cover: numpy.ndarray
        Soil cover (-, [0,1])

    References
    ----------
    .. [1] Verbist, K., Schiettecatte, W., Gabriels, D., n.d. Eindrapport:
     “Computermodel RUSLE C-factor.”

    """
    sp = 100 * (1 - np.exp(-alpha * crop_residu / (100**2)))
    soil_cover = np.exp(-b * sp * ((6.096 / soil_roughness) ** 0.08))
    return sp, soil_cover


def compute_crop_cover(h, fc):
    """Computes crop cover factor based on soil cover crop and effective drop height

    The crop cover or canopy cover subfactor (:math:`CC`) represents the ability of a
    crop to reduce the erosivity of falling raindrops on the soil. The
    effect of the crop cover is expressed as:

    .. math::

        CC = 1-F_c.exp{-0.328H}

    With:
     - :math:`F_c (m²/m²)`: the amount of coverage of the soil by the crop
     - :math:`H (m)`: Effective drop height, the average heigt of falling raindrops
        after they have been intercepted by the crop

    This subfactor changes considerably during the growth of a crop due to the
    increasing crop cover and effective drop height.

    Parameters
    ----------
    h: float or numpy.ndarray
        Effective drop height (m): estimate of average height between rainfall capture
        by crop and soil.
    fc: float or numpy.ndarray
        Soil cover by crop (in %)

    Returns
    -------
    crop_cover: float or nump.ndarray
        Crop cover factor (-, [0,1])

    References
    ----------
    .. [1] Verbist, K., Schiettecatte, W., Gabriels, D., n.d. Eindrapport:
     “Computermodel RUSLE C-factor.”

    """
    if (np.any(fc > 1)) or (np.any(fc < 0)):
        raise ValueError("Soil cover must be between 0 and 1")

    if np.any(h < 0):
        raise ValueError("Effective drop height cannot be negative")

    crop_cover = 1 - fc * np.exp(-0.328 * h)
    return crop_cover


def compute_soil_moisture():
    """
    Computes soil moisture factor

    This function is not yet implemented
    """
    raise NotImplementedError("compute soil moisture is not implemented")


def compute_plu():
    """
    Computes prior land use factor

    This function is not yet implemented as Verstraeten et al.(2002) estimate that PLU
    lies between 0.9 and 1 for a soil that experiences yearly tillage.

    References
    ----------
    Verstraeten et al. (2002) TO DO

    """
    raise NotImplementedError("compute prior land use is not implemented")


def compute_harvest_residu_decay_rate(rain, temperature, p, r0=r0, t0=t0, a=a):
    """Computes crop residu decay coefficient [1]_

    The soil cover by harvest residues changes in time by decay processes.
    RUSLE uses an exponential decay function

    .. math::

        a = p[min(W,F)]

    with:

    .. math::

        W = \\frac{R}{R_0}

    and

    .. math::

        F = \\frac{2(T_a+A)^2.(T_0+a)^2-(T_a+A)^4}{(T_0+A)^4}

    with:

        - :math:`R`: half-monthly rainfall (mm)
        - :math:`R_0`: minimum half-monthly average rainfall (mm)
        - :math:`T_a`: average temperature in half-montlhy period (degree f)
        - :math:`T_0`: optimal temperature for decay (degree f)
        - :math:`A`: coefficient used to express the shape of the decay function
         as a function of temperature.


    Parameters
    ----------
    rain: float or numpy.ndarray
        (Summed) half monthly rainfall (mm)
    temperature: float or numpy.ndarray
        (Average) temperature (degree C)
    p: float or numpy.ndarray
        Maximum decay speed (-) #TODO: check unit
    r0: float
        Average half monthly rainfall (mm)
    t0: float
        Optimal temperature for decay (degree C)
    a: float
        coefficient used to express the shape of the decay function
        as a function of temperature (degree C)

    Returns
    -------
    w: float or numpy.ndarray
        Coefficients weighing rainfall
    f: float or numpy.ndarray
        Coefficients weighing temperature
    harvest_decay_coefficient: float or numpy.ndarray
        Harvest decay coefficient (-)

    References
    ----------
    .. [1] Verbist, K., Schiettecatte, w., Gabriels, D., n.d. Eindrapport:
     “Computermodel RUSLE C-factor.”

    """
    if np.any(np.asarray(rain) < 0):
        raise ValueError("Halfmonthly rainfall cannot be negative")

    w = rain / r0

    temperature = celc_to_fahr(temperature)
    t0 = celc_to_fahr(t0)
    a = celc_to_fahr(a)

    f = (2 * ((temperature + a) ** 2) * ((t0 + a) ** 2) - (temperature + a) ** 4) / (
        (t0 + a) ** 4
    )

    harvest_decay_coefficient = p * np.min([w, f], axis=0)

    return w, f, harvest_decay_coefficient


def calculate_number_of_days(bdate, edate):
    """Computes the number of days between two timestamps

    Parameters
    ----------
    bdate: str or numpy.ndarray

    edate: str or numpy.ndarray

    Returns
    -------
    d: int
        number of days between two timestamps

    """
    d = (pd.to_datetime(edate) - pd.to_datetime(bdate)).days
    return d


@jit(nopython=True)
def compute_crop_residu_timeseries(d, harvest_decay_coefficient, initial_crop_residu):
    """Computes harvest remains on timeseries

    The function :func:`cfactor.cfactor.compute_crop_residu`. is applied on numpy
    arrays. An initial crop residu is given to the function and for every timestep
    the remaining crop residu after decay is calculated

    Parameters
    ----------
    d: numpy.ndarray
        number of days, see
        :func:`cfactor.cfactor.calculate_number_of_days`
    harvest_decay_coefficient: numpy.ndarray
        Harvest decay coefficient (-), see
        :func:`cfactor.cfactor.compute_harvest_residu_decay_rate`
    initial_crop_residu: float
        Initial amount of crop residu (kg dry matter / ha)

    Returns
    -------
    bsi: numpy.ndarray
        Crop residu (kg/m2) at the start of each period
    bse: numpy.ndarray
        Crop residu (kg/m²) at the end of each period

    """
    if not (d.shape == harvest_decay_coefficient.shape):
        raise ValueError("dimension mismatch")

    bse = np.zeros(d.shape[0])
    bsi = np.zeros(d.shape[0])
    bsi[0] = initial_crop_residu
    bse[0] = compute_crop_residu(
        d[0], harvest_decay_coefficient[0], initial_crop_residu
    )
    for i in range(1, d.shape[0]):
        bsi[i] = bse[i - 1]
        bse[i] = compute_crop_residu(d[i], harvest_decay_coefficient[i], bsi[i])
    return bsi, bse


@jit(nopython=True)
def compute_crop_residu(d, harvest_decay_coefficient, initial_crop_residu):
    """
    Computes harvest remains per unit of area over nodes [1]_:

    .. math::
        Bse = Bsb.exp{-a.D}


    with

    - Bse: amount of crop residu at end of period (kg dry matter . :math:`m^{-2}`)
    - Bsb: amount of crop residu at start of period (kg dry matter . :math:`m^{-2}`)
    - a: harvest decay coefficient, see
        :func:`cfactor.cfactor.compute_harvest_residu_decay_rate`.
    - d: number of days

    Parameters
    ----------
    d: int or numpy.ndarray
        number of days, see
        :func:`cfactor.cfactor.calculate_number_of_days`
    harvest_decay_coefficient: float or numpy.ndarray
        Harvest decay coefficient (-), see
        :func:`cfactor.cfactor.compute_harvest_residu_decay_rate`
    initial_crop_residu: float or numpy.ndarray
        Initial amount of crop residu (kg dry matter / ha)

    Returns
    -------
    crop_residu: float
        Crop residu (kg/m2)

    References
    ----------
    .. [1] Verbist, K., Schiettecatte, W., Gabriels, D., n.d. Eindrapport:
     “Computermodel RUSLE C-factor.”

    """
    crop_residu = initial_crop_residu * np.exp(-harvest_decay_coefficient * d)

    return crop_residu


def compute_soil_loss_ratio(
    soil_cover, surface_roughness, crop_cover, soil_moisture=1.0, prior_landuse=1.0
):
    """Computes subcomponents of soil loss ratio

    The soil loss ratio (soil_loss_ratio) is computed by following formula:

    .. math::

        soil_loss_ratio = SC.CC.SR.SM.PLU

    with

    - :math:`SC`: impact of surface cover (due to crop residu), see
      :func:`cfactor.cfactor.compute_soil_cover`.

    - :math:`CC`: impact of canopy cover, :func:`cfactor.cfactor.compute_crop_cover`.

    - :math:`SR`: impact of surface roughness (due to farming operations)
      :func:`cfactor.cfactor.compute_surface_roughness`.

    - :math:`SM`: impact of surface moisture

    - :math:`PLU`: impact of prior land use


    Parameters
    ----------
    soil_cover: float or numpy.ndarray
        soil cover subfactor, see `cfactor.cfactor.compute_soil_cover`
    surface_roughness: float or numpy.ndarray
        surface roughness subfactor, see `cfactor.cfactor.compute_surface_roughness`
    crop_cover: float or numpy.ndarray
        canopy cover subfactor, see `cfactor.cfactor.compute_crop_cover`
    soil_moisture: float or numpy.ndarray
        soil moisture subfactor, see `cfactor.cfactor.compute_soil_moisture`
    prior_landuse: float or numpy.ndarray
        prior land use subfactor, see `cfactor.cfactor.compute_plu`

    Returns
    -------
    float or numpy.ndarray
        soil loss ratio
    """

    for i in [soil_cover, crop_cover, surface_roughness, soil_moisture, prior_landuse]:
        if (np.any(i > 1)) or (np.any(i < 0)):
            raise ValueError("All soil_loss_ratio subfactors must lie between 0 and 1")

    return soil_cover * crop_cover * surface_roughness * soil_moisture * prior_landuse
