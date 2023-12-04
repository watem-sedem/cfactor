import numpy as np
import pandas as pd
from numba import jit

from cfactor.util import celc_to_fahr

b = 0.035
rii = 6.096

R0 = 25.76  # minimum gemiddelde halfmaandelijks neerslag nodig voor afbraak
# (opm. pagina 6?)
T0 = 37  # degree C
A = 7.76  # degree C


def compute_soil_loss_ratio(sc, sr, cc, sm=1, plu=1):
    """Computes subcomponents of soil loss ratio

    The soil loss ratio (SLR) is computed by following formula:

    .. math::

        SLR = SC.CC.SR.SM.PLU

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
    sc: float or np.ndarray
        soil cover subfactor, see `cfactor.cfactor.compute_soil_cover`
    sr: float or np.ndarray
        surface roughness subfactor, see `cfactor.cfactor.compute_surface_roughness`
    cc: float or np.ndarray
        canopy cover subfactor, see `cfactor.cfactor.compute_crop_cover`
    sm: float or np.ndarray
        soil moisture subfactor, see `cfactor.cfactor.compute_soil_moisture`
    plu: float or np.ndarray
        prior land use subfactor, see `cfactor.cfactor.compute_PLU`

    Returns
    -------
    float or np.ndarray
        soil loss ratio
    """

    for i in [sc, cc, sr, sm, plu]:
        if (np.any(i > 1)) or (np.any(i < 0)):
            raise ValueError("All SLR subfactors must lie between 0 and 1")

    slr = sc * cc * sr * sm * plu
    return slr


def compute_surface_roughness(ru):
    """Computes surface roughness subfactor SR

    Computes surface roughness :math:`SR` as

    .. math::

        SR = e(−0.026*(R_u-6.096))


    With :math:`R_u` is a measure for roughness of a parcell (mm). See
    :func:`cfactor.cfactor.compute_soil_roughness` for more information about this
    parameter.

    Parameters
    ----------
    ru: float or numpy.ndarray
        Surface roughness, output from :func:`cfactor.cfactor.compute_soil_roughness`

    Returns
    -------
    float or numpy.ndarray
        Surface roughness ([0,1])

    """
    return np.exp(-0.026 * (ru - 6.096))


def compute_soil_roughness(ri, rain, rhm):
    """Compute soil roughness subfactor per identifier (ri_tag).

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
    ru:  numpy.ndarray
        Soil roughness (mm)
    f1_N:  numpy.ndarray
        Factor rainfall
    f2_EI:  numpy.ndarray
        Factor erosivity

    Notes
    -----
    1. A slight different formulation is used from [1]_ where the parameter for EI is
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

    f1_N = -0.14 / 25.4 * rain
    f2_EI = -0.012 / 17.02 * rhm

    dr = np.exp(0.5 * f1_N + 0.5 * f2_EI)

    ru = 6.096 + (dr * (ri - 6.096))

    return ru, f1_N, f2_EI


def compute_soil_cover(
    identifier,
    begin_date,
    end_date,
    rain,
    temperature,
    maximum_speed,
    initial_crop_residu,
    alpha,
    ru,
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
    identifier: numpy.ndarray
        Period identifier, i.e. every +1 in the identifier marks the start of a new
        'crop residue'-period.
    begin_date: pandas.Series
        Begin dates of periods #TODO: checdk
    end_date: pandas.Series
        End dates of periods #TODO: checdk
    rain: pandas.Series
        Rainfall (in mm)
    temperature: numpy.ndarray
        Temperature (in degree F)
    maximum_speed: numpy.ndarray
        Maximum decay speed (-)
    initial_crop_residu: numpy.ndarray
        Initial amount of crop residu (kg dry matter / ha) for harvest period id (one
        number per period id)
    alpha: numpy.ndarray
        Soil cover in comparison to weight residu (:math:`m^2/kg`)
    ru: numpy.ndarray
        Soil roughness (mm), see :func:`cfactor.cfactor.compute_soil_roughness`

    Returns
    -------
    a: numpy.ndarray
        Harvest decay coefficient
    Bsi: numpy.ndarray
        Harvest remains per unit of area(kg/m2)
    sp: numpy.ndarray
        Percentage soil cover of harvest remains
    w: numpy.ndarray
        Precipitation coefficient weighting (half-)monthly rainfall and minimum average
        monthly rainfall.
    f: numpy.ndarray
        Coefficients computed based on temperature, used to define shape crop residu
        decay rate.
    sc: numpy.ndarray
        Soil cover (-, [0,1])

    References
    ----------
    .. [1] Verbist, K., Schiettecatte, W., Gabriels, D., n.d. Eindrapport:
     “Computermodel RUSLE C-factor.”

    """
    a_val = np.zeros([len(identifier)])
    crop_residu = np.zeros([len(identifier)])
    sp = np.zeros([len(identifier)])
    w = np.zeros([len(identifier)])
    f = np.zeros([len(identifier)])
    sc = np.ones([len(identifier)])

    for i in identifier[identifier != 0].unique():
        cond = (identifier == i).values.flatten()
        w[cond], f[cond], a_val[cond] = compute_harvest_residu_decay_rate(
            rain[cond], temperature[cond], maximum_speed[cond]
        )
        crop_residu[cond] = compute_crop_residu(
            begin_date[cond], end_date[cond], a_val[cond], initial_crop_residu[cond]
        )
        sp[cond] = 100 * (1 - np.exp(-alpha[cond] * crop_residu[cond] / (100**2)))
        sc[cond] = np.exp(-b * sp[cond] * ((6.096 / (ru[cond])) ** 0.08))
    return a_val, crop_residu, sp, w, f, sc


def compute_crop_cover(H, Fc):
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
    H: float or numpy.ndarray
        Effective drop height (m): estimate of average height between rainfall capture
        by crop and soil.
    Fc: float or numpy.ndarray
        Soil cover by crop (in %)

    Returns
    -------
    cc: float or nump.ndarray
        Crop cover factor (-, [0,1])

    References
    ----------
    .. [1] Verbist, K., Schiettecatte, W., Gabriels, D., n.d. Eindrapport:
     “Computermodel RUSLE C-factor.”

    """
    if (np.any(Fc > 1)) or (np.any(Fc < 0)):
        raise ValueError("Soil cover must be between 0 and 1")

    if np.any(H < 0):
        raise ValueError("Effective drop height cannot be negative")

    cc = 1 - Fc * np.exp(-0.328 * H)
    return cc


def compute_soil_moisture():
    """
    Computes soil moisture factor

    This function is not yet implemented
    """
    raise NotImplementedError("compute soil moisture is not implemented")


def compute_PLU():
    """
    Computes prior land use factor

    This function is not yet implemented as Verstraeten et al.(2002) estimate that PLU
    lies between 0.9 and 1 for a soil that experiences yearly tillage.

    References
    ----------
    Verstraeten et al. (2002) TO DO

    """
    raise NotImplementedError("compute prior land use is not implemented")


def aggregate_slr_to_c_factor(SLR, EI30):
    """Aggregate  SLR according to erosivity

    Parameters
    ----------
    SLR: numpy.ndarray
        Soil loss ratio, see :func:`cfactor.cfactor._ratio`
    EI30: numpy.ndarray
        # TODO: refer to R-factor package with intersphinx

    Returns
    -------
    C: float
        C-factor (-, [0,1])

    """
    sumR = np.sum(EI30)
    product = SLR * EI30
    C = np.sum(product) / sumR
    return C


def compute_harvest_residu_decay_rate(rain, temperature, p, R0=R0, T0=T0, A=A):
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

        F = \\frac{2(T_a+A)^2.(T_0+A)^2-(T_a+A)^4}{(T_0+A)^4}

    with:

        - :math:`R`: half-monthly rainfall (mm)
        - :math:`R_0`: minimum half-monthly average rainfall (mm)
        - :math:`T_a`: average temperature in half-montlhy period (degree F)
        - :math:`T_0`: optimal temperature for decay (degree F)
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
    R0: float
        Average half monthly rainfall (mm)
    T0: float
        Optimal temperature for decay (degree C)
    A: float
        coefficient used to express the shape of the decay function
        as a function of temperature (degree C)

    Returns
    -------
    a: float or numpy.ndarray
        Harvest decay coefficient (-)
    W: float or numpy.ndarray
        Coefficients weighing rainfall
    F: float or numpy.ndarray
        Coefficients weighing temperature

    References
    ----------
    .. [1] Verbist, K., Schiettecatte, W., Gabriels, D., n.d. Eindrapport:
     “Computermodel RUSLE C-factor.”

    """
    if np.any(np.asarray(rain) < 0):
        raise ValueError("Halfmonthly rainfall cannot be negative")

    W = rain / R0

    temperature = celc_to_fahr(temperature)
    T0 = celc_to_fahr(T0)
    A = celc_to_fahr(A)

    F = (2 * ((temperature + A) ** 2) * ((T0 + A) ** 2) - (temperature + A) ** 4) / (
        (T0 + A) ** 4
    )

    a = p * np.min([W, F], axis=0)

    return W, F, a


def calculate_number_of_days(bdate, edate):
    """Computes the number of days between two timestamps

    Parameters
    ----------
    bdate: str or np.array

    edate: str or np.array

    Returns
    -------
    d: int
        number of days between two timestamps

    """
    d = (pd.to_datetime(edate) - pd.to_datetime(bdate)).days
    return d


@jit(nopython=True)
def compute_crop_residu_timeseries(d, a, initial_crop_residu):
    """Computes harvest remains on timeseries

    The function :func:`cfactor.cfactor.compute_crop_residu`. is applied on numpy
    arrays. An intial crop residu is given to the function and for every timestep
    the remaining crop residu after decay is calculated

    Parameters
    ----------
    d: np.array
        number of days, see
        :func:`cfactor.cfactor.calculate_number_of_days`
    a: np.array
        Harvest decay coefficient (-), see
        :func:`cfactor.cfactor.compute_harvest_residu_decay_rate`
    initial_crop_residu: float
        Initial amount of crop residu (kg dry matter / ha)

    Returns
    -------
    bsi: np.array
        Crop residu (kg/m2) at the start of each period
    bse: np.array
        Crop residu (kg/m²) at the end of each period

    """
    if not (d.shape == a.shape):
        raise ValueError("dimension mismatch")

    bse = np.zeros(d.shape[0])
    bsi = np.zeros(d.shape[0])
    bsi[0] = initial_crop_residu
    bse[0] = compute_crop_residu(d[0], a[0], initial_crop_residu)
    for i in range(1, d.shape[0]):
        bsi[i] = bse[i - 1]
        bse[i] = compute_crop_residu(d[i], a[i], bsi[i])
    return bsi, bse


@jit(nopython=True)
def compute_crop_residu(d, a, initial_crop_residu):
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
    d: int
        number of days, see
        :func:`cfactor.cfactor.calculate_number_of_days`
    a: float
        Harvest decay coefficient (-), see
        :func:`cfactor.cfactor.compute_harvest_residu_decay_rate`
    initial_crop_residu: float
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
    crop_residu = initial_crop_residu * np.exp(-a * d)

    return crop_residu
