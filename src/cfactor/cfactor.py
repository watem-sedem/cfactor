import numpy as np

from cfactor.util import celc_to_fahr

b = 0.035
rii = 6.096

R0 = 25.76  # minimum gemiddelde halfmaandelijks neerslag nodig voor afbraak
# (opm. pagina 6?)
T0 = 37  # ° C
A = celc_to_fahr(7.76)


def compute_soil_loss_ratio(
    roughness_period_id,
    ri,
    rain,
    rhm,
    harvest_period_id,
    bdate,
    edate,
    temperature,
    p,
    bsi,
    alpha,
    H,
    Fc,
    SM,
    PLU,
):
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
    roughness_period_id: numpy.ndarray
        See :func:`cfactor.cfactor.compute_soil_roughness`
    ri: numpy.ndarray
        See :func:`cfactor.cfactor.compute_soil_roughness`
    rain: numpy.ndarray
        See :func:`cfactor.cfactor.compute_soil_roughness`
    rhm: numpy.ndarray
        See :func:`cfactor.cfactor.compute_soil_roughness`
    harvest_period_id: numpy.ndarray
    bdate: numpy.ndarray
    edate: numpy.ndarray
    temperature: numpy.ndarray
    p: numpy.ndarray
    bsi: numpy.ndarray

    Returns
    -------
    SLR:

    """
    f1_N, f2_EI, ru = compute_soil_roughness(
        roughness_period_id.values.flatten(), ri, rain, rhm
    )
    SR = compute_surface_roughness(ru)
    a, Bsb, Sp, W, F, SC = compute_soil_cover(
        harvest_period_id, bdate, edate, rain, temperature, p, bsi, alpha, ru
    )
    CC = compute_crop_cover(H, Fc)
    SM = 1
    PLU = 1
    SLR = SC * CC * SR * SM * PLU
    SLR[np.isnan(SLR)] = 1

    return SLR


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
    ru: numpy.ndarray
        Surface roughness, output from :func:`cfactor.cfactor.compute_soil_roughness`

    Returns
    -------
    numpy.ndarray
        Surface roughness ([0,1])

    """
    return np.exp(-0.026 * (ru - rii))


def compute_soil_roughness(identifier, ri, rain, rhm):
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
    identifier: numpy.ndarray
        Period identifier, i.e. every +1 in the identifier marks the start of a new
        'decay in roughness'-period.
    ri:  numpy.ndarray
        #TODO
    rain:  numpy.ndarray
        Amount of rainfall (in mm, during period).
    rhm:  numpy.ndarray
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
    f1_N = np.zeros([len(identifier)])
    f2_EI = np.zeros([len(identifier)])

    for i in np.unique(identifier):
        cond = identifier == i
        f1_N[cond] = -0.14 / 25.4 * rain[cond].cumsum()
        f2_EI[cond] = -0.012 / 17.02 * rhm[cond].cumsum()

    dr = np.exp(0.5 * f1_N + 0.5 * f2_EI)

    ru = rii + (dr * (ri - rii))
    ru[ru.isnull()] = rii

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
        Temperature (in °F)
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
    H: numpy.ndarray
        Effective drop height (m): estimate of average height between rainfall capture
        by crop and soil.
    Fc: numpy.ndarray
        Soil cover by crop (in %)

    Returns
    -------
    cc: nump.ndarray
        Crop cover factor (-, [0,1])

    References
    ----------
    .. [1] Verbist, K., Schiettecatte, W., Gabriels, D., n.d. Eindrapport:
     “Computermodel RUSLE C-factor.”

    """
    if (Fc > 1) or (Fc < 0):
        raise ValueError("Soil cover must be between 0 and 1")

    if H < 0:
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
    C = np.sum(product / sumR)
    return C


def compute_harvest_residu_decay_rate(rain, temperature, p, R0=R0, T0=T0):
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
        - :math:`T_a`: average temperature in half-montlhy period (°F)
        - :math:`T_0`: optimal temperature for decay (°F)
        - :math:`A`: coefficient used to express the shape of the decay function
         as a function of temperature.


    Parameters
    ----------
    rain: numpy.ndarray
        (Summed) half monthly rainfall (mm)
    temperature: numpy.ndarray
        (Average) temperature (°C)
    p: numpy.ndarray
        Maximum decay speed (-) #TODO: check unit
    R0: float
        Average half montly rainfall (mm)
    T0: float
        Optimal temperature for decay (°C)

    Returns
    -------
    a: numpy.ndarray
        Harvest decay coefficient (-)
    W: numpy.ndarray
        Coefficients weighing rainfall
    F: numpy.ndarray
        Coefficients weighing temperature

    References
    ----------
    .. [1] Verbist, K., Schiettecatte, W., Gabriels, D., n.d. Eindrapport:
     “Computermodel RUSLE C-factor.”

    """
    W = rain / R0

    temperature = celc_to_fahr(temperature)
    T0 = celc_to_fahr(T0)

    F = (2 * ((temperature + A) ** 2) * ((T0 + A) ** 2) - (temperature + A) ** 4) / (
        (T0 + A) ** 4
    )

    a = (
        p * np.min([W, F], axis=0)
        if len(W) > 1
        else p * np.min([W.values[0], F.values[0]])
    )

    return W, F, a


def compute_crop_residu(bdate, edate, a, initial_crop_residu):
    """
    Computes harvest remains per unit of area over nodes [1]_:

    .. math::
        Bse = Bsb.exp{-a.D}


    with

    - Bse: amount of crop residu at end of period (kg dry matter . :math:`m^{-2}`)
    - Bsb: amount of crop residu at start of period (kg dry matter . :math:`m^{-2}`)
    - a: harvest decay coefficient, see
        :func:`cfactor.cfactor.compute_harvest_residu_decay_rate`.
    - D: number of days # TODO: check unit

    Parameters
    ----------
    bdate: # TODO
    edate: # TODO
    a: numpy.ndarray
        Harvest decay coefficient (-), see
        :func:`cfactor.cfactor.compute_harvest_residu_decay_rate`. # TODO: check unit
    initial_crop_residu: numpy.ndarray
        Initial amount of crop residu (kg dry matter / ha)

    Returns
    -------
    crop_residu: numpy.ndarray
        Crop residu (kg/m2)

    References
    ----------
    .. [1] Verbist, K., Schiettecatte, W., Gabriels, D., n.d. Eindrapport:
     “Computermodel RUSLE C-factor.”

    """
    crop_residu = np.zeros(len(a))
    # (SG) compute remains on middle of computational node
    D = [7] + [
        (
            bdate[i]
            + (edate[i] - bdate[i]) / 2
            - (bdate[i - 1] + (edate[i - 1] - bdate[i - 1]) / 2)
        ).days
        for i in range(1, len(edate), 1)
    ]
    exp = np.exp(-a * D)
    # (SG) compute harvest remains
    for i in range(len(exp)):
        crop_residu[i] = (
            crop_residu[i - 1] * exp[i] if i != 0 else initial_crop_residu[i] * exp[i]
        )

    return crop_residu
