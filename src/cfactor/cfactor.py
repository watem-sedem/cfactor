import numpy as np
from util import celc_to_fahr

b = 0.035
Rii = 6.096

R0 = 25.76  # minimum gemiddelde halfmaandelijks neerslag nodig voor afbraak
# (opm. pagina 6?)
T0 = celc_to_fahr(37)
A = celc_to_fahr(7.76)


def compute_crop_management_factor(
    Ri_tag, Ri, rain, Rhm, har_tag, bdate, edate, temp, p, Bsi, alpha, H, Fc, SM, PLU
):
    """
    #TODO

    """
    f1_N, f2_EI, Ru = compute_Ru(Ri_tag.values.flatten(), Ri, rain, Rhm)

    SR = compute_SR(Ru)

    a, Bsb, Sp, W, F, SC = compute_SC(
        har_tag, bdate, edate, rain, temp, p, Bsi, alpha, Ru
    )

    CC = compute_CC(H, Fc)

    SM = compute_SM(SM)

    PLU = compute_PLU(PLU)

    SLR = compute_SLR(SC, CC, SR, SM, PLU)

    return SLR


def compute_Ru(Ri_tag, Ri, rain, Rhm):
    """
    Computes parcel roughness  Ru = 6.096+[Dr*(Ri-6.096)]

    Parameters
    ----------
    Ri_id (pd series, float): see series ``grid`` in :func:`ComputeCFactor`
    and :func:`prepare_grid`
    Ri (pd series, float): see series ``grid`` in :func:`ComputeCFactor` and
    :func:`prepare_grid`
    Rain (pd series, float): see series ``grid`` in :func:`ComputeCFactor`
    EI30 (pd series, float): see series ``grid`` in :func:`ComputeCFactor`

    Returns
    -------
    'Ru' (series, float): soil roughness
    'f1_N' (series, float): first part formula, cumulative rainfall, weighted
    with factor
    'f2_EI' (series, float): second part formula, cumulative rainfall
    erosivity, weighted with factor

    (SG) Koen Verbist( 2004)
    Dr = exp(0.5*(-0.0055*Pt)+0.5*(-0.0705*EIt))
    met EIt in MJ mm/(ha jaar)
    Koen Verbist, factor 0.0705 in documentatie is fout (zie hierboven),
    gezien in codering: 0.012/17.02=0.000705
    Dr[date] = Math.exp(0.5*(-0.14*(cumulOperationP/25.4))+0.5*(-0.012*(
    cumulOperationEI/17.02)));
    met EIt in (MJ m)/(mm jaar) (volgt uit code, maar deze eenheid klopt niet?)
    Inputwaarden in code zijn erosiviteitswaarden (jaargemiddelden of jaar),
    en deze kloppen volgens grootte-orde
    (eenheid: (MJ mm)/(ha h) zoals in  Verstraeten, G., Poesen, J., Demarée, G.,
    & Salles, C. (2006). Long-term (105 years) variability " \
                in rain erosivity as derived from 10-min rainfall depth data for
                Ukkel (Brussels, Belgium): Implications for assessing " \
                soil erosion rates. Journal of Geophysical Research Atmospheres,
                111(22), 1–11.)"""
    # (SG) compute per Ri id
    Dr = np.zeros([len(Ri_tag)])
    f1_N = np.zeros([len(Ri_tag)])
    f2_EI = np.zeros([len(Ri_tag)])

    for i in np.unique(Ri_tag):
        cond = Ri_tag == i
        f1_N[cond] = -0.14 / 25.4 * rain[cond].cumsum()
        f2_EI[cond] = -0.012 / 17.02 * Rhm[cond].cumsum()

    Dr = expDr(f1_N, f2_EI)
    # Dr[cond] = [math.exp(0.5*f1_N[i] + 0.5*f2_EI[i]) for i in range(len(f1_N))]

    Ru = Rii + (Dr * (Ri - Rii))

    # (SG) set empy Ru equal to Ri
    Ru[Ru.isnull()] = Rii

    return f1_N, f2_EI, Ru


def expDr(f1_N, f2_EI):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    return np.exp(0.5 * f1_N + 0.5 * f2_EI)


def compute_SR(Ru, Rii):
    """
    Computes Surface Roughness (SR): SR e(−0.026*(R-6.096))

    Parameters
    ----------
    'Ru' (series, flloat): soil roughness

    Returns
    -------
    'SR' (series float): surface roughness
    """
    SR = np.exp(-0.026 * (Ru - Rii))
    return SR


def compute_SC(har_id, bdate, edate, rain, temp, p, Bsi, alpha, Ru):
    """
    Computes soil cover from harvest remains (SC): exp(-b*Sp*(6.096/Ru)**0.08))

    Parameters
    ----------
    har_id (pd series, int): see series ``grid`` in :func:`prepare_grid`
    bdate_id (pd series, datetime): see series ``grid`` in
    :func:`ComputeCFactor`
    edate_id (pd series, datetime): see series ``grid`` in
    :func:`ComputeCFactor`
    rain (pd series, float): see series ``grid`` in :func:`ComputeCFactor`
    temp (pd series, float): see series ``grid`` in :func:`ComputeCFactor`
    p (pd series, float): see series ``grid`` in  :func:`prepare_grid`
    Bsi (pd series, float): see series ``grid`` in :func:`prepare_grid`
    alpha (pd series, float): see series ``grid`` in :func:`prepare_grid`
    Ru (pd series, float): see series ``Ru`` in :func:compute_Ru

    Returns
    -------
    'a' ( pd series, float): harvest decay coefficient
    'Bsi'(pd series, float): harvest remains per unit of area(kg/m2)
    'Sp' (pd series, float): Percentage soil cover of harvest remains
    'W' and 'F' (pd series, float): coefficients weighing rainfall and
    temperature
    'SC' (pd series, float): Surface cover (-)
    """
    a_val = np.zeros([len(har_id)])
    Bsb = np.zeros([len(har_id)])
    Sp = np.zeros([len(har_id)])
    W = np.zeros([len(har_id)])
    F = np.zeros([len(har_id)])
    SC = np.ones([len(har_id)])

    for i in har_id.loc[har_id != 0].unique():
        cond = (har_id == i).values.flatten()
        # (SG) compute degradation rate
        W[cond], F[cond], a_val[cond] = compute_a(
            rain.loc[cond], temp.loc[cond], p.loc[cond]
        )
        # (SG) compute Bsi
        Bsb[cond] = compute_Bsb(
            bdate.loc[cond], edate.loc[cond], a_val[cond], Bsi.loc[cond]
        )
        # (SG) compute Sp (divide by 10 000 Bsi (kg/ha -> kg/m2)
        Sp[cond] = 100 * (1 - np.exp(-alpha[cond] * Bsb[cond] / (100**2)))
        # (SG) compute SC
        SC[cond] = np.exp(-b * Sp[cond] * ((6.096 / (Ru[cond])) ** 0.08))
        # SC[cond] = [math.exp(-b*Sp[i]*((6.069 / (Ru[i]**0.08)))) for i in
        # range(len(Sp))]
    return a_val, Bsb, Sp, W, F, SC


def compute_CC(H, Fc):
    """
    Computes crop cover factor based on formula's Verbist, K. (2004).
    Computermodel RUSLE C-factor.

    Parameters
    ----------
    'H' (pd series, float): see parameter ``ggg`` in :func:`ComputeCFactor`.
    'F (pd series, float): see parameter ``ggg`` in :func:`ComputeCFactor`.

    Returns
    -------
    'CC' (pd series, float): crop cover factor
    """

    # (SG) compute CC
    CC = 1 - Fc * np.exp(-0.328 * H)
    return CC


def compute_SM(grid):
    """
    Computes soil moisture factor based on formula's Verbist, K. (2004).
    Computermodel RUSLE C-factor.

    Parameters
    ----------
    'grid' (pd df): see parameter ``grid`` in :func:`ComputeCFactor`.

    Returns
    -------
    'SM' (pd series, float): soil moisture

    Verbist et al. (2004): Deze factor brengt de invloed in rekening van het
    bodemvochtgehalte op de watererosie.
    Deze parameter moet enkel veranderd worden ingeval de bodemvochtsituatie
    gedurende het jaar significant verschilt
    van een situatie waarbij het perceel het hele jaar braak wordt gelaten en
    jaarlijks wordt geploegd.
    Voor een normaal akkerperceel wordt de waarde één voorgesteld.
    Deze waarde wordt dan ook gebruikt bij de berekening van de gewasfactoren
    voor akkerbouwpercelen in Vlaanderen,
    zodat deze subfactor geen rol speelt in de berekeningen
    """

    return [1] * len(grid)


def compute_PLU(grid):
    """
    Computes prior land use factor based on formula's Verbist, K. (2004).
    Computermodel RUSLE C-factor.

    Parameters
    ----------
    'grid' (pd df): see parameter ``grid`` in :func:`ComputeCFactor`.

    Returns
    -------
    'PLU' (pd series, float): past land use
    """

    """
    Verbist et al. (2004): Volgens Verstraeten et al. (2002) kan deze subfactor
    geschat worden tussen 0,9 en 1 voor
    een jaarlijks geploegde bodem. In de berekening van de gewasfactor wordt dan
    ook verder een rekening gehouden met
    deze subfactor (de subfactor PLU wordt gelijkgesteld aan 1).
    """

    return [1] * len(grid)


def compute_SLR(SC, CC, SR, SM, PLU):
    """
    Computes soil loss ration factor based on formula's Verbist, K. (2004).
    Computermodel RUSLE C-factor.

    Parameters
    ----------
    SC (pd series, float): soil cover
    CC (pd series, float): crop cover
    SR (pd series, float): surface cover
    SM (pd series, float): soil moistur
    PLU (pd series, float): past land use

    Returns
    -------
         'SLR' (pd series, float): soil loss ratio
    """
    SLR = SC * CC * SR * SM * PLU
    SLR.loc[SLR.isnull()] = 1
    return SLR


def weight_SLR(SLR, EI30):
    """
    Weight SLR factor based on formula's Verbist, K. (2004). Computermodel RUSLE
    C-factor.

    Parameters
    ----------
    'SLR' (pd series, float): soil loss ratio
    'EI30' (pd series, float): see series ``grid`` in :func:`ComputeCFactor`
    'bdate' (pd series, float): see series ``grid`` in :func:`ComputeCFactor
    'output_interval' (str): output interval for C-factor (!= interval
    computation grid), either monthly 'M',
                            semi-monthly'SMS' or yearly 'Y'
    Returns
    -------
         'grid' (pd df): see parameter ``grid`` in :func:`ComputeCFactor`.
    """
    # (SG) compute sum of average EI30 values
    sumR = np.sum(EI30)
    product = SLR * EI30
    # (SG) compute weighted C according to to output interval
    C = np.sum(product / sumR)
    return C


def compute_a(rain, temp, p):
    """
    Computes decay coefficient.

    Parameters
    ----------
    'rain' (pd series, float): see parameter ``grid`` in :func:`ComputeCfactor`.
    'temp' (pd series, float): see parameter ``grid`` in :func:`ComputeCfactor`.
    'p' (pd series, float): see parameter ``grid`` in :func:`ComputeCFactor`.

    Returns
    -------
    'a' ( pd series, float): harvest decay coefficient
    'W' and 'F' (pd series, float): coefficients weighing rainfall and
    temperature
    """
    # (opm. pagina 6?)

    # (SG) compute W
    W = rain / R0
    # (SG) compute F (in fahr!)
    temp = celc_to_fahr(temp)
    F = (2 * ((temp + A) ** 2) * ((T0 + A) ** 2) - (temp + A) ** 4) / ((T0 + A) ** 4)
    # (SG) compute degradation speed (page 6 Verbist, K. (2004). Computermodel
    # RUSLE C-factor.)
    # (SG) special case if only one record
    a = (
        p * np.min([W, F], axis=0)
        if len(W) > 1
        else p * np.min([W.values[0], F.values[0]])
    )
    return W, F, a


def compute_Bsb(bdate, edate, a, Bsi):
    """
    Computes harvest remains per unit of area over nodes

    Parameters
    ----------
    'edate' (pd series,timedate): see series ``grid`` in
    :func:`ComputeCFactor` and :func:`prepare_grid`
    'a' (pd series,float): see series ``grid`` in :func:`ComputeCFactor` and
    :func:`prepare_grid`
    'Bsi' (pd series,float): see series ``grid`` in :func:`ComputeCFactor`
    and :func:`prepare_grid`
    'bd' (datetime): begin date of series harvest remains with index har_id

    Returns
    -------
    'Bsi' (pd series, float): harvest remains per unit area on time step i (
    kg/m2)
    """
    Bsb = np.zeros(len(a))
    # (SG) compute remains on middle of computational node
    D = [7] + [
        (
            bdate.iloc[i]
            + (edate.iloc[i] - bdate.iloc[i]) / 2
            - (bdate.iloc[i - 1] + (edate.iloc[i - 1] - bdate.iloc[i - 1]) / 2)
        ).days
        for i in range(1, len(edate), 1)
    ]
    exp = np.exp(-a * D)
    # (SG) compute harvest remains
    for i in range(len(exp)):
        Bsb[i] = Bsb[i - 1] * exp[i] if i != 0 else Bsi.iloc[i] * exp[i]

    return Bsb
