.. _cfactor:

Cover-Management Factor
=======================

Introduction
------------

The crop or cover management factor (C-factor)  is a measure used in erosion and
(overland) sediment modelling to quantify the effect of of crop cover and soil management
on soil loss. It is typically defined in the context of the Revised Soil Loss Equation
(RUSLE - Renard et al., 1997) [1]_, in which gross erosion for an agricultural parcel is
estimated.

Specifically, the C-factor is based on the concept of deviation
from a standard, in this case defined by a parcel under clean-tilled
continuous-fallow conditions (Renard et al., 1997) [1]_. It can be quantified
as the ratio of the soil loss of a specific parcel with crop cover -
cultivated under specific conditions - and soil loss that would occur on the
same parcel without crop growth (with plowing perpendicular to the
contour lines) (Verbist et al., 2004) [2]_.

The C-factor is calculated as

.. math::
    C = \frac{\sum_i^t{R_i} \cdot SLR_i}{\sum_i^t{R_i}}

with
 - :math:`R_i`: rainfall erosivity factor (:math:`\frac{\text{J.mm}}{\text{m}^2.\text{h.TR}}`) with :math:`\text{TR}`: temporal resolution.
 - :math:`t`: the maximum number of the increments.
 - :math:`SLR`: the soil loss ratio (-)

The soil loss ratio :math:`SLR` is calculated as

.. math::
    SLR = PLU \cdot CC \cdot SC \cdot SR \cdot SM

with
 - :math:`PLU`: Prior-land-use subfactor
 - :math:`CC`: Canopy-cover subfactor
 - :math:`SC`: Surface-cover subfactor
 - :math:`SR`: Surface-roughness subfactor
 - :math:`SM`: Soil-moisture subfactor

In the following paragraphs we discuss these subfactors.

.. note::

The following code is developed in the context on an application of Flanders.
Therefore, not all factors are explicitly defined because data was not available.
We encourage to add formulations using a Pull request.

.. _subfactors:

Subfactors
----------

Prior-land-use subfactor (PLU)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The influence of crop residues and prior soil management practices on soil erosion is
represented by the Prior-land-use subfactor (:math:`PLU`).

According to Verstraeten et al. (2002) [3]_, a value between 0.9 and 1 is a good estimate for
the PLU subfactor for a soil that expierences tillage every year. In this package we do
not take this factor into account due to this reason. More information about this
subfactor and how to calculate it, can be found in Renard et al. (1997) [1]_.

Canopy-cover subfactor (CC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The crop cover or canopy cover subfactor (:math:`CC`) represents the ability of a
crop to reduce the erosivity of falling raindrops on the soil. The effect of the crop
cover is expressed as:

.. math::

    CC = 1-F_c \cdot e^{-0.328H}

With:
 - :math:`F_c (m²/m²)`: the amount of coverage of the soil by the crop

 - :math:`H (m)`: Effective drop height, the average heigt of falling raindrops
 after they have been intercepted by the crop

This subfactor changes considerably during the growth of a crop due to the
increasing crop cover and effective drop height.

Surface-cover subfactor (SC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This subfactor is defined as the erosion limiting influence of the ground cover
by crop residues, stones and non-erodible material in direct contact with the soil

.. math::

    SC = \exp{\left(-b \cdot sp \cdot {\frac{6.096}{Ru}}^{0.08} \right)}


with sp being the amount of land being covered by residu

.. math::

    sp = 10 \cdot (1 - {\exp{\left(-\alpha \cdot B_s \right)}})

with
 - :math:`\alpha`: soil cover in comparison to weight residu (:math:`m^2/kg`)
 - :math:`B_s`: amount of residu per unit of area (:math:`kg/m^2`)

The crop residu `B_s` can be calculated with an exponention decay function:

.. math::
        B_{se} = B_{sb} \cdot \exp{\left(-a \cdot D \right)}


with
 - :math:`B_{se}`: amount of crop residu at end of period (kg dry matter . :math:`m^{-2}`)
 - :math:`B_{sb}`: amount of crop residu at start of period (kg dry matter . :math:`m^{-2}`)
 - :math:`a`: harvest decay coefficient
 - :math:`D`: number of days

The harvest decay coefficient :math:`a` is calculated as

.. math::

    a = p[min(W,F)]

with:

.. math::

    W = \frac{R}{R_0}

and

.. math::

    F = \frac{2(T_a+A)^2 \cdot (T_0+A)^2-(T_a+A)^4}{(T_0+A)^4}

with:

    - :math:`R`: half-monthly rainfall (mm)
    - :math:`R_0`: minimum half-monthly average rainfall (mm)
    - :math:`T_a`: average temperature in half-montlhy period (°F)
    - :math:`T_0`: optimal temperature for decay (°F)
    - :math:`A`: coefficient used to express the shape of the decay function
      as a function of temperature.

Surface-roughness subfactor (SR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The surface roughness :math:`SR` is caluclated as

.. math::

    SR = \exp(−0.026 \cdot (R_u-6.096))


With :math:`R_u` is a measure for roughness of a parcell (mm).
:math:`R_u` (-) is calculated by:

.. math::

    R_u = 6.096+(D_r \cdot (R_i-6.096))

The final roughness is referred to as :math:`r_{ii}`, i.e. 6.096.
The initial roughness is crop dependent (soil preparation dependent).

The roughness decay function :math:`D_r` is defined as:

.. math::

    D_r = \exp{\left(0.5 \cdot \frac{-0.14}{25.4}P_t + 0.5 \cdot \frac{-0.012}{17.02}EI_t \right)}

with

- :math:`P_t`: the cumulative rainfall (in mm)
- :math:`EI_t`: the cumulative rainfall erosivity (in :math:`MJ \cdot mm \cdot ha^{-1} \cdot year^{-1}`)

Under the influence of precipitation, the roughness of an agricultural field,
left undisturbed, will systematically decrease until an (average) minimum roughness
of 6.096 mm (0.24 inches) is reached. The decrease function :math:`D_r` is defined to
compute this decrease.

Soil-moisture subfactor (SM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Antecedent soil moisture has a substantial influence on infiltration and
runoff, and thus soil erosion. As this package was developed in the context of
flanders, we assume this value is equal to 1 (Verbist et al., 2004) [2]_.

References
----------

.. [1] Renard, K.G., Foster, G.R., Weesies, G.A., McCool, D.K., Yoder, D.C.,
 1997, Predicting soil erosion by water: a guide to conservation planning with
 the revised universal soil loss equation (RUSLE), Agriculture Handbook. U.S.
 Department of Agriculture, Washington.
 https://www.ars.usda.gov/ARSUserFiles/64080530/RUSLE/AH_703.pdf

.. [2] Verbist, K., Schiettecatte, W., Gabriels, D., 2004, Eindrapport.
 Computermodel RUSLE c-factor. Universiteit Gent, Gent. (In dutch)

.. [3] Verstraeten, G., Van Oost, K., Van Rompaey, A., Poesen, J. & Govers, G. 2002.
 Integraal land- en waterbeheer in landelijke gebieden met het oog op het beperken
 van erosie en modderoverlast (proefproject gemeente Gingelom). Ministerie van de
 Vlaamse Gemeenschap, Departement Leefmilieu en Infrastructuur, AMINAL,
 Afdeling Land, Brussel, 69p
