.. _gettingstarted:

Getting started
===============

The C-factor scripts can be used to calculate all the subfactors of
:ref:`the Soil Loss Ratio <subfactors>`. There is also a function provided to calculate
the C-factor based on the calculated soil loss ratios.

In this tutorial we first give an overview on how to calculate the soil loss ratios.

Calculating the SLR
-------------------

Three possibilities exist to calculate the soil loss ratio (SLR):

- for a single period and a single crop/location
- for different crops/locations on the same time
- for a timeseries of a crop on a given location

To enable these possibilities, the functions in this package were made as generic as
possible.
They work both with float values as with (one-dimensional) numpy arrays.

We will explore these three possibilities in the following tutorial.

For all possibilities we need input data from the environment, the crop and soil.
Following data is needed for every location/crop and period:

- startdate of the period
- enddate of the period
- total amount of rain during the period
- average temperature (°C) during the period
- Effective drop height (m), an estimate of average height between rainfall
  capture by crop and soil
- The soil cover by the crop (in %)
- The maximum decay speed
- The crop residu present at the soil at the start of the period,
- cumulative rainfall erosivity (MJ.mm/ha.year)
- ri: initial soil roughness?

The input data can be typically found in publications and reports (Cecelja et al., 2019)

Start with importing the package

.. code-block:: python

    from cfactor import cfactor

.. note::
    Make sure to activate the conda environment ``conda activate cfactor`` with the
    cfactor package installed (see :ref:`the installation page <installation>`)

As stated before, we can calculate the soil loss ratio for a single location and a
single period. For example:

.. code-block:: python

    begin_date = '2023-06-01'
    end_date = '2023-06-15'
    rain = 35.41
    temperature = 18.48
    rhm = 109.22
    ri = 6.1
    h = 0.15
    fc = 0.905
    p = 0.03
    alpha = 5.53
    initial_crop_residu = 5000
    mode = 'space'

    crop_residu, harvest_decay_coefficient, \
    days, soil_roughness, crop_cover, \
    surface_roughness, soil_cover, \
    soil_loss_ratio = cfactor.calculate_soil_loss_ratio(begin_date,
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
                                                        mode)

    print(soil_loss_ratio)
    >>> 0.1384131864957308

The output contains all subfactors and the soil_loss_ratio calculated for the crop
and location between the start and end date.

We can use the same function To calculate the SLR and its subfactors
for different locations and crops. Therefore, we need to change some inputs to numpy
arrays.

.. code-block:: python

    import numpy as np

    begin_date = '2023-06-01'
    end_date = '2023-06-15'
    rain = np.array([35.41, 33.95, 28.51, 26.76])
    temperature = np.array([18.48, 17.23, 18.86, 1.47])
    rhm = np.array([109.22, 145.195, 53.505, 28.47])
    ri = np.array([6.1, 10.2, 6.096, 6.1])
    h = np.array([0.15, 0.015, 0.13, 0])
    fc = np.array([0.905, 0.875, 0.725, 0.405])
    p = np.array([0.03, 0.01, 0.05, 0.03])
    alpha = np.array([5.53, 5.53, 9.21, 23.03])
    initial_crop_residu = np.array([5000, 4500, 150, 3500])
    mode = 'space'

    crop_residu, harvest_decay_coefficient, \
    days, soil_roughness, crop_cover, \
    surface_roughness, soil_cover, \
    soil_loss_ratio = cfactor.calculate_soil_loss_ratio(begin_date,
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
                                                        mode)

    print(crop_residu)
    >>>[3523.15134387, 4015.53444796,   83.34852713, 2807.18026399]

    print(soil_loss_ratio)
    >>>[0.00521205, 0.00534468, 0.19433155, 0.01798863]

Of course, you can also use a pandas dataframe to structurize your input data:

+----------+-------+-------------+---------+-------+-------+-------+------+---------------------+
| field_id | rain  | temperature | rhm     | ri    | H     | Fc    | p    | initial_crop_residu |
+==========+=======+=============+=========+=======+=======+=======+======+=====================+
| 1        | 35.41 | 18.48       | 109.22  | 6.1   | 0.15  | 0.905 | 0.03 | 5000                |
+----------+-------+-------------+---------+-------+-------+-------+------+---------------------+
| 2        | 33.95 | 17.23       | 145.195 | 10.2  | 0.015 | 0.875 | 0.01 | 4500                |
+----------+-------+-------------+---------+-------+-------+-------+------+---------------------+
| 3        | 28.51 | 18.86       | 53.505  | 6.096 | 0.13  | 0.725 | 0.05 | 150                 |
+----------+-------+-------------+---------+-------+-------+-------+------+---------------------+
| 4        | 26.76 | 14.47       | 28.47   | 6.1   | 0     | 0.405 | 0.03 | 3500                |
+----------+-------+-------------+---------+-------+-------+-------+------+---------------------+

When using a pandas dataframe, we can calculate the soil loss ratio and the subfactors
like the example below:

.. code-block:: python

    import pandas as pd

    begin_date = '2023-06-01'
    end_date = '2023-06-15'

    df = pd.read_csv('crop_data_timestamp_x.csv')

    df[['crop_residu', 'harvest_decay_coefficient', \
    'days', 'soil_roughness', 'crop_cover', \
    'surface_roughness', 'soil_cover', \
    'soil_loss_ratio']] = cfactor.calculated_slr(begin_date,
                                               end_date,
                                               df['rain'],
                                               df['temperature'],
                                               df['rhm'],
                                               df['ri'],
                                               df['H'],
                                               df['Fc'],
                                               df['p'],
                                               df['initial_crop_residu'],
                                               df['alpha'],
                                               mode='space')

However, we can use the functions in the package also to calculate timeseries for every
subfactor for a single crop on a certain location. To do this, we need different input.

.. code-block:: python

    begin_date = np.array(['2016-01-01', '2016-01-15', '2016-02-01', '2016-02-15'])
    end_date = np.array(['2016-01-15', '2016-02-01', '2016-02-15', '2016-03-01'])
    rain = np.array([35.41, 10.2, 28.51, 26.76])
    temperature = np.array([18.48, 17.23, 18.86, 1.47])
    rhm = np.array([109.22, 145.195, 53.505, 28.47])
    ri = np.array([6.1, 10.2, 6.096, 6.1])
    h = np.array([0.15, 0.015, 0.13, 0])
    fc = np.array([0.905, 0.875, 0.725, 0.405])
    p = np.array([0.03, 0.01, 0.05, 0.03])
    alpha = np.array([5.53, 5.53, 9.21, 23.03])
    initial_crop_residu = 5000
    mode = 'time'

    crop_residu, harvest_decay_coefficient, \
    days, soil_roughness, crop_cover, \
    surface_roughness, soil_cover, \
    soil_loss_ratio = cfactor.calculate_soil_loss_ratio(begin_date,
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
                                                        mode)


References
----------
Cecelja, A., Ruysschaert, G., Vanden Nest, T. & Deproost, P. (2019). Verzamelen van data voor de
verdere verfijning van de RUSLE gewas- en bedrijfsvoeringsfactor C voor de Vlaamse teeltpraktijken
en erosiebestrijdingsmaatregelen. Rapport in opdracht van Departement Omgeving. 28p.
