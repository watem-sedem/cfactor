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

To enable these possibilities, the functions were made as generic as possible.
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

The input data can be typically found in publications and reports e.g. ILVO (2019):
TO DO: REF

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
    H = 0.15
    Fc = 0.905
    p = 0.03
    alpha = 5.53
    intitial_crop_residu = 0.12

    slr = cfactor.calculate_slr(begin_date,
                                end_date,
                                rain,
                                temperature,
                                rhm,
                                ri,
                                H,
                                Fc,
                                p,
                                initial_crop_residu)

    print(slr)
    >>>> 0.1384131864957308

The output is a single value that represents the soil loss ratio for the crop and location
between the start and end date.
If you want to know the calculated subfactors of the soil loss ratio, enable the
option `return_subfactors`in the function:

.. code-block:: python

    crop_residu, cc, sr, sc, slr = cfactor.calculate_slr(begin_date,
                                                         end_date,
                                                         rain,
                                                         temperature,
                                                         rhm,
                                                         ri,
                                                         H,
                                                         Fc,
                                                         p,
                                                         initial_crop_residu,
                                                         return_subfactors = True)

We can use the same function To calculate the slr
for different locations and crops. Therefore, we need to change some inputs to numpy
arrays.

.. code-block:: python

    import numpy as np

    begin_date = '2023-06-01'
    end_date = '2023-06-15'
    rain = np.array()
    temperature = np.array()
    rhm = np.array()
    ri = np.array()
    H = np.array()
    Fc = np.array()
    p = np.array()
    intitial_crop_residu =

    slr = cfactor.calculate_slr(begin_date,
                                end_date,
                                rain,
                                temperature,
                                rhm,
                                ri,
                                H,
                                Fc,
                                p,
                                initial_crop_residu)

Of course, you can also use a pandas dataframe to structurize your input data:

+----------+------+-------------+-----+----+---+----+---+---------------------+
| field_id | rain | temperature | rhm | ri | H | Fc | p | initial_crop_residu |
+==========+======+=============+=====+====+===+====+===+=====================+
| 1        |      |             |     |    |   |    |   |                     |
| 2        |      |             |     |    |   |    |   |                     |
| 3        |      |             |     |    |   |    |   |                     |
| 4        |      |             |     |    |   |    |   |                     |
+----------+------+-------------+-----+----+---+----+---+---------------------+


.. code-block:: python

    import pandas as pd

    begin_date = '2023-06-01'
    end_date = '2023-06-15'

    df = pd.read_csv(crop_data_timestamp_x.csv)

    df['slr'] = cfactor.calculated_slr(begin_date,
                                       end_date,
                                       df['rain'],
                                       df['temperature'],
                                       df['rhm'],
                                       df['ri'],
                                       df['H'],
                                       df['Fc'],
                                       df['p'],
                                       df['initial_crop_residu'])

If you run the function above for several timestamps, it is recomended to store the
intermediate results and subfactors too, as some outputs at time t are used in the
calculation of time t+1.

.. code-block:: python

    import pandas as pd

    begin_date = '2023-06-01'
    end_date = '2023-06-15'

    df = pd.read_csv(crop_data_timestamp_x.csv)

    df['slr'] = cfactor.calculated_slr(begin_date,
                                       end_date,
                                       df['rain'],
                                       df['temperature'],
                                       df['rhm'],
                                       df['ri'],
                                       df['H'],
                                       df['Fc'],
                                       df['p'],
                                       df['initial_crop_residu'])
