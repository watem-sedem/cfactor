import pandas as pd
from general import load_db, save_db, initialize_files
import numpy as np


def pertube(GWSCOD, parameter, paths_teeltdata, path_destinations):
    """
    Pertube input, pertubatie wordt in een tijdelijk map opgeslagen:
    NOTE:  only maximale bedekking groei (%), maximale effectieve valhoogte (m),
    initiële hoeveelheid oogstresten (kg/ha) en zaaidatum zijn geimplementeerd

    Input:
    :param GWSCODE (int): 'gewascode' of crop
    :param parameter (string): name of parameter that will be pertubed
    :param path_teeltdata (dictionary): tag of inputdata and path
    :path path_destinations (string): name of path where temporary data has to be written

    Output:
    None
    """
    ggg, te, gwscod, hmr, hmt, hmR = load_db(paths_teeltdata)

    GROEP_ID = search_GROEP_ID(GWSCOD, gwscod)

    for i in list(parameter.keys()):
        if i == "max_bedekking(%)":
            ggg = pertube_growth(GROEP_ID, ggg, parameter[i], "bedekking(%)")

        if i == "max_effval(m)":
            ggg = pertube_growth(GROEP_ID, ggg, parameter[i], "effectieve_valhoogte(m)")

        if i == "oogstresten(kg/ha)":
            te = pertube_cropproperties(GROEP_ID, te, parameter[i], "Bsi")

        if i == "zaaidatum":
            te = pertube_cropproperties(GROEP_ID, te, parameter[i], "zaaidatum")

    save_db(ggg, te, gwscod, hmr, hmt, hmR, path_destinations)

    # Get f
    paths_pertubed_teeltdata = initialize_files(path_destinations)

    return paths_pertubed_teeltdata


def search_GROEP_ID(GWSCOD, gwscod):
    """
    Search for the GROEP_ID that is coupled to the GWSCOD

    Input:
    :param GWSCODE (int): 'gewascode' of crop
    :param gwscod (pandas df): holds links between GWSCOD and GROEP_ID's

    Output:
    :return GROEP_ID (int): 'GROEP_ID' of crop
    """

    GROEP_ID = gwscod["groep_id"][gwscod["GWSCOD"] == GWSCOD].values[0]

    return GROEP_ID


def pertube_growth(GROEP_ID, ggg, value, col):
    """
    pertube inputdata in growth curve information, the timeseries growth values
    of coverage (bedekking,%) and heigh (valhoogte,m) are rescaled with the value
    'value'

    Input:
    :param GROEP_ID (int): 'GROEP_ID' of crop
    :param ggg (pandas df): holds link between growth curves and
    :param value (float): value to rescale timeseries (not random, fixed!)
    :param col (string): name of column where values should be changed

    Output:
    None
    """

    cond = [str(i)[-2:] == "{:02d}".format(GROEP_ID) for i in ggg["subgroep_id"].values]
    ggg.loc[cond, col] = ggg.loc[cond, col] / np.max(ggg.loc[cond, col]) * value

    return ggg


def pertube_cropproperties(GROEP_ID, te, value, col):
    """
    pertube inputdata in crop properties information, the property is appointed
    a new value 'value'

    Input:
    :param GROEP_ID (int): 'GROEP_ID' of crop
    :param te(pandas df): holds link between GROEP_IDs,conditions,subgroep_id
                            and crop
    :param value (float): new value appointed to property
    :param col (string): name of column where values should be changed

    Output:
    None
    """

    te.loc[te["GROEP_ID"] == GROEP_ID, col] = value

    return te
