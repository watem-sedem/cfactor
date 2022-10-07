from general import load_GWSCOD
import pdb
import os
import numpy as np
import pandas as pd


def generate_synthetic_parcels(path_sources, path_destinations, GWSCOD=-1):
    """
    Generate a synthetic parcel list (pandas pd) by making a generic parcel with
    the main crop being the subject of interest

    Input:
    :param path_source (string): name of path of source data
    :param path_syntheticparcels (pandas df): name of path synthetic parcel (temp)
    :param GWSCOD_ (int): single gewascode for which synthetic parcel has to be
                        generated. -1: all GWSCOD in gewascode lists
    Output:
    :return parcel_list (pandas df): links synthetic parcels to GWSCOD (as main crop)
    """

    # (SG) load gewascodes and select complete entries
    gwscod = load_GWSCOD(path_sources["gwscod"])
    gwscod = gwscod[gwscod["onvolledig"] == 0]

    # (SG)  initialize synthetic parcel
    parcel_list = []
    ind = 0

    # (SG) if one 'GWSCOD' is defined, only a synthetic parcxel is generated for
    # one GWSCOD
    gwscod = [GWSCOD] if GWSCOD != -1 else gwscod["GWSCOD"].unique()

    # (SG)  generate van synthetische perceelslijst
    for i in gwscod:
        if ~np.isnan(i) & (i != 20) & (i != 21):
            db = prepare_synthetic_parcel(i)
            parcel_list.append(db)
            ind += 1

    parcel_list = pd.concat(parcel_list)
    parcel_list.to_csv(os.path.join(path_destinations, "synthetic_parcels.csv"))

    return parcel_list


def prepare_synthetic_parcel(GWSCOD_):

    """
    Generate synthetic parcel list

    Input:
    :param GWSCOD (int): gewascode

    Output:
    :return parcel_list (pandas df): links synthetic parcels to GWSCOD (as main crop)
    """

    if GWSCOD_ in [60]:

        # (SG) tijdelijk en permanent grasland
        parcel_list = pd.DataFrame(columns=["GWSCOD", "jaar", "perceel_id", "type"])
        parcel_list.loc[:, "type"] = [2] * 2 + [3, 2]  # twee keer twee jaar
        parcel_list.loc[:, "GWSCOD"] = (
            [GWSCOD_] * 2 + [-1] + [GWSCOD_]
        )  # permanent gras (twee keer gewascode ) + tijdelijke (braak + gewasode)
        parcel_list.loc[:, "jaar"] = [2016, 2017] * 2
        parcel_list.loc[:, "perceel_id"] = [10 ** 6 + GWSCOD_] * 2 + [
            10 ** 6 - GWSCOD_
        ] * 2
    # elif GWSCOD in [643]:

    #    parcel_list= pd.DataFrame(columns=["GWSCOD","jaar","perceel_id","type"])
    #    parcel_list.loc[:,"type"] = [3]*2

    #    parcel_list.loc[:,"GWSCOD"] =[GWSCOD]*2
    #    parcel_list.loc[:,"jaar"] = [2016,2017]
    #    parcel_list.loc[:,"perceel_id"] = [GWSCOD]*2
    elif GWSCOD_ in [311, 321]:

        parcel_list = pd.DataFrame(columns=["GWSCOD", "jaar", "perceel_id", "type"])
        parcel_list.loc[:, "type"] = [2] * 2 + [3]
        parcel_list.loc[:, "GWSCOD"] = [GWSCOD_] * 3
        parcel_list.loc[:, "jaar"] = [2016, 2017, 2017]
        parcel_list.loc[:, "perceel_id"] = [10 ** 6 + GWSCOD_] * 3

    # elif GWSCOD in [643]:

    #    parcel_list= pd.DataFrame(columns=["GWSCOD","jaar","perceel_id","type"])
    #    parcel_list.loc[:,"type"] = [3]*2

    #    parcel_list.loc[:,"GWSCOD"] =[GWSCOD]*2
    #    parcel_list.loc[:,"jaar"] = [2016,2017]
    #    parcel_list.loc[:,"perceel_id"] = [GWSCOD]*2

    else:

        parcel_list = pd.DataFrame(columns=["GWSCOD", "jaar", "perceel_id", "type"])
        parcel_list.loc[:, "type"] = [2] * 2
        parcel_list.loc[:, "GWSCOD"] = [GWSCOD_] * 2
        parcel_list.loc[:, "jaar"] = [2016, 2017]
        parcel_list.loc[:, "perceel_id"] = [10 ** 6 + GWSCOD_] * 2

    return parcel_list
