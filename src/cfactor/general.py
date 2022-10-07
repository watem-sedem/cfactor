import os
import pandas as pd
from cnws import utils
import geopandas as gpd
from pathlib import Path
from copy import deepcopy
import numpy as np


def initialize_files(path_sources, path_resmap=""):
    """
    initialize inputfiles for cfactor scripts

    Input:
    :param path_sources (string): name of path to where inputfiles are

    Output:
    :param path_teeltdata (dictionary): tag of inputdata and path
    """

    paths_teeltdata = {}

    paths_teeltdata["ggg"] = os.path.join(path_sources, "groeicurves.csv")
    paths_teeltdata["te"] = os.path.join(path_sources, "teelteigenschappen.csv")
    paths_teeltdata["gwscod"] = os.path.join(path_sources, "GWSCOD_groep_id.csv")
    paths_teeltdata["fname_halfmonthly_rain"] = os.path.join(
        path_sources, "Halfmonthlymean_rainfall.csv"
    )
    paths_teeltdata["fname_halfmonthly_temp"] = os.path.join(
        path_sources, "Halfmonthlymean_temperature.csv"
    )
    paths_teeltdata["fname_halfmonthly_R"] = os.path.join(
        path_sources, "Monthlymean_R.csv"
    )

    return paths_teeltdata


def prepare_parcel_input(parcelshapes, catchmentshape, path_tempfolder):
    """
    Prepare input for cfactor model, steps:
    1. clip 'perceelskaart jaar X, Y, Z' shape to catchment shape
    2. convert clipped 'perceelskaart X, Y, Z' shape to parcel list (input cfactor script)

    Input:
    :param parcelshapes (dict): holding year coupled to the name of shapefiles
                                of 'perceelskaart' as defined by VPO (vb. landbouw
                                 en visserij shape file)
    :param catchmentshape (string): name of shape file which holds the extent of
                                    the catchment
    :param path_tempfolder (string): name of path where temporary files can be written to

    Output:
    :return parcel_list (pandas pd): a list of parcels holding information on
                                    the crops on the field coupled to a unique
                                    id. Used as input format for script cfactor.
    """
    clipped_parcelshapes = clip_parcels(parcelshapes, catchmentshape, path_tempfolder)

    parcel_list = convert_parcelshapes_to_parcellist(clipped_parcelshapes)

    return parcel_list


def clip_parcels(parcelshapes, catchmentshape, path_tempfolder):
    """
    Input:
    :param parcelshapes (dict): holding year coupled to the name of shapefiles
                                of 'perceelskaart' as defined by VPO (vb. landbouw
                                 en visserij shape file)
    :param catchmentshape (string): name of shape file which holds the extent of
                                    the catchment
    :param path_tempfolder (string): name of path where temporary files can be written to

    Output:
    :return out_Shp (dict): holding year coupled to the name of catchment-clipped
                            shapefiles of 'perceelskaart' as defined by VPO (vb.
                            landbouw en visserij shape file)
    """

    out_Shp = {}

    # (SG) extract name for tempfiles
    fname = os.path.basename(catchmentshape)
    fname = fname[: fname.index(".")]

    for i in list(parcelshapes.keys()):

        out_Shp[i] = Path(os.path.join(path_tempfolder, "fname_parcel_clip%s.shp" % i))
        # (SG) clip shape to Maarkedal catchment
        utils.clip_shp(parcelshapes[i], out_Shp[i], catchmentshape)

    return out_Shp


def convert_parcelshapes_to_parcellist(parcelshapes):
    """
    Input:
    :param parcelshapes (dict): holding year coupled to the name of (catchment-clipped)
                            shapefiles of 'perceelskaart' as defined by VPO (vb.
                            landbouw en visserij shape file)
    Output:
    :return parcel_list (pandas pd): a list of parcels holding information on
                                    the crops on the field coupled to a unique
                                    id. Used as input format for script cfactor.
    """

    cond = True

    for j in list(parcelshapes.keys()):

        temp = gpd.read_file(parcelshapes[j])[
            ["CODE_OBJ", "REF_ID", "GWSCOD_H", "GWSCOD_V", "GWSCOD_N"]
        ].drop_duplicates()

        for i in ["GWSCOD_H", "GWSCOD_V", "GWSCOD_N"]:
            tempx = temp[["CODE_OBJ", "REF_ID", i]].rename(columns={i: "GWSCOD"})
            if "_H" in i:
                tempx["type"] = 2  #'hoofdteelt'
            elif "_V" in i:
                tempx["type"] = 1  #'voorteelt'
            else:
                tempx["type"] = 3  #'nateelt'
            tempx["jaar"] = int(j)

            if cond == True:
                parcel_list = deepcopy(tempx)
                cond = False
            else:
                parcel_list = parcel_list.append(tempx)

    # appoint id
    temp = parcel_list[["REF_ID"]].drop_duplicates()
    temp["perceel_id"] = np.arange(0, len(temp), 1)
    parcel_list = parcel_list.merge(temp, on=["REF_ID"], how="left")[
        ["REF_ID", "CODE_OBJ", "perceel_id", "jaar", "type", "GWSCOD"]
    ]
    parcel_list = parcel_list.sort_values(["perceel_id", "jaar", "type"])
    parcel_list = parcel_list.loc[~parcel_list["GWSCOD"].isnull()]

    return parcel_list


def search_GROEP_ID(GWSCOD, gwscod):
    """
    Search for the GROEP_ID that is coupled to the GWSCOD

    Input:
    :param GWSCODE (int): 'gewascode' of crop
    :param gwscod (pandas df): holds links between GWSCOD and GROEP_ID's

    Output:
    :return GROEP_ID (int): 'GROEP_ID' of crop
    """

    GROEP_ID = gwscod["GROEP_ID"][gwscod["GWSCOD"] == GWSCOD].values[0]

    return GROEP_ID


def load_db(paths_teeltdata):

    """
    load data

    Input:
    :param path_source (string): name of path of source data

    Output:
    :return gwscod (pandas df): holds links between GWSCOD and GROEP_ID's
    :return ggg (pandas df): holds link between growth curves and
    :return te (pandas df): holds link between GROEP_IDs,conditions,subGROEP_IDs
                            and crop properties

    """
    ggg = pd.read_csv(paths_teeltdata["ggg"])
    te = pd.read_csv(paths_teeltdata["te"])
    gwscod = pd.read_csv(paths_teeltdata["gwscod"])
    hmr = pd.read_csv(paths_teeltdata["fname_halfmonthly_rain"])
    hmt = pd.read_csv(paths_teeltdata["fname_halfmonthly_temp"])
    hmR = pd.read_csv(paths_teeltdata["fname_halfmonthly_R"])

    return ggg, te, gwscod, hmr, hmt, hmR


def load_GWSCOD(path_source):

    gwscod = pd.read_csv(path_source, encoding="latin8")

    return gwscod


def save_db(ggg, te, gwscod, hmr, hmt, hmR, path_destinations):
    """
    save data to a temporary folder

    Input:
    :param ggg (pandas df): holds link between growth curves and
    :param te(pandas df): holds link between GROEP_IDs,conditions,subGROEP_IDs
                            and crop properties

    Output:
    None
    """
    ggg.to_csv(os.path.join(path_destinations, "groeicurves.csv"))
    te.to_csv(os.path.join(path_destinations, "teelteigenschappen.csv"))
    gwscod.to_csv(os.path.join(path_destinations, "GWSCOD_groep_id.csv"))
    hmr.to_csv(os.path.join(path_destinations, "Halfmonthlymean_rainfall.csv"))
    hmt.to_csv(os.path.join(path_destinations, "Halfmonthlymean_temperature.csv"))
    hmR.to_csv(os.path.join(path_destinations, "Monthlymean_R.csv"))

    return None


def create_dir(resmap, L):
    """ create directory for output to which results are written to

    Parameters
    ----------
        'resmap' (str): name/path of main output directory

    Returns
    -------
        'L' (list): list of names which have to be written under res directory
    """

    for i in range(len(L)):
        if not os.path.exists(os.path.join(resmap, L[i])):
            os.makedirs(os.path.join(resmap, L[i]))
