import datetime
import os
import subprocess
import sys
from copy import deepcopy
from datetime import date, timedelta

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from cfactor.cfactor import rii
from cfactor.util import create_dir


def generate_datetime_instance(dd, mm, yyyy):
    """ "
    #TODO

    Parameters
    ----------
    Returns
    -------
    """
    date = datetime.strptime(str(dd) + str(mm) + str(yyyy), "%d%m%Y")
    return date


def load_data(fname_input):
    """ "
    #TODO

    Parameters
    ----------
    Returns
    -------
    """
    inputdata = {}
    for i in list(fname_input.keys()):
        inputdata[i] = pd.read_csv(fname_input[i], encoding="latin8")
    return inputdata


def generate_report(fname_inputs, resmap, fname_output, GWSCOD=-1):
    """ "
    #TODO

    Parameters
    ----------
    Returns
    -------
    """
    inputdata = load_data(fname_inputs)

    gwscod_id = inputdata["gwscod"]
    te = inputdata["te"]
    ggg = inputdata["ggg"]
    ggg = ggg[ggg["dagen_na"] != 2000]  # (SG) delete synth. dagen na of 2000

    lo = latexobject(os.path.join(resmap, fname_output))
    lo.writeheader()

    # selecteer één Specifieke gewascodes
    if GWSCOD != -1:
        gwscod_id = gwscod_id[gwscod_id["GWSCOD"] == GWSCOD]

    create_dir(resmap, ["temp"])

    for i in gwscod_id["groep_id"].unique():
        lo = initialiseer_fiches_teelt(lo, gwscod_id, i)
        lo = compileer_teeltwisseling(lo, te, ggg, i, resmap)

    lo.close()
    lo.compile(resmap)


def initialiseer_fiches_teelt(lo, gwscod_id, i):
    """ "
    #TODO

    Parameters
    ----------
    Returns
    -------
    """
    cond = gwscod_id["groep_id"] == i
    groep_id = gwscod_id.loc[cond, "groep_id"].values[0]
    groep_name = gwscod_id.loc[cond, "groep_naam"].values[0]
    gwscod = gwscod_id.loc[cond, "GWSCOD"].values.flatten().tolist()
    gwsnam = gwscod_id.loc[cond, "GWSNAM"].values.flatten().tolist()
    meerjarig = gwscod_id.loc[cond, "meerjarige_teelt"].values[0]
    groente = gwscod_id.loc[cond, "groente"].values[0]
    groenbedekker = gwscod_id.loc[cond, "groenbedekker"].values[0]

    gwsnam = [fix_string(i) for i in gwsnam]
    groep_name = fix_string(groep_name)

    # (SG) write title
    lo.init_crop(groep_id, groep_name)

    # (SG) write GWSNAM and GWSCOD that are coupled
    lo.write_gws(gwsnam, gwscod, meerjarig, groenbedekker, groente)
    return lo


def compileer_teeltwisseling(lo, te, ggg, i, resmap):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    # (SG) extract different scenario's
    te_i = te.loc[te["groep_id"] == i]
    for j in te_i["subgroep_id"].unique():
        n_groeps = len(te_i)
        lo, referenties, opmerkingen = write_eigenschappen(
            lo, te_i[te_i["subgroep_id"] == j], j, n_groeps
        )
        lo = write_gewasgroeicurve(lo, ggg, j, resmap)
        cmd = r" \textbf{Referenties:} %s" % referenties
        lo.commandnl(cmd)
        opmerkingen = opmerkingen if str(opmerkingen) != "nan" else "geen"
        cmd = r" \textbf{Opmerkingen?} %s" % opmerkingen
        lo.commandnl(cmd)
        lo.newpage()
    return lo


def write_gewasgroeicurve(lo, ggg, subgroep_id, resmap):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    plot_growth_curve(subgroep_id, ggg, resmap)
    fname = r"temp/%i" % subgroep_id
    lo.add_figure(fname)
    return lo


def write_eigenschappen(lo, eigenschappen, j, n_groeps):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    subgroep_id = j
    alpha = eigenschappen["alpha"].values[0]
    p = eigenschappen["p"].values[0]
    Bsi = eigenschappen["Bsi"].values[0]
    Ri = eigenschappen["Ri"].values[0]
    SCi = eigenschappen["Spi"].values[0]
    D = eigenschappen["D"].values[0]
    zaaidatum = eigenschappen["zaaidatum"].values[0]
    zaaidatum = str(int(zaaidatum))[4:]
    zaaidatum = zaaidatum[2:] + "/" + zaaidatum[0:2]
    oogstdatum = eigenschappen["oogstdatum"].values[0]
    oogstdatum = str(int(oogstdatum))[4:]
    oogstdatum = oogstdatum[2:] + "/" + oogstdatum[0:2]
    voorwaarde = eigenschappen["voorwaarde"].values[0]
    teeltwisseling = eigenschappen["teeltwisseling"].values[0]
    referenties = eigenschappen["referenties"].values[0]
    opmerkingen = eigenschappen["opmerking"].values[0]

    if isinstance(teeltwisseling, np.float):
        teeltwisseling = "Hoofdteelt, voorteelt en nateelt"

    title = (
        teeltwisseling
        if isinstance(voorwaarde, np.float)
        else "%s (%s)" % (teeltwisseling, voorwaarde)
    )

    if n_groeps > 1:
        lo.writesubsection(title + r" (subgroep\_id %i)" % int(j))

    cmd = r" \textbf{%s}: %s " % ("Zaaidatum (dd/mm)", zaaidatum)
    lo.commandnl(cmd)
    cmd = r" \textbf{%s}: %s " % ("Oogstdatum (dd/mm)", oogstdatum)
    lo.commandnl(cmd)
    cmd = r" \textbf{Oogstresten}"
    lo.commandnl(cmd, vspace=0.05)
    if np.isnan(Bsi):
        cmd = r" \tab %s: /" % (r"Initi\"{e}le hoeveelheid (kg ha$^{-1}$)")
    else:
        cmd = r" \tab %s: %.2f" % (r"Initi\"{e}le hoeveelheid (kg ha$^{-1}$)", Bsi)
    lo.commandnl(cmd, vspace=0.05)
    if np.isnan(p):
        cmd = r" \tab %s: /" % (r"Afbraakcoefficient (-)")
    else:
        cmd = r" \tab %s: %.2f" % (r"Afbraakcoefficient (-)", p)
    lo.commandnl(cmd, vspace=0.05)
    if np.isnan(alpha):
        cmd = r" \tab %s: /" % (r"Bodembedekking (m$^2$ kg$^{-1}$)")
    else:
        cmd = r" \tab %s: %.2f" % (r"Bodembedekking (m$^2$ kg$^{-1}$)", alpha)
    lo.commandnl(cmd, vspace=0.05)
    if np.isnan(SCi):
        cmd = r" \tab %s: /" % (r"Initieel percentage bedekking (\%)")
    cmd = r" \tab %s: %i" % (r"Initieel percentage bedekking (\%)", int(SCi))
    lo.commandnl(cmd, vspace=0.05)
    if np.isnan(D):
        cmd = r" \tab %s: /" % (r"Halfwaarde tijd (dagen)")
    else:
        cmd = r" \tab %s: %i" % (r"Halfwaarde tijd (dagen)", D)
    lo.commandnl(cmd, vspace=0.05)
    if np.isnan(Ri):
        cmd = r" \tab %s: /" % (r"Initi\"{e}le bodemruwheid (mm)")
    else:
        cmd = r" \textbf{%s}: %.2f" % (r"Initi\"{e}le bodemruwheid (mm)", Ri)
    lo.commandnl(cmd, vspace=0.05)

    cmd = r" \textbf{Gewasgroeicurve subgroep\_id %i:}" % subgroep_id
    lo.command(cmd)
    return lo, referenties, opmerkingen


def fix_string(string):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    string = string.replace("_", " ")
    return string


class latexobject:
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, fname):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        self.fname = fname + ".tex"

    def writeheader(self):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        packages = [
            r"\usepackage{graphicx}",
            r"\usepackage{float}",
            r"\usepackage[ampersand]{easylist}",
            r"\usepackage{multicol}",
            r"\usepackage[raggedright]{titlesec}",
            r"\usepackage{amssymb}",
        ]
        with open(self.fname, "w") as f:
            f.write(r"\documentclass{article}")
            f.write("\n")
            f.write(r"\title{Appendix A}")
            f.write("\n")
            self.writepackages(f, packages)
            f.write(r"\date{}")
            f.write("\n")
            f.write(r"\begin{document}")
            f.write("\n")
            f.write(r"\setlength\parindent{0pt}")
            f.write("\n")
            f.write(r"\newcommand\tab[1][1cm]{\hspace*{#1}}")
            f.write("\n")

    def writepackages(self, f, packages):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        for i in packages:
            f.write(r"" + i)
            f.write("\n")
        return f

    def add_figure(self, fname):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        cmd = (
            r"\begin{center} \begin{figure}[H] \includegraphics[width=12.5cm]{%s.png} "
            r"\end{figure} \end{center}" % fname.replace("\\", "/")
        )
        self.command(cmd)

    def writesubsection(self, title):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        self.command(r"\subsection{%s}" % title.capitalize())

    def newpage(self):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        self.command(r"\newpage")

    def init_crop(self, groep_id, groep_name):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        cmd = r"\section{%s (groep\_id %i)}" % (groep_name.capitalize(), groep_id)
        self.command(cmd)

    def write_gws(self, gwsnam, gwscod, meerjarig, groenbedekker, groente):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        cmd = r"\textbf{Van toepassing op gewasnamen (en codes):} " + " , ".join(
            [gwsnam[i] + " (" + str(int(gwscod[i])) + ")" for i in range(len(gwsnam))]
        )

        self.command(cmd)

        props = {
            "Meerjarig": meerjarig,
            "Groenbedekker": groenbedekker,
            "Groente": groente,
        }

        self.write_checklist(props, vspace=0)

    def write_checklist(self, props, vspace=0.1):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        cmd = (
            r"\begin{multicols}{3} \begin{itemize} "
            + " ".join(
                [
                    r"%s %s" % (r"\item[$\boxtimes$]", i)
                    if props[i] == 1
                    else "%s %s" % (r"\item[$\square$]", i)
                    for i in list(props.keys())
                ]
            )
            + r" \end{itemize} \end{multicols}"
        )
        self.command(cmd)

    def command(self, cmd):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        with open(self.fname, "a+") as f:
            f.write(cmd)
            f.write(" \n ")

    def commandnl(self, cmd, vspace=0.1):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        with open(self.fname, "a+") as f:
            f.write(cmd)
            f.write(r" \vspace{%.2fcm} \\" % vspace)
            f.write(" \n ")

    def close(self):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        self.command(r"\end{document} \n")

    def compile(self, fmap):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        cwd = os.getcwd()
        os.chdir(fmap)
        proc = subprocess.Popen(["pdflatex", self.fname])
        proc.communicate()
        os.chdir(cwd)


def plot_growth_curve(subgroep_id, data, resmap):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    from matplotlib import rc

    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
    # for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc("text", usetex=True)

    cond = data["subgroep_id"] == subgroep_id
    Fc = data.loc[cond, "bedekking(%)"]
    H = data.loc[cond, "hoogte(m)"]
    He = data.loc[cond, "effectieve_valhoogte(m)"]
    x = data.loc[cond, "dagen_na"]

    fig, ax = plt.subplots(figsize=[12.5, 7])
    ax2 = ax.twinx()
    l1 = ax2.plot(x, Fc, label=r"Bedekking (\%)", ls="-", lw=2, color="#00AFBB")
    l2 = ax.plot(x, He, label="Effectieve valhoogte (m)", lw=2, color="#FC4E07")
    if np.sum(~np.isnan(H)) != 0:
        l3 = ax.plot(x, H, label="Hoogte (m)", ls="-", lw=2, color="#E7B800")
        lns = [l1, l2, l3]
    else:
        lns = [l1, l2]

    lns = [i[0] for i in lns]
    labs = None  # [l.get_label() for l in lns]

    ylim = ax.get_ylim()[-1]
    ax.set_ylim([0, ylim])
    ax.set_yticks([0, ylim / 2, ylim])
    ax2.set_ylim([0, 100])
    ax2.set_yticks([0, 50, 100])
    # ax.set_xticklabels(np.arange(0, len(ax.get_xticks())))
    xlim = ax.get_xlim()[-1]
    if xlim < 49:
        xticks = np.arange(0, int(np.ceil(xlim)), 7)
        xticklabels = [
            "%i \n  (%i)" % (i, i / 7) for i in np.arange(0, int(np.ceil(xlim)), 7)
        ]
    elif xlim < 49 * 2:
        xticks = np.arange(0, int(np.ceil(xlim)), 14)
        xticklabels = [
            "%i \n (%i)" % (i, i / 7) for i in np.arange(0, int(np.ceil(xlim)), 14)
        ]
    elif xlim < 49 * 4:
        xticks = np.arange(0, int(np.ceil(xlim)), 28)
        xticklabels = [
            "%i \n (%i)" % (i, i / 7) for i in np.arange(0, int(np.ceil(xlim)), 28)
        ]
    elif xlim < 49 * 8:
        xticks = np.arange(0, int(np.ceil(xlim)), 56)
        xticklabels = [
            "%i \n (%i)" % (i, i / 7) for i in np.arange(0, int(np.ceil(xlim)), 56)
        ]
    elif xlim < 49 * 16:
        xticks = np.arange(0, int(np.ceil(xlim)), 112)
        xticklabels = [
            "%i \n (%i)" % (i, i / 7) for i in np.arange(0, int(np.ceil(xlim)), 112)
        ]
    elif xlim < 49 * 32:
        xticks = np.arange(0, int(np.ceil(xlim)), 224)
        xticklabels = [
            "%i \n (%i)" % (i, i / 7) for i in np.arange(0, int(np.ceil(xlim)), 224)
        ]
    elif xlim < 49 * 64:
        xticks = np.arange(0, int(np.ceil(xlim)), 448)
        xticklabels = [
            "%i \n (%i)" % (i, i / 7) for i in np.arange(0, int(np.ceil(xlim)), 448)
        ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.legend(lns, labs, loc=0, prop={"size": 16})

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax2.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlabel(r"Dagen (weken) na inzaai", fontsize=18)
    ax.set_ylabel(r"Hoogte (m)", fontsize=18)
    ax2.set_ylabel(r"Bedekking (\%)", fontsize=18)
    plt.savefig(r"" + os.path.join(resmap, "temp", "%i.png" % subgroep_id), dpi=500)
    # plt.savefig(os.path.join("gewasgroeicurves_plots","%i-%i-%s_%s_%s.png"
    # %(tag,subgroep_id,GWSNAM,voorwaarde,teeltwisseling)))
    plt.close()


def compute_near_identical_parcels(percelen, jaar, output_map):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    # (SG) als de betrefferende percelenkaarten nog niet ingeladen zijn als dataframe:
    # doen
    wcond = False
    if isinstance(percelen[jaar], str):
        fnames = {}
        wcond = True
        for j in list(percelen.keys()):
            fnames[j] = percelen[j]
            percelen[j] = gpd.read_file(fnames[j])

    # (SG) check if coupling has not already been done in prior runs
    # (SG) if yes, don't overwrite
    if "NR_po" not in percelen[jaar]:
        # (SG) bepaal overlap percelen prior jaar
        intersection = near_identical_parcels(
            percelen[jaar], percelen[jaar - 1], output_map, "pr"
        )
        # (SG) ... and couple codes if area overlaps 80 %
        temp = deepcopy(percelen[jaar].loc[:, ["NR"]])
        temp = temp.merge(
            intersection[["normalized_overlap", "NR", "NR_pr"]], on=["NR"], how="left"
        ).rename(columns={"normalized_overlap": "overlap_prior"})
        percelen[jaar] = percelen[jaar].merge(temp, on="NR", how="left")

        # (SG) bepaal overlap percelen posterior jaar
        intersection = near_identical_parcels(
            percelen[jaar], percelen[jaar + 1], output_map, "po"
        )
        # (SG) ... and couple codes if area overlaps 80 %
        temp = deepcopy(percelen[jaar].loc[:, ["NR"]])
        temp = temp.merge(
            intersection[["normalized_overlap", "NR", "NR_po"]], on=["NR"], how="left"
        ).rename(columns={"normalized_overlap": "overlap_posterior"})
        percelen[jaar] = percelen[jaar].merge(temp, on="NR", how="left")

        # (SG) koppel perceelsnummer aan dataframe van prior en posterior jaar
        temp = deepcopy(percelen[jaar])[["NR", "NR_pr"]]
        temp = temp.rename(
            columns={
                "NR": "NR_po",
                "NR_pr": "NR",
            }
        )[["NR", "NR_po"]]
        percelen[jaar - 1] = percelen[jaar - 1].merge(temp, on="NR", how="left")
        temp = deepcopy(percelen[jaar])[["NR", "NR_po"]]
        temp = temp.rename(columns={"NR": "NR_pr", "NR_po": "NR"})[["NR", "NR_pr"]]
        percelen[jaar + 1] = percelen[jaar + 1].merge(temp, on="NR", how="left")

        # (SG) sort for easy look-up
        percelen[jaar + 1] = percelen[jaar + 1].sort_values("NR_pr")
        percelen[jaar - 1] = percelen[jaar - 1].sort_values("NR_po")

        # (SG) write percelenkaarten back to disk
        if wcond:
            for j in list(percelen.keys()):
                percelen[j].to_file(fnames[j])
    return percelen


def near_identical_parcels(parcels1, parcels2, output_folder, tag, perc_overlap=80):
    """compute overlap and intersection, filter on 80 % perc overlap

    Parameters
    ----------
        'parcels1' (gpd df ): parcel gpd df from shapefile each record holding parcel
        NR and its geometry, for year at interest
        'parcels2' (gpd df ): parcel gpd df from shapefile each record holding parcel
         NR and its geometry, for year prior or posterior to year parcels1
        'output_folder' (string): path to which files have to be written
        'tag' (string): identifier whether it is a prior (pr) or posterior year (po)
        'perc_overlap' (int): threshold of minimal overlap

    Returns
    -------
        'intersection" (gpd df): dataframe holding intersects between parcels1 and
        parcels2  based on 80 % overlap intersection
        (normalized with area of parcels1)
    """
    # (SG) name of files on which saga has to perform analysis
    fname1 = os.path.join(output_folder, "parcels_intersection.shp")
    fname2 = os.path.join(output_folder, "parcels_%s_intersection.shp" % tag)
    fname_temp = os.path.join(output_folder, "temp.shp")

    # (SG) write to disk
    parcels1[["NR", "geometry"]].to_file(fname1)
    parcels2["NR_%s" % tag] = parcels2["NR"]
    parcels2[["NR_%s" % tag, "geometry"]].to_file(fname2)

    # (SG) saga intersection execution
    try:
        import CNWS
    except IOError:
        sys.exit(
            "[Cfactor scripts ERROR] CNWS script not found, check, terminating "
            "computation Cfactor"
        )

    CNWS.saga_intersection(
        '"' + fname1 + '"', '"' + fname2 + '"', '"' + fname_temp + '"'
    )
    intersection = gpd.read_file(fname_temp)

    # (SG) couple considered parcels from this year and prior year (only consider
    # largest intersect)
    intersection["area_intersection"] = intersection["geometry"].area

    # intersection = intersection.rename({"NR_p":})
    # (SG) keep largest intersect area per CODE_OBJ
    intersection = (
        intersection.groupby("NR")
        .aggregate({"area_intersection": np.max})
        .reset_index()
        .merge(
            intersection[["NR", "area_intersection", "NR_%s" % tag]],
            on=["NR", "area_intersection"],
            how="left",
        )
    )
    parcels1.loc[:, "area"] = parcels1["geometry"].area.values
    intersection = intersection.merge(parcels1[["NR", "area"]], on="NR", how="left")

    # (SG) normalize area intersction  with area of parcel of considered year
    intersection["normalized_overlap"] = (
        intersection["area_intersection"] / intersection["area"] * 100
    )

    return intersection.loc[intersection["normalized_overlap"] > 80].drop_duplicates()


def load_perceelskaart_for_compute_C(percelenshp, jaar):
    """
    functie die percelenshp omzet naar perceelslist nodig voor initialisatie functie
     ```init'''

    Parameters
    ----------
        'percelenshp' (dict): dictionary met keys jaar (vb. 2016, 2017 en 2018).
        Elke bevatten ze de perceelskaart
        waarbij
        het perceelsnummer 'NR' aangegeven is, alsook het nummer van het bijna-gelijk
        perceel (80% overeenstemming in
        oppervlakte) van het jaar ervoor ('NR_pr') en erna ('NR_po').
        'jaar' (int): simulatiejaar

    Returns
    -------
        'perceelslijst' nodig voor initialisatie functie ```init'''
    """
    parcel_list = []
    cols = ["GWSCOD_V", "GWSCOD_H", "GWSCOD_N", "GWSNAM_V", "GWSNAM_H", "GWSNAM_N"]

    # (SG) laad huidig jaar and reformat
    cols_jaar = cols + ["NR", "NR_pr", "NR_po"]
    if type(percelenshp[jaar]) == str:
        df_current = gpd.read_file(percelenshp[jaar])[cols_jaar]
    else:
        df_current = deepcopy(percelenshp[jaar])[cols_jaar]
    temp1 = reformat_perceelskaart_for_compute_C(
        df_current, jaar, ["_V", "_H", "_N"], "NR"
    )

    # (SG) laad prior jaar (and filter on priors jaar i, which are NRs jaar-1)
    prior_NRs = df_current["NR_pr"].unique()
    prior_NRs = prior_NRs[~np.isnan(prior_NRs)]
    if type(percelenshp[jaar - 1]) == str:
        df_prior = gpd.read_file(percelenshp[jaar - 1])[cols_jaar]
    else:
        df_prior = deepcopy(percelenshp[jaar - 1])[cols_jaar]
    df_prior = df_prior.loc[df_prior["NR"].isin(prior_NRs)]
    temp2 = reformat_perceelskaart_for_compute_C(
        df_prior, jaar - 1, ["_H", "_N"], "NR_po"
    )

    # (SG) laad posterior jaar
    post_NRs = df_current["NR_po"].unique()
    post_NRs = post_NRs[~np.isnan(post_NRs)]
    cols_post = cols + ["NR", "NR_pr"]
    if type(percelenshp[jaar + 1]) == str:
        df_post = gpd.read_file(percelenshp[jaar + 1])[cols_post]
    else:
        df_post = deepcopy(percelenshp[jaar + 1])[cols_post]
    df_post = df_post.loc[df_post["NR"].isin(post_NRs)]
    temp3 = reformat_perceelskaart_for_compute_C(df_post, jaar + 1, ["_V"], "NR_pr")

    parcel_list = pd.concat([temp1, temp2, temp3])
    parcel_list = parcel_list[~parcel_list["GWSCOD"].isnull()]
    parcel_list = parcel_list.sort_values(["NR", "jaar", "type"])

    parcel_list["perceel_id"] = deepcopy(parcel_list["NR"])

    return parcel_list


def reformat_perceelskaart_for_compute_C(db, jaar, crop_types, rename):
    """
    Verander format van de perceelskaart als voorbereiding als input voor
    ComputeCFactor.py

    Parameters
    ----------
        db (pandas df): perceelskaart van jaar i
        jaar (int): zelf-verklarend
        crop_types (list): lijst van teelt types (volgens _V,_H,_N, resp voorteelt,
        hoofdteelt en nateelt
        rename (string): kolom dat moet hernoemt worden naar NR
    Returns
    -------
        output (pandas df): gefilterde data van gewascodes, gewasnaame, type
        (1: voorteelt, 2: hoofdteelt, 3: nateelt)
          en jaar
    """
    output = []
    for i in crop_types:
        temp = db[["GWSCOD" + i, "GWSNAM" + i, rename]]
        temp = temp.rename(
            columns={rename: "NR", "GWSNAM" + i: "GWSNAM", "GWSCOD" + i: "GWSCOD"}
        )
        temp = temp[~temp["GWSCOD"].isnull()]
        if i == "_V":
            temp["type"] = 1
        elif i == "_H":
            temp["type"] = 2
        else:
            temp["type"] = 3
        temp["jaar"] = jaar
        output.append(temp)
    return pd.concat(output).drop_duplicates()


def init(parcel_list, fname_input, year, frequency="SMS"):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    keys = list(fname_input.keys())
    # (SG) initialize crop properties
    if "te" in keys:
        crop_prop = init_crop_properties(fname_input["te"])
    else:
        sys.exit("Crop properties not found, please check tag and filename")

    # (SG) initialize list of parcel, and join with crop prop
    if "gwscod" in keys:
        parcel_list = init_groep_ids(fname_input["gwscod"], parcel_list)
    else:
        sys.exit(
            "Table link GWSCOD and groep_id not found, please check tag and filename"
        )

    # (SG) flag parcels for which no data are available for main crop
    parcel_list = flag_incomplete_parcels(parcel_list, year)

    # (SG) get crop growth data (ggg)
    if "ggg" in keys:
        ggg = init_ggg(fname_input["ggg"])
    else:
        sys.exit("Growth curve crops not found, please check tag and filename")

    # (SG) make a time grid to perform calculations
    if (
        ("fname_halfmonthly_rain" in keys)
        & ("fname_halfmonthly_temp" in keys)
        & ("fname_halfmonthly_R" in keys)
    ):
        grid = init_time_grid(
            year,
            fname_input["fname_halfmonthly_rain"],
            fname_input["fname_halfmonthly_temp"],
            fname_input["fname_halfmonthly_R"],
            frequency,
        )
    else:
        sys.exit(
            "Half-monthly total rainfall, total erosivity and mean temperature not"
            "found, please check tag and filename"
        )

    return parcel_list, crop_prop, ggg, grid


def init_crop_properties(fname_crop_prop):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    # (SG) load properties of groups
    col = [
        "groep_id",
        "subgroep_id",
        "voorwaarde",
        "teeltwisseling",
        "zaaidatum",
        "oogstdatum",
        "alpha",
        "Bsi",
        "p",
        "Ri",
        "default",
    ]
    crop_prop = pd.read_csv(fname_crop_prop, usecols=col)
    # (SG) make sure type of column  voorwaarde and teeltwisseling is a string
    for i in ["voorwaarde", "teeltwisseling"]:
        crop_prop.loc[crop_prop[i].isnull(), i] = ""
        crop_prop[i] = crop_prop[i].astype(str)
    return crop_prop


def init_groep_ids(fname_gwscod, parcel):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    # (SG) load couple matrix GWSCD/groep id
    col = [
        "GWSCOD",
        "groep_id",
        "voorteelt",
        "hoofdteelt",
        "nateelt",
        "groente",
        "groenbedekker",
        "meerjarige_teelt",
        "onvolledig",
    ]

    if "GWSNAM" not in parcel.columns:
        col = ["GWSNAM"] + col

    gwscod = pd.read_csv(fname_gwscod, usecols=col, encoding="latin8")
    # (SG) transform GWSCOD to ints and couple groep_ids
    parcel["GWSCOD"] = parcel["GWSCOD"].astype(int)
    parcel = parcel.merge(gwscod, on="GWSCOD", how="left")

    # (SG) delete GWSCOD with no identified groep_id
    parcel = parcel[~parcel["groep_id"].isnull()]
    # (SG) fix types
    parcel["jaar"] = parcel["jaar"].astype(int)
    parcel["type"] = parcel["type"].astype(int)
    # (SG) error message for incomplete data
    if np.sum(parcel["onvolledig"] == 1) > 0:
        print("Some crop inputdata are incomplete, removing records")
        parcel = parcel[parcel["onvolledig"] == 0]
    return parcel


def flag_incomplete_parcels(parcels, year):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    # (SG) check of hoofdteelt aanwezig is voor jaar j per perceel
    hoofdteelten_perceelsid = parcels.loc[
        (parcels["jaar"] == year) & (parcels["type"] == 2), "perceel_id"
    ].unique()

    # (SG) filter enkel hoofdteelt en zie welke percelen geen groeps_id toegekent
    # hebben aan de hoofdteelt
    percelen_beschikbare_gegevens = parcels.loc[
        ~parcels["groep_id"].isnull(), "perceel_id"
    ].unique()

    # (SG) teeltgegevens beschikbaar
    parcels["teeltgegevens_beschikbaar"] = 0.0

    # (SG) toekennen
    parcels.loc[
        (parcels["perceel_id"].isin(percelen_beschikbare_gegevens))
        & (parcels["perceel_id"].isin(hoofdteelten_perceelsid)),
        "teeltgegevens_beschikbaar",
    ] = 1.0

    # (SG) tag for computation in ComputeC function
    parcels["compute"] = deepcopy(parcels["teeltgegevens_beschikbaar"])
    return parcels


def init_ggg(fname_ggg):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    # (SG) load ggg_ids
    col = [
        "subgroep_id",
        "dagen_na",
        "bedekking(%)",
        "hoogte(m)",
        "effectieve_valhoogte(m)",
    ]
    ggg = pd.read_csv(fname_ggg, usecols=col)
    # (SG) rename columns according to formula's page 13  (divide bedekking with
    # 100, unit of Fc is -)
    ggg["Fc"] = ggg["bedekking(%)"] / 100
    ggg["H"] = ggg["effectieve_valhoogte(m)"]
    return ggg


def init_time_grid(year, rain, temperature, Rhm, frequency):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    # (SG) initialize rainfall, temperature and R data
    rain = pd.read_csv(rain)
    rain["timestamp"] = pd.to_datetime(rain["timestamp"], format="%d/%m/%Y")
    temperature = pd.read_csv(temperature)
    temperature["timestamp"] = pd.to_datetime(
        temperature["timestamp"], format="%d/%m/%Y"
    )
    Rhm = pd.read_csv(Rhm)
    Rhm["timestamp"] = pd.to_datetime(Rhm["timestamp"], format="%d/%m/%Y")

    # (SG) identify years
    pre_date = generate_datetime_instance("01", "01", str(year - 1))
    # begin_date = generate_datetime_instance("01", "01", str(year))
    end_date = generate_datetime_instance("01", "01", str(year + 1))

    # (SG) make calculation grid of two years based on frequency
    nodes = pd.date_range(pre_date, end_date, freq=frequency)
    grid = pd.DataFrame(data=nodes, index=range(len(nodes)), columns=["timestamp"])
    grid["bdate"] = pd.to_datetime(grid["timestamp"], format="%Y%m%d")
    grid["D"] = [
        (grid["bdate"].iloc[i] - grid["bdate"].iloc[i - 1]) for i in range(1, len(grid))
    ] + [timedelta(days=15)]
    grid["year"] = grid["bdate"].dt.year
    grid["bmonth"] = grid["bdate"].dt.month
    grid["bday"] = grid["bdate"].dt.day
    grid["edate"] = grid["bdate"] + grid["D"]
    # (SG) remove 29th of februari, to avoid compatibility problems
    grid = grid[~((grid["bmonth"] == 2) & (grid["bday"] == 29))]

    # (SG) rename cols for rain and temperature
    rain["rain"] = np.nanmean(
        rain[[i for i in rain.columns if i.isdigit()]].values, axis=1
    )
    temperature["temp"] = np.nanmean(
        temperature[[i for i in temperature.columns if i.isdigit()]], axis=1
    )
    Rhm["Rhm"] = Rhm["value"]

    # (SG) grid merge
    rain["bday"] = rain["timestamp"].dt.day
    temperature["bday"] = temperature["timestamp"].dt.day
    Rhm["bday"] = Rhm["timestamp"].dt.day
    rain["bmonth"] = rain["timestamp"].dt.month
    temperature["bmonth"] = temperature["timestamp"].dt.month
    Rhm["bmonth"] = Rhm["timestamp"].dt.month

    grid = grid.merge(
        rain[["bmonth", "bday", "rain"]], on=["bmonth", "bday"], how="left"
    )
    grid = grid.merge(
        temperature[["bmonth", "bday", "temp"]], on=["bmonth", "bday"], how="left"
    )
    grid = grid.merge(Rhm[["bmonth", "bday", "Rhm"]], on=["bmonth", "bday"], how="left")

    # (SG) other properties grid
    props = [
        "GWSCOD",
        "ggg_id",
        "har_tag",
        "Ri_tag",
        "groep_id",
        "subgroep_id",
        "meerjarig",
    ]
    for i in props:
        grid[i] = 0.0

    # (SG) prepare output
    calcgrid = [
        "f1_N",
        "f2_EI",
        "Ru",
        "a",
        "Bsb",
        "Sp",
        "W",
        "F",
        "SC",
        "SR",
        "CC",
        "SM",
        "PLU",
        "SLR",
        "C",
    ]
    for i in calcgrid:
        grid[i] = np.nan

    return grid[
        ["bdate", "edate", "bmonth", "bday", "rain", "temp", "Rhm"] + props + calcgrid
    ]


def prepare_grid(
    parcel, grid, ggg, cp, year, parcel_id, output_map, ffull_output=False
):
    """
    Prepare grid for array-like calculations by assigning crop properties found in
    parcel to grid

    Parameters
    ----------
        'parcel': considered parcels, see parameter ``parcel_list`` in
        :func:`ComputeCFactor`.
        'grid': see parameter ``grid`` in :func:`ComputeCFactor`.
        'ggg' (pd df): see parameter ``ggg`` in :func:`ComputeCFactor`.
        'gts' (pd df): see parameters 'gts in :func:`ComputeCFactor`
        'cp' (pd df): see parameters 'cp' in :func:`ComputeCFactor`
        'year' (int): see parameters 'year' in :func:`ComputeCFactor`

    Returns
    -------
         'grid' (pd df): see parameter ``grid`` in :func:`ComputeCFactor`.
            ...
            'GWSCOD': see parameter ``parcel_list`` in :func:`ComputeCFactor`.
            'Ri': see parameter ``parcel_list`` in :func:`ComputeCFactor`.
            'ggg_id': see parameter ``parcel_list`` in :func:`ComputeCFactor`.
            'har_tag'(int): number/id of harvest remains (sequential in time).
            'Ri_tag'(int): number/id of roughness class (sequential in time,
            != Ri_id!!!).
            'H'(float): see see parameter ``ggg'` in :func:`ComputeCFactor`.
            'Fc'(int):  see see parameter ``ggg'` in :func:`ComputeCFactor`.
            'alpha'(float):  see see parameter ``cp'` in :func:`ComputeCFactor`.
            'beta'(float):  see see parameter ``cp'` in :func:`ComputeCFactor`.
            'p'(float):  see see parameter ``cp'` in :func:`ComputeCFactor`.
            'Ri'(float):  see see parameter ``cp'` in :func:`ComputeCFactor`.

    """

    # (SG) simplify crop scheme with a number of rules
    parcel, year = adjust_rotation_scheme(
        parcel, year, output_map, parcel_id, ffull_output=ffull_output
    )

    # (SG) create crop objects
    teelten = create_crop_objects(parcel, cp, year)

    # (SG) map parcel to grid
    grid = map_crops_to_grid(teelten, grid)

    # (SG) assign zero to no crops on field
    grid["H"] = 0.0
    grid["Fc"] = 0.0

    # (SG) assign properties
    cp = cp[["groep_id", "subgroep_id", "alpha", "Bsi", "p", "Ri"]].drop_duplicates()
    grid = grid.merge(cp, on=["groep_id", "subgroep_id"], how="left")
    grid.loc[np.isnan(grid["Ri"]), "Ri"] = rii

    # (SG) assign growth curves
    grid = assign_growth_curvs(grid, ggg)
    return grid


def assign_growth_curvs(grid, ggg):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    for i in grid["Ri_tag"].unique():
        if i != 0:
            cond_i = grid["Ri_tag"] == i
            subgroep_id = grid.loc[cond_i, "subgroep_id"].values[0]
            # (SG) begin and endate
            bdate = grid.loc[
                (grid["subgroep_id"] == subgroep_id) & (cond_i), "bdate"
            ].values[0]
            edate = grid.loc[
                (grid["subgroep_id"] == subgroep_id) & (cond_i), "edate"
            ].values[-1]
            cond = (ggg["subgroep_id"] == subgroep_id) & (
                ggg["dagen_na"] < (edate - bdate) / np.timedelta64(1, "D") + 30
            )
            ggg_i = deepcopy(ggg.loc[cond])
            # (SG) set datetime series format
            if (edate - bdate).astype("timedelta64[D]") / np.timedelta64(
                1, "D"
            ) > ggg_i.loc[ggg_i.index[-2], "dagen_na"]:
                ggg_i.loc[ggg_i.index[-1], "dagen_na"] = (edate - bdate).astype(
                    "timedelta64[D]"
                ) / np.timedelta64(1, "D")

            # (SG) set index and round on day
            ggg_i.index = pd.DatetimeIndex(
                [
                    bdate + np.timedelta64(int(ggg_i.loc[j, "dagen_na"]), "D")
                    for j in ggg_i.index
                ]
            )
            ggg_i.index = ggg_i.index.round("D")

            # (SG) append dates on which should be interpolated
            dates = (
                grid.loc[grid["Ri_tag"] == i, "bdate"]
                + (
                    grid.loc[grid["Ri_tag"] == i, "edate"]
                    - grid.loc[grid["Ri_tag"] == i, "bdate"]
                )
                / 2
            )
            dates = dates.dt.round("D")
            ggg_i = ggg_i.reindex(
                ggg_i.index.tolist() + [i for i in dates if i not in ggg_i.index]
            ).sort_index()

            # (SG) Resample too slow
            ggg_i = ggg_i.interpolate()

            # (SG) Assign to grid
            grid.loc[grid["Ri_tag"] == i, ["H", "Fc"]] = ggg_i.loc[
                dates, ["H", "Fc"]
            ].values
            cond = grid["meerjarig"]
            if np.sum(cond) != 0:
                grid.loc[cond, "H"] = np.max(
                    ggg.loc[(ggg["subgroep_id"] == subgroep_id), "H"]
                )
                grid.loc[cond, "Fc"] = np.max(
                    ggg.loc[(ggg["subgroep_id"] == subgroep_id), "Fc"]
                )
    return grid


def adjust_rotation_scheme(parcel, year, output_map, parcel_id, ffull_output=False):
    """
    Hard-coded simplifications of crop rotation scheme
    Implemented to simplify build-up grid in in :func:`map_parcel_to_grid`.
    NOTA: alle uitzonderingen mbt wintergranen etc worden hier geimplementeerd!

    Parameters
    ----------
        'parcel': (pd df)  considered parcel, see parameter ``parcel_list``
        in :func:`ComputeCFactor`.
        'year' (int): see parameters 'year' in :func:`ComputeCFactor`

    Returns
    -------
        'parcel' (pd df): considered parcel, see parameter ``parcel_list``
        in :func:`ComputeCFactor`.
                        updated by removed rows/records (simplication scheme)

    """
    # max_year!=jaar!!!!!!"
    # (SG) Only consider crop with highest type id for year prior

    max_type = np.max(parcel.loc[(parcel["jaar"] == year - 1), "type"])
    cond = (parcel["jaar"] == year - 1) & (parcel["type"] == max_type)
    parcel = parcel.loc[cond].append(
        parcel.loc[(parcel["jaar"] == year) | (parcel["jaar"] == year + 1)]
    )

    # (SG) Filter parcel list based on whether a crop can be a specfic type
    parcel = filter_types(parcel)

    # (SG) Als er geen nateelt is in jaar i, beschouw dan voorteelt i+1 als nateelt i
    # (SG) Als er een nateelt is in jaar i, verwijder dan voorteelt i+1
    parcel = exceptions_voorteelt_nateelt(parcel, year)

    # (SG) hard gecodeerde uitzonderingen voor wintergraan
    parcel = exceptions_winterteelten(parcel, year)

    # (SG) hard gecodeerde uitzonderingen voor meerjarige teelten
    parcel, year = exceptions_meerjarigeteelten(parcel, year)

    # (SG) wanneer conflict is tussen hoofdteelt jaar j en nateelt jaar j-1,
    # vertrouw hoofdteelt jaar j
    if np.sum((parcel["jaar"] == year - 1) & (parcel["type"] == 3)) == 2:
        parcel = parcel.iloc[1:]

    # (SG) filter teelten die in jaar j-2 gepositioneerd zijn (vb van hoofdteelt j-1
    # naar nateelt jaar j-2 geplaatst)
    parcel = parcel[parcel["jaar"] != year - 2]

    # (SG) als er conflicten afgeleid zijn uit vereenvoudigingen (twee of drie types
    # gedefinieerd in eenzelfde jaar),
    # neem dan de laatste teelt per koppel (jaar,type): dat wordt beschouwd als het
    # meest betrouwbaar
    parcel = parcel.drop_duplicates(subset=["jaar", "type"], keep="last")

    # (SG) Only consider crop with highest type id for year prior
    max_type = np.max(parcel.loc[(parcel["jaar"] == year - 1), "type"])
    parcel = parcel.loc[
        (parcel["jaar"] == year - 1) & (parcel["type"] == max_type)
    ].append(parcel.loc[(parcel["jaar"] == year) | (parcel["jaar"] == year + 1)])

    # (SG) print parcel to disk
    if ffull_output:
        parcel.to_csv(
            os.path.join(
                output_map, "%i_rotation_scheme_simplified.csv" % int(parcel_id)
            )
        )

    return parcel, year


def filter_types(parcel):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    # (SG) filter hoofdteelt, nateelten en voorteelten
    cond = (
        (parcel["type"] == 1) & (parcel["voorteelt"] == 0)
        | (parcel["type"] == 2) & (parcel["hoofdteelt"] == 0)
        | (parcel["type"] == 3) & (parcel["nateelt"] == 0)
    )
    return parcel[~cond]


def exceptions_voorteelt_nateelt(parcel, year):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    # (SG) als er conflict is tussen de voorteelt jaar j (j+1) en nateelt j-1 (j)
    # geloof de voorteelt jaar (j+1)
    for i in [year, year + 1]:
        cond_voorteelt = (parcel["type"] == 1) & (parcel["jaar"] == i)
        cond_nateelt = (parcel["type"] == 3) & (parcel["jaar"] == i - 1)
        # cond_groente = (parcel["groente"] == 1) & (parcel["jaar"] == i)

        GWSCOD_voorteelt = parcel.loc[cond_voorteelt, "GWSCOD"]
        GWSCOD_nateelt = parcel.loc[cond_nateelt, "GWSCOD"]

        if len(GWSCOD_voorteelt) > 0:
            if len(GWSCOD_nateelt) > 0:
                # (SG) als GWSCOD nateelt gelijk is aan GWSCOD voorteelt: behoud
                # enkel nateelt
                if GWSCOD_nateelt.values[0] == GWSCOD_voorteelt.values[0]:
                    parcel = parcel[~cond_voorteelt]
                # (SG) als GWSCOD nateelt niet gelijk is aan GWSCOD voorteelt:
                # verwijder nateelt en maak voorteelt nateelt ALS voorteelt geen
                # groente is!
                # (SG) de voorteelt in de perceelskaart wordt altijd geloofd!
                else:
                    if (
                        parcel.loc[
                            (parcel["GWSCOD"] == GWSCOD_voorteelt.values[0])
                            & (parcel["jaar"] == i),
                            "groente",
                        ].values[0]
                        == 0
                    ):
                        parcel = deepcopy(parcel[~cond_nateelt])
                        parcel.loc[cond_voorteelt, ["type", "jaar"]] = [3, i - 1]
            # (SG) als er geen nateelt is zet dan de voorteelt gelijk aan nateelt
            else:
                parcel.loc[cond_voorteelt, ["type", "jaar"]] = [3, i - 1]
        # (SG) als er geen voorteelt is: doe niets :)
    return parcel


def exceptions_winterteelten(parcel, year):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    groep_id_wintercrops = [8, 9]

    if np.sum(parcel["groep_id"].isin(groep_id_wintercrops)) > 0:
        for i in groep_id_wintercrops:
            # (SG) Als de hoofdteelt gelijk is aan een wintergewas, maak het dan
            # een nateelt:
            cond = (parcel["groep_id"] == i) & (parcel["type"] == 2)
            parcel.loc[cond, "type"] = 3
            parcel.loc[cond, "jaar"] = parcel.loc[cond, "jaar"] - 1

    return parcel


def exceptions_meerjarigeteelten(parcel, year):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    # (SG) Als een teeltschema enkel één type gewas bevat, en het is meerjarig,
    # beschouw dan enkel hoofdteelten en reken enkel jaar-1 door!
    # (SG) als er enkel gegevens zijn over teelt jaar i beschouw het dan als permanent
    # grasland
    # (SG) beschouw enkel hoofdgewassen van meerjarige teelten
    parcel["meerjarig"] = 0.0

    # (SG) verwijder twee opeenvolgende meerjarige teelten, behoud de eerste!
    if np.sum(parcel["meerjarige_teelt"] == 1) > 1:
        cond = [True] + [
            False
            if (
                (parcel["groep_id"].iloc[i - 1] == parcel["groep_id"].iloc[i])
                & (parcel["meerjarige_teelt"].iloc[i] == 1)
            )
            else True
            for i in range(1, len(parcel), 1)
        ]
        parcel = parcel[cond]

    # if np.sum(parcel["meerjarige_teelt"]==1)==len(parcel):
    #     # (SG) enkel hoofdgewassen (tenzij er geen hoofgewassen zijn, doe dan
    #     een aanpassing aan type)
    #     temp = deepcopy(parcel)
    #     parcel = parcel[parcel["type"] == 2]
    #     #(SG) voeg meerjaar gewas toe
    #     if len(parcel) == 1:
    #         year_ = parcel["jaar"].unique()[0]
    #         parcel = parcel.append(deepcopy(parcel.iloc[0])).reset_index()
    #         assign_year = year_ - 1 if year_ == year else year_ + 1
    #         parcel.loc[parcel.index[0], "jaar"] = assign_year
    #         parcel.loc[parcel.index[0],"meerjarig"] = 1
    #     #(SG) als de parcel_list empty was dan betekent dit dat gras enkel als
    #     voor en nateelt beschreven was,
    #     #(SG) verander in dit geval de grassen tot hoofdteelten
    #     if len(parcel) ==0:
    #         parcel = temp
    #         parcel.loc[:,"type"] = 2
    #
    #     if len(parcel)==2:
    #         parcel.loc[parcel["jaar"]==year,["type","jaar"]] = [3,year-1]

    return parcel, year


def eval_scheme_statement(parcel, statement, max_year):
    """
    Function to apply string on parcel dataframe which implements the simplication of
    the rotatop, scheme

    Parameters
    ----------
        'parcel' (pd df): considered parcels, see parameter ``parcel_list`` in
         :func:`ComputeCFactor`.
        'statement' (string): condition which should be applied to dataframe parcel
        'max_year' (int): maximum year found in specific parcel (not per se equal to
        year of simulation (
        e.g. not crops reported for specific year))

    Returns
    -------
        'cond' (list): list of bool stating wether to 'keep' (true) or remove (false)
         record/row of df

    """
    indices = parcel.index
    cond = []
    if "i+1" in statement:
        for i in range(0, len(indices) - 1, 1):
            if eval(statement):
                cond.append(False)
            else:
                cond.append(True)
        cond = cond + [True]
    else:
        for i in range(1, len(indices), 1):
            if eval(statement):
                cond.append(False)
            else:
                cond.append(True)
        cond = [True] + cond
    return np.array(cond)


def create_crop_objects(parcel, cp, year):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    print(parcel)
    print(cp)
    print(year)
    return None


#     # (SG) get 'hoofdteelt' i
#     groep_id = parcel.loc[
#         (parcel["type"] == 2) & (parcel["jaar"] == year) & (parcel["meerjarig"] == 0),
#         "groep_id",
#     ]
#     GWSCOD = parcel.loc[
#         (parcel["type"] == 2) & (parcel["jaar"] == year) & (parcel["meerjarig"] == 0),
#         "GWSCOD",
#     ]
#     groenbedekker = parcel.loc[
#         (parcel["type"] == 2) & (parcel["jaar"] == year), "groenbedekker"
#     ]
#
#     if len(groep_id) > 0:
#         hoofdteelt = Crop(
#             "hoofdteelt",
#             year,
#             cp,
#             groep_id.values[0],
#             GWSCOD.values[0],
#             groenbedekker=groenbedekker.values[0],
#         )
#     else:
#         hoofdteelt = None
#
#     # (SG) get 'hoofdteelt' i-1
#     groep_id = parcel.loc[
#         (parcel["type"] == 2) & (parcel["jaar"] == year - 1), "groep_id"
#     ]
#     GWSCOD = parcel.loc[(parcel["type"] == 2) & (parcel["jaar"] == year - 1),
#     "GWSCOD"]
#     meerjarig = parcel.loc[
#         (parcel["type"] == 2) & (parcel["jaar"] == year - 1), "meerjarige_teelt"
#     ]
#     groenbedekker = parcel.loc[
#         (parcel["type"] == 2) & (parcel["jaar"] == year - 1), "groenbedekker"
#     ]
#
#     if len(groep_id) > 0:
#         hoofdteelt_prior = Crop(
#             "hoofdteelt",
#             year - 1,
#             cp,
#             groep_id.values[0],
#             GWSCOD.values[0],
#             meerjarig=meerjarig.values[0],
#             groenbedekker=groenbedekker.values[0],
#         )
#         # (SG) pas oogstdatum voor hoofdteelt_prior aan naar laatste jaar
#         # (SG) als hoofdteelt prior groendbedekker en hoofdteelt niet, pas
#         oogstdatum hoofdteelt prior aan
#         if hoofdteelt is None:
#             # (SG) hoofdteelt bestaat niet, alsook  voorteelt/nateelt jaar j niet
#             cond = (
#                 (hoofdteelt_prior.meerjarig == True)
#                 & (np.sum((parcel["type"] == 1) & (parcel["jaar"] == year)) == 0)
#                 & (np.sum((parcel["type"] == 3) & (parcel["jaar"] == year)) == 0)
#             )
#             if cond:
#                 hoofdteelt_prior.harvest_date = generate_datetime_instance(
#                     "01", "01", year + 1
#                 )
#         else:
#             if hoofdteelt_prior.groenbedekker == 1:
#                 hoofdteelt_prior.harvest_date = hoofdteelt.sowing_date
#             cond = (hoofdteelt_prior.meerjarig == True) & (
#                 np.sum((parcel["type"] == 1) & (parcel["jaar"] == year)) == 0
#             )
#             if cond:
#                 hoofdteelt_prior.harvest_date = hoofdteelt.sowing_date
#     else:
#         hoofdteelt_prior = None
#
#     # (SG) get 'nateelt' i
#     groep_id = parcel.loc[(parcel["type"] == 3) & (parcel["jaar"] == year),
#     "groep_id"]
#     GWSCOD = parcel.loc[(parcel["type"] == 3) & (parcel["jaar"] == year), "GWSCOD"]
#     groenbedekker = parcel.loc[
#         (parcel["type"] == 3) & (parcel["jaar"] == year), "groenbedekker"
#     ]
#
#     if len(groep_id) > 0:
#         nateelt = Crop(
#             "nateelt",
#             year,
#             cp,
#             groep_id.values[0],
#             GWSCOD.values[0],
#             groenbedekker=groenbedekker.values[0],
#         )
#         if hoofdteelt is not None:
#             hoofdteelt, nateelt = fit_dates_hoofd_nateelt(hoofdteelt, nateelt)
#     else:
#         nateelt = None
#
#     # (SG) get 'voorteelt' i
#     groep_id = parcel.loc[(parcel["type"] == 1) & (parcel["jaar"] == year),
#     "groep_id"]
#     GWSCOD = parcel.loc[(parcel["type"] == 1) & (parcel["jaar"] == year), "GWSCOD"]
#
#     if len(groep_id) > 0:
#         voorteelt = Crop("voorteelt", year, cp, groep_id.values[0], GWSCOD.values[0])
#         if hoofdteelt is not None:
#             hoofdteelt, voorteelt = fit_harvest_date(hoofdteelt, voorteelt)
#     else:
#         voorteelt = None
#
#     # (SG) get 'nateelt_prior' i
#     groep_id = parcel.loc[
#         (parcel["type"] == 3) & (parcel["jaar"] == year - 1), "groep_id"
#     ]
#     GWSCOD = parcel.loc[(parcel["type"] == 3) & (parcel["jaar"] == year - 1),
#     "GWSCOD"]
#     groenbedekker = parcel.loc[
#         (parcel["type"] == 3) & (parcel["jaar"] == year - 1), "groenbedekker"
#     ]
#
#     if len(groep_id) > 0:
#         nateelt_prior = Crop(
#             "nateelt",
#             year - 1,
#             cp,
#             groep_id.values[0],
#             GWSCOD.values[0],
#             groenbedekker=groenbedekker.values[0],
#         )
#         if hoofdteelt_prior is not None:
#             hoofdteelt_prior, nateelt_prior = fit_dates_hoofd_nateelt(
#                 hoofdteelt_prior, nateelt_prior
#             )
#
#         if hoofdteelt is not None:
#             if voorteelt is None:
#                 hoofdteelt, nateelt_prior = fit_harvest_date(hoofdteelt,
#                 nateelt_prior)
#             else:
#                 voorteelt, nateelt_prior = fit_harvest_date(voorteelt, nateelt_prior)
#         else:
#             if (voorteelt is None) & (nateelt is not None):
#                 nateelt, nateelt_prior = fit_harvest_date(nateelt, nateelt_prior)
#     else:
#         nateelt_prior = None
#
#     # (SG) if harvest date hoofdteelt is not filled, set it equal to 01/01 next year
#     if hoofdteelt is not None:
#         if hoofdteelt.harvest_date is None:
#             hoofdteelt.harvest_date = generate_datetime_instance(
#                 "01", "01", hoofdteelt.year + 2
#             )
#
#     # (SG) if harvest date nateelt is not filled, set it equal to 01/01 next year
#     if nateelt is not None:
#         if nateelt.harvest_date is None:
#             nateelt.harvest_date = generate_datetime_instance(
#                 "01", "01", nateelt.year + 2
#             )
#
#     # (SG) if harvest date nateelt_prior is not filled, set it equal to 01/01
#     next year
#     if nateelt_prior is not None:
#         if nateelt_prior.harvest_date is None:
#             nateelt_prior.harvest_date = generate_datetime_instance(
#                 "01", "01", nateelt_prior.year + 3
#             )
#
#     if hoofdteelt_prior is not None:
#         if hoofdteelt_prior.harvest_date is None:
#             if (nateelt is not None) & (hoofdteelt is None) & (voorteelt is None):
#                 hoofdteelt_prior.harvest_date = nateelt.sowing_date
#
#     teelten = {}
#     teelten["hoofdteelt"] = hoofdteelt
#     teelten["hoofdteelt_prior"] = hoofdteelt_prior
#     teelten["nateelt"] = nateelt
#     teelten["nateelt_prior"] = nateelt_prior
#     teelten["voorteelt"] = voorteelt
#     return teelten

# (SG) get 'voorteelt' i
# groep_id = parcel.loc[(parcel["type"]==1) & (parcel["jaar"]==year),"groep_id"]
# if len(groep_id)>0:
#    voorteelt = Crop("voorteelt", year, cp, groep_id.values[0])
#    if hoofdteelt!=None:
#        hoofdteelt,voorteelt = fit_harvest_date(hoofdteelt,voorteelt)
# else:
#    voorteelt = None

# (SG) get dates in 'nateelt' i-1
# (note that a 'nateelt' year i-1 is only considered if there is a 'voorteelt' year i),
# see simplifications in simplify rotation crops
# groep_id = parcel.loc[(parcel["type"]==3) & (parcel["jaar"]==year-1),"groep_id"]
# if len(groep_id)>0:
#    nateelt_prior = Crop("nateelt", year-1, cp, groep_id.values[0])
#    if hoofdteelt_prior!=None:
#        hoofdteelt_prior, nateelt_prior =
#        fit_dates_hoofd_nateelt(hoofdteelt_prior, nateelt_prior)
#    if voorteelt!=None:
#        nateelt_prior.fit_sowing_date(voorteelt.sowing_date)
#    else:
#        if hoofdteelt!=None:
#            nateelt_prior.fit_sowing_date(hoofdteelt.sowing_date)
# else:
#    nateelt_prior= None


class Crop:
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(
        self, type, year, cp, groep_id, GWSCOD, meerjarig=False, groenbedekker=0
    ):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        self.type = type
        self.year = year
        self.groep_id = groep_id
        self.source = cp.loc[cp["groep_id"] == self.groep_id]
        self.GWSCOD = GWSCOD
        self.meerjarig = meerjarig
        self.groenbedekker = groenbedekker

        self.check_conditions()

        # (SG) uitzondering brouwersgerst, deze worden pas toegekent na vergelijken
        # nateelt_prior en hoofdteelt
        self.set_default_values()

    def check_conditions(self):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        # (SG) filter subgroeps based on condition on type (vb. gras is ingeschreven
        # als hoofdteelt)
        if (
            np.sum(
                self.source["teeltwisseling"].isin(
                    ["hoofdteelt", "nateelt", "voorteelt"]
                )
            )
            > 0
        ):
            if self.type == "voorteelt":
                string = "voorteelt"
            elif self.type == "hoofdteelt":
                string = "hoofdteelt"
            else:
                string = "nateelt"
            self.source = self.source[self.source["teeltwisseling"] == string]
            if len(self.source) == 1:
                self.subgroep_id = self.source["subgroep_id"].values[0]
                self.source["default"] = 1

    def set_default_values(self):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        self.default_sow_harvest_date()
        self.default_growth_curve()

    def default_sow_harvest_date(self):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        cond = self.source["default"] == 1
        self.sowing_date_int = int(self.source.loc[cond, "zaaidatum"].values[0])
        self.harvest_date_int = int(self.source.loc[cond, "oogstdatum"].values[0])
        # (SG) convert to datetime object
        self.create_datetime_objects_sowingharvest()

    def default_growth_curve(self):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        cond = self.source["default"] == 1
        self.subgroep_id = self.source.loc[cond, "subgroep_id"].values[0]

    def create_datetime_objects_sowingharvest(self):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        # (SG) als er een nateelt is en deze is een groenbedekker, kap dan de
        # oogstdatum van het hoofdgewas a
        self.sowing_date = self.create_datetime_objects(self.sowing_date_int)
        if self.harvest_date_int != -9999.0:
            self.harvest_date = self.create_datetime_objects(self.harvest_date_int)
            # (SG)  if harvest date before sowing date: correct
            if self.harvest_date < self.sowing_date:
                self.harvest_date = self.harvest_date + timedelta(days=365)
        else:
            self.harvest_date = None

    def update_sowhardate(self, sowingdate, harvestdate):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        self.update_sowingdate(sowingdate)
        self.update_harvestdate(harvestdate)

    def update_sowingdate(self, inputdate):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        """
        Function to change sowing date

        Parameters
        ----------
            'inputdate' (datimetime object or int): datetime/int object that should be
            used to update sowing date.
                    int should be in yyyymmmdd

        Returns
        -------

        """

        if ~isinstance(inputdate, date):
            self.sowing_date = inputdate
            self.sowing_date_int = int(self.sowing_date.strftime("%Y%m%d"))
        else:
            self.sowing_date_int = int(inputdate)
            self.sowing_date = self.create_datetime_objects(int(inputdate))

    def update_harvestdate(self, inputdate):
        """
        Function to change sowing date

        Parameters
        ----------
            'inputdate' (datimetime object or int): datetime/int object that
            should be used to update harvest date.
                    int should be in yyyymmmdd

        Returns
        -------
        """
        if isinstance(inputdate, date):
            self.harvest_date = inputdate
            self.harvest_date_int = int(self.harvest_date.strftime("%Y%m%d"))
        else:
            self.harvest_date_int = int(inputdate)
            self.harvest_date = self.create_datetime_objects(int(inputdate))

    def create_datetime_objects(self, date):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        date = str(int(date))
        dd = date[-2:]
        mm = date[4:6]
        yyyy = self.year
        return generate_datetime_instance(dd, mm, yyyy)

    def extract_non_default_growth_curve(self, indices):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        # (SG) loop over sowing dates
        for i in indices:
            string = self.source.loc[i, "voorwaarde"]
            [lower_bound, upper_bound] = self.extract_bounds_from_string(string)
            if (int(str(self.sowing_date_int)[4:]) >= lower_bound) & (
                int(str(self.sowing_date_int)[4:]) <= upper_bound
            ):
                self.subgroep_id = self.source.loc[i, "subgroep_id"]

    def extract_bounds_from_string(self, string):
        """
        #TODO

        Parameters
        ----------

        Returns
        -------

        """
        lower_bound = int(string[string.index("[") + 1 : string.index(",")])
        upper_bound = int(string[string.index(",") + 1 : string.index("]")])
        return lower_bound, upper_bound


def fit_dates_hoofd_nateelt(hoofdteelt, nateelt):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    WINTERGRAAN = [8, 9, 10]
    # (SG) check of de hoofdteelt wel een oogstdatum gedefinieerd heeft
    # (SG) zoniet, stel die gelijk aan de default zaaidatum van de nateelt

    if hoofdteelt.harvest_date is None:
        hoofdteelt.harvest_date = nateelt.sowing_date

    # (SG) Als de oogstdatum van de hoofdteelt na de zaaidatum ligt van de nateelt,
    # plaats dan de zaaidatum van de nateelt 15 dagen na de oogstdatum van de
    # hoofdteelt
    threshold_date = generate_datetime_instance(15, 10, hoofdteelt.year)
    if (nateelt.sowing_date <= hoofdteelt.harvest_date) & (
        hoofdteelt.groenbedekker == 0
    ):
        nateelt.update_sowingdate(hoofdteelt.harvest_date + timedelta(days=15))
    # (SG) Als de nateelt een wintergraan is en de hoofdteelt oogstdatum na 15 oktober
    # ligt
    # pas dan de oogst en zaaidatum aan = 15 oktober (threshold_date)
    elif (nateelt.sowing_date <= hoofdteelt.harvest_date) & (
        hoofdteelt.groenbedekker != 0
    ):
        hoofdteelt.update_harvestdate(nateelt.sowing_date)
        nateelt.update_sowingdate(hoofdteelt.harvest_date)
    elif (nateelt.groep_id in WINTERGRAAN) & (
        hoofdteelt.harvest_date >= threshold_date
    ):
        nateelt.update_sowingdate(threshold_date)
        hoofdteelt.update_harvestdate(threshold_date)

    # (SG) selecteer de gepaste gewasgroeicurve via subgroup_id
    indices = [i for i in nateelt.source["voorwaarde"] if "inzaaidatum" in i]
    indices = nateelt.source[nateelt.source["voorwaarde"].isin(indices)].index
    if len(indices) > 0:
        nateelt.extract_non_default_growth_curve(indices)
    return hoofdteelt, nateelt


def fit_harvest_date(hoofdteelt, nateelt_prior):
    """
    #TODO

    Parameters
    ----------

    Returns
    -------

    """
    GRASSEN = [16]
    MAIS = [1, 2]

    # (SG) implenteer uitzondering voor brouwersgerst
    # if (hoofdteelt.groep_id == 10):
    #    if (nateelt_prior.groep_id == 10):
    #        nateelt_prior.source =
    #        nateelt_prior.source[nateelt_prior.source["teeltwisseling"] ==
    #        "voorwaarde1"]
    #        nateelt_prior.set_default_values()
    #    else:
    #        hoofdteelt.source =
    #        hoofdteelt.source[hoofdteelt.source["teeltwisseling"] == "voorwaarde2"]
    #        hoofdteelt.set_default_values()

    # (SG) in geval dat de inzaaidatum van de groenbdekker afhangt van de hoofdteelt
    if nateelt_prior.harvest_date is None:
        nateelt_prior.harvest_date = hoofdteelt.sowing_date

    # (SG) Als de zaaidatum van de hoofdteelt voor de oogstdatum van de voorteelt ligt,
    if nateelt_prior.harvest_date >= hoofdteelt.sowing_date:
        # (SG) En de groep_id duidt op en gras
        # Dan moet het gras geoogst worden 15 dagen voor de zaaidatum van het
        # hoofdgewas (behalve bij mais)
        if nateelt_prior.groep_id in GRASSEN:
            nateelt_prior.harvest_date = (
                hoofdteelt.sowing_date
                if hoofdteelt.groep_id in MAIS
                else hoofdteelt.sowing_date - timedelta(days=15)
            )
        else:
            nateelt_prior.harvest_date = hoofdteelt.sowing_date

    return hoofdteelt, nateelt_prior


def map_crops_to_grid(teelten, grid):
    """
    Prepare grid for array-like calculations by assigning crop properties found in
    parcel to grid

    Parameters
    ----------
        'grid': see parameter ``grid`` (Returns) in :func:`prepare_grid`.
        'gts' (pd df): see parameters 'gts in :func:`ComputeCFactor`
        'year' (int): see parameters 'year' in :func:`ComputeCFactor`

    Returns
    -------
        'grid': see parameter ``grid`` (Returns) in :func:`prepare_grid`.

    """

    Ri_tag = 1
    har_tag = 1

    hoofdteelt = teelten["hoofdteelt"]
    nateelt = teelten["nateelt"]
    hoofdteelt_prior = teelten["hoofdteelt_prior"]
    nateelt_prior = teelten["nateelt_prior"]

    # (SG) map hoofdteelt prior naar het grid
    # ken een Ri_id toe, en de ggg_id voor gewasgroei
    if hoofdteelt_prior is not None:
        bcrop, ecrop = allocate_grid(
            grid, hoofdteelt_prior.sowing_date, hoofdteelt_prior.harvest_date
        )
        cond = (bcrop <= grid["bdate"]) & (ecrop > grid["edate"])
        Ri_tag += 1
        grid.loc[
            cond,
            ["groep_id", "subgroep_id", "GWSCOD", "Ri_tag", "har_tag", "meerjarig"],
        ] = [
            hoofdteelt_prior.groep_id,
            hoofdteelt_prior.subgroep_id,
            hoofdteelt_prior.GWSCOD,
            Ri_tag,
            0,
            hoofdteelt_prior.meerjarig,
        ]

    # (SG) map nateelt prior to grid
    # ken een Ri_id toe, en de ggg_id voor gewasgroei
    # NOTA: als er geen nateelt is, laat gewastresten hoofdteelt prior op terrein
    # liggen!
    if nateelt_prior is not None:
        bcrop, ecrop = allocate_grid(
            grid, nateelt_prior.sowing_date, nateelt_prior.harvest_date
        )
        cond = (bcrop <= grid["bdate"]) & (ecrop > grid["edate"])
        Ri_tag += 1
        grid.loc[
            cond,
            ["groep_id", "subgroep_id", "GWSCOD", "Ri_tag", "har_tag", "meerjarig"],
        ] = [
            nateelt_prior.groep_id,
            nateelt_prior.subgroep_id,
            nateelt_prior.GWSCOD,
            Ri_tag,
            0,
            nateelt_prior.meerjarig,
        ]
    else:
        # (SG) map harvest remains of hoofdteelt proir to grid
        if hoofdteelt_prior is not None:
            cond = ecrop <= grid["edate"]
            har_tag += 1
            # (SG) Ri van hoofdteelt laten doorgaan!!!
            grid.loc[
                cond,
                ["groep_id", "subgroep_id", "GWSCOD", "har_tag", "Ri_tag", "meerjarig"],
            ] = [
                hoofdteelt_prior.groep_id,
                hoofdteelt_prior.subgroep_id,
                hoofdteelt_prior.GWSCOD,
                har_tag,
                0,
                hoofdteelt_prior.meerjarig,
            ]

    # (SG) map hoofdteelt naar het grid
    # ken een Ri_id toe, en de ggg_id voor gewasgroei
    if hoofdteelt is not None:
        bcrop, ecrop = allocate_grid(
            grid, hoofdteelt.sowing_date, hoofdteelt.harvest_date
        )
        cond = (bcrop <= grid["bdate"]) & (ecrop > grid["edate"])
        Ri_tag += 1
        grid.loc[
            cond,
            ["groep_id", "subgroep_id", "GWSCOD", "Ri_tag", "har_tag", "meerjarig"],
        ] = [
            hoofdteelt.groep_id,
            hoofdteelt.subgroep_id,
            hoofdteelt.GWSCOD,
            Ri_tag,
            0,
            hoofdteelt.meerjarig,
        ]
    else:
        # (SG) voeg harvest remains van nateelt vorig jaar toe
        if nateelt_prior is not None:
            cond = ecrop <= grid["edate"]
            har_tag += 1
            grid.loc[
                cond,
                ["groep_id", "subgroep_id", "GWSCOD", "har_tag", "Ri_tag", "meerjarig"],
            ] = [
                nateelt_prior.groep_id,
                nateelt_prior.subgroep_id,
                nateelt_prior.GWSCOD,
                har_tag,
                0,
                nateelt_prior.meerjarig,
            ]

    # (SG) map nateelt to grid
    # ken een Ri_id toe, en de ggg_id voor gewasgroei
    # NOTA: als er geen nateelt is, laat gewastresten hoodteelt op terrein liggen!
    if nateelt is not None:
        bcrop, ecrop = allocate_grid(grid, nateelt.sowing_date, nateelt.harvest_date)
        cond = (bcrop <= grid["bdate"]) & (ecrop >= grid["edate"])
        Ri_tag += 1
        grid.loc[
            cond,
            ["groep_id", "subgroep_id", "GWSCOD", "Ri_tag", "har_tag", "meerjarig"],
        ] = [
            nateelt.groep_id,
            nateelt.subgroep_id,
            nateelt.GWSCOD,
            Ri_tag,
            0,
            nateelt.meerjarig,
        ]
    else:
        if hoofdteelt is not None:
            cond = ecrop <= grid["edate"]
            har_tag += 1
            grid.loc[
                cond,
                ["groep_id", "subgroep_id", "GWSCOD", "har_tag", "Ri_tag", "meerjarig"],
            ] = [
                hoofdteelt.groep_id,
                hoofdteelt.subgroep_id,
                hoofdteelt.GWSCOD,
                har_tag,
                0,
                hoofdteelt.meerjarig,
            ]

    # (SG) map voorteelt to grid
    # ken een Ri_id toe, en de ggg_id voor gewasgroei
    #  if zaaidatumV is not None:
    #      bcrop, ecrop = allocate_grid(grid, zaaidatumV, oogstdatumV)
    #      cond = (bcrop <= grid["bdate"]) & (ecrop > grid["edate"])
    #      Ri_tag += 1
    #      grid.loc[cond, ["subgroep_id", "GWSCOD", 'Ri_tag', 'har_tag']] =
    #      [ggg_idV, 0, Ri_tag, 0]
    # else:
    #    bcrop, ecrop = allocate_grid(grid, oogstdatumN_, zaaidatumH)
    #    cond = (bcrop <= grid["bdate"]) & (ecrop > grid["edate"])
    #    har_tag += 1
    #    grid.loc[cond, ["ggg_id", "GWSCOD", 'har_tag', 'groep_id']] =
    #    [ggg_idH, 0, har_tag, groep_idV]

    return deepcopy(grid.iloc[:-1])


def get_dates_crop(gts, cond, parcel, year, zaaidatum=None):
    """
    Get sowing and harvest dates crop rotation scheme (gts) from parcel on record/row
    conditioned by 'cond'

    Parameters
    ----------
        'grid': see parameter ``grid`` (Returns) in :func:`prepare_grid`.
        'gts' (pd df): see parameters 'gts in :func:`ComputeCFactor`
        'cond' (string): condition which states which rows/records  parcel should be
        considered.
        'year' (int): see parameters 'year' in :func:`ComputeCFactor`
        'zaaidatum' (int): sowing date which is based on harvest date of another crop
        (when used: only groep_id is important to know (see return!)

    Returns
    -------
        if zaaidatum == None
            'zaaidatum' (datime object): sowing date of crop
            'oogstdatum' (datime object): harvest date of crop
            'ggg_id' (int): id of crop growth curve
            'groep_id' (int): groep id of crop
        else:
            'groep_id' (int): groep id of crop

    """
    groep_id = parcel.loc[cond, "groep_id"].values[0]
    # (SG) get sowing date
    if zaaidatum is None:
        cond = gts.loc[(gts["groep_id"] == groep_id)]
        index = cond.loc[(cond["zaaidatum1"] == np.min(cond["zaaidatum1"]))].index
        [zaaidatum, dagen_tot_oogst, ggg_id] = gts.loc[
            index, ["zaaidatum1", "dagen_tot_oogst", "ggg_id"]
        ].values[0]
        zaaidatum = string_to_date(zaaidatum, year)
        oogstdatum = zaaidatum + timedelta(days=int(dagen_tot_oogst))
        return zaaidatum, oogstdatum, ggg_id, groep_id
    else:
        zaaidatum_int = int("1900" + zaaidatum.strftime("%m%d"))
        cond = (
            (gts["zaaidatum1"] <= zaaidatum_int)
            & (gts["zaaidatum2"] > zaaidatum_int)
            & (gts["groep_id"] == groep_id)
        )
        # (SG) als de 'cond' series allemaal False is: you're in trouble.
        # Neenee, dit betekent dat de laatste zaai-instantie gebruikt moet worden,
        # i.e. niet gebonden door einddatum!
        if np.sum(cond) == 0:
            cond[(gts["groep_id"] == groep_id) & (gts["zaaidatum2"].isnull())] = True
        return gts.loc[cond, "ggg_id"].values[0]


def allocate_grid(grid, zaaidatum, oogstdatum):
    """
    Get begin and end date of period which should be considered for grid

    Parameters
    ----------
        'grid': see parameter ``grid`` (Returns) in :func:`prepare_grid`.
        'zaaidatum' (datime object): sowing date of crop
        'oogstdatum' (datime object): harvest date of crop

    Returns
    -------
        'bcrop' (datime object): begin date of period in which crop is sowed (grid)
        'ecrop' (datime object):  end date of period in which crop is harvested (grid)

    """

    # (SG) allocate to grid
    if zaaidatum > grid["bdate"].iloc[-1]:
        bcrop = grid["bdate"].iloc[-1]
    else:
        condz = (grid["bdate"] <= zaaidatum) & (grid["edate"] > zaaidatum)
        bcrop = grid.loc[condz, "bdate"].iloc[0]

    if oogstdatum > grid["edate"].iloc[-1]:
        ecrop = grid["edate"].iloc[-1]
    else:
        condo = (grid["bdate"] <= oogstdatum) & (grid["edate"] > oogstdatum)
        ecrop = grid.loc[condo, "edate"].iloc[0]

    return bcrop, ecrop


def string_to_date(input, year):
    """
    format string object of date to datetime object

    Parameters
    ----------
        'input' (str): string object of date format 19000101
        'year' (int): year which should be filled in

    Returns
    -------
        'ouput' (datime object): time date object of date

    """
    output = datetime.strptime(str(int(year)) + str(input)[4:], "%Y%m%d")

    return output
