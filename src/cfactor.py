__author__ = 'Sacha Gobeyn (SG, Fluves)' \
             'Daan Renders (DR, Fluves)'
__maintainer__ = 'Sacha Gobeyn / Daan Renders'
__email__ = 'sacha@fluves.com, sachagobeyn@gmail.com, daan@fluves.com'

import logging

try:
    import pandas as pd
    import numpy as np
    import os
    from datetime import datetime, timedelta, date
    import pdb
    import sys
    from copy import deepcopy
    import time
    import subprocess
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

except ImportError as e:
    logging.error('not all necessary libraries are available for import!')
    logging.error(e)
    sys.exit()


b = 0.035
Rii = 6.096

R0 = 25.76  # minimum gemiddelde halfmaandelijks neerslag nodig voor afbraak (opm. pagina 6?)
T0 = 37  # celsius!
A = 7.76  # celsius!


def generate_datetime_instance(dd, mm, yyyy):
    date = datetime.strptime(str(dd) + str(mm) + str(yyyy), "%d%m%Y")
    return date


def load_data(fname_input):
    inputdata = {}
    for i in list(fname_input.keys()):
        inputdata[i] = pd.read_csv(fname_input[i], encoding="latin8")
    return inputdata


def generate_report(fname_inputs, resmap, fname_output,GWSCOD=-1):
    inputdata = load_data(fname_inputs)

    gwscod_id = inputdata["gwscod"]
    te = inputdata["te"]
    ggg = inputdata["ggg"]
    ggg = ggg[ggg["dagen_na"] != 2000]  # (SG) delete synth. dagen na of 2000

    lo = latexobject(os.path.join(resmap, fname_output))
    lo.writeheader()

    # selecteer één Specifieke gewascodes
    if GWSCOD!=-1:
        gwscod_id = gwscod_id[gwscod_id["GWSCOD"] == GWSCOD]

    create_dir(resmap, ['temp'])

    for i in gwscod_id["groep_id"].unique():
        lo = initialiseer_fiches_teelt(lo, gwscod_id, i)
        lo = compileer_teeltwisseling(lo, te, ggg, i, resmap)

    lo.close()
    lo.compile(resmap)


def initialiseer_fiches_teelt(lo, gwscod_id, i):
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
    # (SG) extract different scenario's
    te_i = te.loc[te["groep_id"] == i]
    for j in te_i["subgroep_id"].unique():
        n_groeps = len(te_i)
        lo, referenties, opmerkingen = write_eigenschappen(lo, te_i[te_i["subgroep_id"] == j], j, n_groeps)
        lo = write_gewasgroeicurve(lo, ggg, j, resmap)
        cmd = r' \textbf{Referenties:} %s' % referenties
        lo.commandnl(cmd)
        opmerkingen = opmerkingen if str(opmerkingen) != "nan" else "geen"
        cmd = r' \textbf{Opmerkingen?} %s' % opmerkingen
        lo.commandnl(cmd)
        lo.newpage()
    return lo


def write_gewasgroeicurve(lo, ggg, subgroep_id, resmap):
    plot_growth_curve(subgroep_id, ggg, resmap)
    fname = r"temp/%i" % subgroep_id
    lo.add_figure(fname)
    return lo


def write_eigenschappen(lo, eigenschappen, j, n_groeps):
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

    title = teeltwisseling if isinstance(voorwaarde, np.float) else "%s (%s)" % (teeltwisseling, voorwaarde)

    if n_groeps > 1:
        lo.writesubsection(title + r" (subgroep\_id %i)" % int(j))

    cmd = r" \textbf{%s}: %s " % ("Zaaidatum (dd/mm)", zaaidatum)
    lo.commandnl(cmd)
    cmd = r" \textbf{%s}: %s " % ("Oogstdatum (dd/mm)", oogstdatum)
    lo.commandnl(cmd)
    cmd = r" \textbf{Oogstresten}"
    lo.commandnl(cmd, vspace=0.05)
    if np.isnan(Bsi):
        cmd = r" \tab %s: /" % (r'Initi\"{e}le hoeveelheid (kg ha$^{-1}$)')
    else:
        cmd = r" \tab %s: %.2f" % (r'Initi\"{e}le hoeveelheid (kg ha$^{-1}$)', Bsi)
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
        cmd = r" \tab %s: /" % (r'Initieel percentage bedekking (\%)')
    cmd = r" \tab %s: %i" % (r'Initieel percentage bedekking (\%)', int(SCi))
    lo.commandnl(cmd, vspace=0.05)
    if np.isnan(D):
        cmd = r" \tab %s: /" % (r'Halfwaarde tijd (dagen)')
    else:
        cmd = r" \tab %s: %i" % (r'Halfwaarde tijd (dagen)', D)
    lo.commandnl(cmd, vspace=0.05)
    if np.isnan(Ri):
        cmd = r" \tab %s: /" % (r'Initi\"{e}le bodemruwheid (mm)')
    else:
        cmd = r" \textbf{%s}: %.2f" % (r'Initi\"{e}le bodemruwheid (mm)', Ri)
    lo.commandnl(cmd, vspace=0.05)

    cmd = r' \textbf{Gewasgroeicurve subgroep\_id %i:}' % subgroep_id
    lo.command(cmd)
    return lo, referenties, opmerkingen


def fix_string(string):
    string = string.replace("_", " ")
    return string


class latexobject:
    def __init__(self, fname):
        self.fname = fname + ".tex"

    def writeheader(self):
        packages = [r"\usepackage{graphicx}",
                    r"\usepackage{float}",
                    r"\usepackage[ampersand]{easylist}",
                    r"\usepackage{multicol}",
                    r"\usepackage[raggedright]{titlesec}",
                    r"\usepackage{amssymb}"]
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
        for i in packages:
            f.write(r"" + i)
            f.write("\n")
        return f

    def add_figure(self, fname):
        cmd = r'\begin{center} \begin{figure}[H] \includegraphics[width=12.5cm]{%s.png} \end{figure} \end{center}' % fname.replace(
            "\\", "/")
        self.command(cmd)

    def writesubsection(self, title):
        self.command(r"\subsection{%s}" % title.capitalize())

    def newpage(self):
        self.command(r"\newpage")

    def init_crop(self, groep_id, groep_name):
        cmd = r"\section{%s (groep\_id %i)}" % (groep_name.capitalize(), groep_id)
        self.command(cmd)

    def write_gws(self, gwsnam, gwscod, meerjarig, groenbedekker, groente):
        cmd = r"\textbf{Van toepassing op gewasnamen (en codes):} " + \
              " , ".join([gwsnam[i] + " (" + str(int(gwscod[i])) + ")" for i in range(len(gwsnam))])

        self.command(cmd)

        props = {"Meerjarig": meerjarig, "Groenbedekker": groenbedekker, "Groente": groente}

        self.write_checklist(props, vspace=0)

    def write_checklist(self, props, vspace=0.1):
        cmd = r'\begin{multicols}{3} \begin{itemize} ' \
              + ' '.join(
            [r"%s %s" % (r'\item[$\boxtimes$]', i) if props[i] == 1 else "%s %s" % (r'\item[$\square$]', i) for i
             in list(props.keys())]) + r' \end{itemize} \end{multicols}'
        self.command(cmd)

    def command(self, cmd):
        with open(self.fname, "a+") as f:
            f.write(cmd)
            f.write(" \n ")

    def commandnl(self, cmd, vspace=0.1):
        with open(self.fname, "a+") as f:
            f.write(cmd)
            f.write(r" \vspace{%.2fcm} \\" % vspace)
            f.write(" \n ")

    def close(self):
        self.command(r"\end{document} \n")

    def compile(self, fmap):
        cwd = os.getcwd()
        os.chdir(fmap)
        proc = subprocess.Popen(['pdflatex', self.fname])
        proc.communicate()
        os.chdir(cwd)


def plot_growth_curve(subgroep_id, data, resmap):
    from matplotlib import rc
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    # for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

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
    labs = [l.get_label() for l in lns]

    ylim = ax.get_ylim()[-1]
    ax.set_ylim([0, ylim])
    ax.set_yticks([0, ylim / 2, ylim])
    ax2.set_ylim([0, 100])
    ax2.set_yticks([0, 50, 100])
    # ax.set_xticklabels(np.arange(0, len(ax.get_xticks())))
    xlim = ax.get_xlim()[-1]
    if xlim < 49:
        xticks = np.arange(0, int(np.ceil(xlim)), 7)
        xticklabels = ['%i \n  (%i)' % (i, i / 7) for i in np.arange(0, int(np.ceil(xlim)), 7)]
    elif xlim < 49 * 2:
        xticks = np.arange(0, int(np.ceil(xlim)), 14)
        xticklabels = ['%i \n (%i)' % (i, i / 7) for i in np.arange(0, int(np.ceil(xlim)), 14)]
    elif xlim < 49 * 4:
        xticks = np.arange(0, int(np.ceil(xlim)), 28)
        xticklabels = ['%i \n (%i)' % (i, i / 7) for i in np.arange(0, int(np.ceil(xlim)), 28)]
    elif xlim < 49 * 8:
        xticks = np.arange(0, int(np.ceil(xlim)), 56)
        xticklabels = ['%i \n (%i)' % (i, i / 7) for i in np.arange(0, int(np.ceil(xlim)), 56)]
    elif xlim < 49 * 16:
        xticks = np.arange(0, int(np.ceil(xlim)), 112)
        xticklabels = ['%i \n (%i)' % (i, i / 7) for i in np.arange(0, int(np.ceil(xlim)), 112)]
    elif xlim < 49 * 32:
        xticks = np.arange(0, int(np.ceil(xlim)), 224)
        xticklabels = ['%i \n (%i)' % (i, i / 7) for i in np.arange(0, int(np.ceil(xlim)), 224)]
    elif xlim < 49 * 64:
        xticks = np.arange(0, int(np.ceil(xlim)), 448)
        xticklabels = ['%i \n (%i)' % (i, i / 7) for i in np.arange(0, int(np.ceil(xlim)), 448)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.legend(lns, labs, loc=0, prop={"size": 16})

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(r"Dagen (weken) na inzaai", fontsize=18)
    ax.set_ylabel(r"Hoogte (m)", fontsize=18)
    ax2.set_ylabel(r"Bedekking (\%)", fontsize=18)
    plt.savefig(r'' + os.path.join(resmap, "temp", "%i.png" % subgroep_id), dpi=500)
    # plt.savefig(os.path.join("gewasgroeicurves_plots","%i-%i-%s_%s_%s.png"
    # %(tag,subgroep_id,GWSNAM,voorwaarde,teeltwisseling)))
    plt.close()


def compute_near_identical_parcels(percelen, jaar, output_map):

    # (SG) als de betrefferende percelenkaarten nog niet ingeladen zijn als dataframe: doen
    wcond = False
    if isinstance(percelen[jaar],str):
        fnames = {}
        wcond = True
        for j in list(percelen.keys()):
            fnames[j] = percelen[j]
            percelen[j] = gpd.read_file(fnames[j])

    # (SG) check if coupling has not already been done in prior runs
    # (SG) if yes, don't overwrite
    if "NR_po" not in percelen[jaar]:

        # (SG) bepaal overlap percelen prior jaar
        intersection = near_identical_parcels(percelen[jaar], percelen[jaar - 1], output_map, "pr")
        # (SG) ... and couple codes if area overlaps 80 %
        temp = deepcopy(percelen[jaar].loc[:, ["NR"]])
        temp = temp.merge(intersection[["normalized_overlap", "NR", "NR_pr"]], on=["NR"], how="left").rename(
            columns={"normalized_overlap": "overlap_prior"})
        percelen[jaar] = percelen[jaar].merge(temp, on="NR", how="left")

        # (SG) bepaal overlap percelen posterior jaar
        intersection = near_identical_parcels(percelen[jaar], percelen[jaar + 1], output_map, "po")
        # (SG) ... and couple codes if area overlaps 80 %
        temp = deepcopy(percelen[jaar].loc[:, ["NR"]])
        temp = temp.merge(intersection[["normalized_overlap", "NR", "NR_po"]], on=["NR"], how="left").rename(
            columns={"normalized_overlap": "overlap_posterior"})
        percelen[jaar] = percelen[jaar].merge(temp, on="NR", how="left")

        # (SG) koppel perceelsnummer aan dataframe van prior en posterior jaar
        temp = deepcopy(percelen[jaar])[["NR", "NR_pr"]]
        temp = temp.rename(columns={"NR": "NR_po", "NR_pr": "NR", })[["NR", "NR_po"]]
        percelen[jaar - 1] = percelen[jaar - 1].merge(temp, on="NR", how="left")
        temp = deepcopy(percelen[jaar])[["NR", "NR_po"]]
        temp = temp.rename(columns={"NR": "NR_pr", "NR_po": "NR"})[["NR", "NR_pr"]]
        percelen[jaar + 1] = percelen[jaar + 1].merge(temp, on="NR", how="left")

        # (SG) sort for easy look-up
        percelen[jaar + 1] = percelen[jaar + 1].sort_values("NR_pr")
        percelen[jaar - 1] = percelen[jaar - 1].sort_values("NR_po")

        # (SG) write percelenkaarten back to disk
        if wcond == True:
            for j in list(percelen.keys()):
                percelen[j].to_file(fnames[j])
    return percelen


def near_identical_parcels(parcels1, parcels2, output_folder, tag, perc_overlap=80):
    """ compute overlap and intersection, filter on 80 % perc overlap

    Parameters
    ----------
        'parcels1' (gpd df ): parcel gpd df from shapefile each record holding parcel NR and its geometry, for year at interest
        'parcels2' (gpd df ): parcel gpd df from shapefile each record holding parcel NR and its geometry, for year prior or posterior to year parcels1
        'output_folder' (string): path to which files have to be written
        'tag' (string): identifier whether it is a prior (pr) or posterior year (po)
        'perc_overlap' (int): threshold of minimal overlap

    Returns
    -------
        'intersection" (gpd df): dataframe holding intersects between parcels1 and parcels2  based on 80 % overlap intersection
        (normalized with area of parcels1)
    """
    # (SG) name of files on which saga has to perform analysis
    fname1 = os.path.join(output_folder, "parcels_intersection.shp")
    fname2 = os.path.join(output_folder, "parcels_%s_intersection.shp" % tag)
    fname_temp = os.path.join(output_folder, "temp.shp")

    # (SG) write to disk
    parcels1[["NR", "geometry"]].to_file(fname1)
    parcels2["NR_%s"%tag] = parcels2["NR"]
    parcels2[["NR_%s"%tag, "geometry"]].to_file(fname2)

    # (SG) saga intersection execution
    try:
        import CNWS
    except:
        sys.exit("[Cfactor scripts ERROR] CNWS script not found, check, terminating computation Cfactor")

    CNWS.saga_intersection('"'+fname1+'"', '"'+fname2+'"','"'+fname_temp+'"')
    intersection = gpd.read_file(fname_temp)

    # (SG) couple considered parcels from this year and prior year (only consider largest intersect)
    intersection["area_intersection"] = intersection["geometry"].area

    # intersection = intersection.rename({"NR_p":})
    # (SG) keep largest intersect area per CODE_OBJ
    intersection = intersection.groupby("NR").aggregate({"area_intersection": np.max}).reset_index().merge(intersection[["NR","area_intersection","NR_%s"%tag]],
                                                                                                                on=["NR","area_intersection"],how="left")
    parcels1.loc[:,"area"] = parcels1["geometry"].area.values
    intersection = intersection.merge(parcels1[["NR", "area"]], on="NR", how="left")

    # (SG) normalize area intersction  with area of parcel of considered year
    intersection["normalized_overlap"] = intersection["area_intersection"] / intersection["area"]*100

    return intersection.loc[intersection["normalized_overlap"] > 80].drop_duplicates()

def load_perceelskaart_for_compute_C(percelenshp,jaar):
    """
    functie die percelenshp omzet naar perceelslist nodig voor initialisatie functie ```init'''

    Parameters
    ----------
        'percelenshp' (dict): dictionary met keys jaar (vb. 2016, 2017 en 2018). Elke bevatten ze de perceelskaart
        waarbij
        het perceelsnummer 'NR' aangegeven is, alsook het nummer van het bijna-gelijk perceel (80% overeenstemming in
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
    temp1 = reformat_perceelskaart_for_compute_C(df_current,jaar, ["_V", "_H", "_N"], "NR")

    # (SG) laad prior jaar (and filter on priors jaar i, which are NRs jaar-1)
    prior_NRs = df_current["NR_pr"].unique()
    prior_NRs = prior_NRs[~np.isnan(prior_NRs)]
    cols_prior = cols + ["NR", "NR_po"]
    if type(percelenshp[jaar-1]) == str:
        df_prior = gpd.read_file(percelenshp[jaar - 1])[cols_jaar]
    else:
        df_prior = deepcopy(percelenshp[jaar-1])[cols_jaar]
    df_prior = df_prior.loc[df_prior["NR"].isin(prior_NRs)]
    temp2 = reformat_perceelskaart_for_compute_C(df_prior, jaar-1, ["_H", "_N"], "NR_po")

    # (SG) laad posterior jaar
    post_NRs = df_current["NR_po"].unique();post_NRs = post_NRs[~np.isnan(post_NRs)]
    cols_post = cols + ["NR", "NR_pr"]
    if type(percelenshp[jaar+1]) == str:
        df_post = gpd.read_file(percelenshp[jaar + 1])[cols_post]
    else:
        df_post = deepcopy(percelenshp[jaar+1])[cols_post]
    df_post = df_post.loc[df_post["NR"].isin(post_NRs)]
    temp3 = reformat_perceelskaart_for_compute_C(df_post, jaar+1, ["_V"], "NR_pr")

    parcel_list = pd.concat([temp1,temp2,temp3])
    parcel_list = parcel_list[~parcel_list["GWSCOD"].isnull()]
    parcel_list = parcel_list.sort_values(["NR", "jaar", "type"])

    parcel_list["perceel_id"] = deepcopy(parcel_list["NR"])

    return parcel_list


def reformat_perceelskaart_for_compute_C(db, jaar, crop_types, rename):
    """
    Verander format van de perceelskaart als voorbereiding als input voor ComputeCFactor.py

    Parameters
    ----------
        db (pandas df): perceelskaart van jaar i
        jaar (int): zelf-verklarend
        crop_types (list): lijst van teelt types (volgens _V,_H,_N, resp voorteelt, hoofdteelt en nateelt
        rename (string): kolom dat moet hernoemt worden naar NR
    Returns
    -------
        output (pandas df): gefilterde data van gewascodes, gewasnaame, type (1: voorteelt, 2: hoofdteelt, 3: nateelt)
          en jaar
    """
    output = []
    for i in crop_types:
        temp = db[["GWSCOD"+i,"GWSNAM"+i,rename]]
        temp = temp.rename(columns={rename: "NR", "GWSNAM" + i: "GWSNAM", "GWSCOD" + i: "GWSCOD"})
        temp = temp[~temp["GWSCOD"].isnull()]
        if i == "_V":
            temp["type"] = 1
        elif i == "_H":
            temp["type"] = 2
        else:
            temp["type"] = 3
        temp['jaar'] = jaar
        output.append(temp)
    return pd.concat(output).drop_duplicates()


def init(parcel_list, fname_input, year, frequency="SMS"):
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
        sys.exit("Table link GWSCOD and groep_id not found, please check tag and filename")

    # (SG) flag parcels for which no data are available for main crop
    parcel_list = flag_incomplete_parcels(parcel_list, year)

    # (SG) get crop growth data (ggg)
    if "ggg" in keys:
        ggg = init_ggg(fname_input["ggg"])
    else:
        sys.exit("Growth curve crops not found, please check tag and filename")

    # (SG) make a time grid to perform calculations
    if ("fname_halfmonthly_rain" in keys) & ("fname_halfmonthly_temp" in keys) & ("fname_halfmonthly_R" in keys):
        grid = init_time_grid(year, fname_input["fname_halfmonthly_rain"], fname_input["fname_halfmonthly_temp"], fname_input["fname_halfmonthly_R"],frequency)
    else:
        sys.exit("Half-monthly total rainfall, total erosivity and mean temperature not found, please check tag and filename")

    return parcel_list, crop_prop, ggg, grid


def init_crop_properties(fname_crop_prop):
    # (SG) load properties of groups
    col = ["groep_id", "subgroep_id", "voorwaarde", "teeltwisseling",
           "zaaidatum", "oogstdatum", "alpha", "Bsi", "p", "Ri", "default"]
    crop_prop = pd.read_csv(fname_crop_prop, usecols=col)
    # (SG) make sure type of column  voorwaarde and teeltwisseling is a string
    for i in ["voorwaarde", "teeltwisseling"]:
        crop_prop.loc[crop_prop[i].isnull(), i] = ""
        crop_prop[i] = crop_prop[i].astype(str)
    return crop_prop


def init_groep_ids(fname_gwscod, parcel):
    # (SG) load couple matrix GWSCD/groep id
    col = ["GWSCOD", "groep_id", "voorteelt", "hoofdteelt", "nateelt", "groente",
           "groenbedekker", "meerjarige_teelt", "onvolledig"]

    if "GWSNAM" not in parcel.columns:
        col = ["GWSNAM"]+col

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
        print('Some crop inputdata are incomplete, removing records')
        parcel = parcel[parcel["onvolledig"] == 0]
    return parcel


def flag_incomplete_parcels(parcels, year):
    # (SG) check of hoofdteelt aanwezig is voor jaar j per perceel
    hoofdteelten_perceelsid = parcels.loc[(parcels["jaar"] == year) & (parcels["type"] == 2), "perceel_id"].unique()

    # (SG) filter enkel hoofdteelt en zie welke percelen geen groeps_id toegekent hebben aan de hoofdteelt
    percelen_beschikbare_gegevens = parcels.loc[~parcels["groep_id"].isnull(), "perceel_id"].unique()

    # (SG) teeltgegevens beschikbaar
    parcels["teeltgegevens_beschikbaar"] = 0.

    # (SG) toekennen
    parcels.loc[(parcels["perceel_id"].isin(percelen_beschikbare_gegevens)) & (parcels["perceel_id"].isin(hoofdteelten_perceelsid)), "teeltgegevens_beschikbaar"] = 1.

    # (SG) tag for computation in ComputeC function
    parcels["compute"] = deepcopy(parcels["teeltgegevens_beschikbaar"])
    return parcels


def init_ggg(fname_ggg):
    # (SG) load ggg_ids
    col = ["subgroep_id", "dagen_na", "bedekking(%)", "hoogte(m)", "effectieve_valhoogte(m)"]
    ggg = pd.read_csv(fname_ggg,usecols=col)
    # (SG) rename columns according to formula's page 13  (divide bedekking with 100, unit of Fc is -)
    ggg["Fc"] = ggg["bedekking(%)"] / 100
    ggg["H"] = ggg["effectieve_valhoogte(m)"]
    return ggg


def init_time_grid(year, rain, temperature, Rhm, frequency):
    # (SG) initialize rainfall, temperature and R data
    rain = pd.read_csv(rain)
    rain["timestamp"] = pd.to_datetime(rain["timestamp"], format="%d/%m/%Y")
    temperature =pd.read_csv(temperature)
    temperature["timestamp"] = pd.to_datetime(temperature["timestamp"], format="%d/%m/%Y")
    Rhm = pd.read_csv(Rhm)
    Rhm["timestamp"] = pd.to_datetime(Rhm["timestamp"], format="%d/%m/%Y")

    # (SG) identify years
    pre_date = generate_datetime_instance("01", "01", str(year - 1))
    begin_date = generate_datetime_instance("01", "01", str(year))
    end_date = generate_datetime_instance("01", "01", str(year + 1))

    # (SG) make calculation grid of two years based on frequency
    nodes = pd.date_range(pre_date, end_date, freq=frequency)
    grid = pd.DataFrame(data=nodes, index=range(len(nodes)), columns=["timestamp"])
    grid["bdate"] = pd.to_datetime(grid["timestamp"], format="%Y%m%d")
    grid["D"] = [(grid["bdate"].iloc[i]-grid["bdate"].iloc[i - 1]) for i in range(1, len(grid))] + [timedelta(days=15)]
    grid["year"] = grid["bdate"].dt.year
    grid["bmonth"] = grid["bdate"].dt.month
    grid["bday"] = grid["bdate"].dt.day
    grid["edate"] = grid["bdate"] + grid["D"]
    # (SG) remove 29th of februari, to avoid compatibility problems
    grid = grid[~((grid["bmonth"] == 2) & (grid["bday"] == 29))]

    # (SG) rename cols for rain and temperature
    rain["rain"] = np.nanmean(rain[[i for i in rain.columns if i.isdigit()]].values,axis=1)
    temperature["temp"] = np.nanmean(temperature[[i for i in temperature.columns if i.isdigit()]],axis=1)
    Rhm["Rhm"] = Rhm["value"]

    # (SG) grid merge
    rain["bday"] = rain["timestamp"].dt.day
    temperature["bday"] = temperature["timestamp"].dt.day
    Rhm["bday"] = Rhm["timestamp"].dt.day
    rain["bmonth"] = rain["timestamp"].dt.month
    temperature["bmonth"] = temperature["timestamp"].dt.month
    Rhm["bmonth"] = Rhm["timestamp"].dt.month

    grid = grid.merge(rain[["bmonth", "bday", "rain"]], on=["bmonth", "bday"], how="left")
    grid = grid.merge(temperature[["bmonth", "bday", "temp"]], on=["bmonth", "bday"], how="left")
    grid = grid.merge(Rhm[["bmonth", "bday", "Rhm"]], on=["bmonth", "bday"], how="left")

    # (SG) other properties grid
    props = ["GWSCOD", "ggg_id", "har_tag", "Ri_tag", "groep_id", "subgroep_id", "meerjarig"]
    for i in props:
        grid[i] = 0.

    # (SG) prepare output
    calcgrid = ["f1_N", "f2_EI", "Ru", "a", "Bsb", "Sp", "W", "F", "SC", "SR", "CC", "SM", "PLU", "SLR", "C"]
    for i in calcgrid:
        grid[i] = np.nan

    return grid[["bdate", "edate", "bmonth", "bday", "rain", "temp", "Rhm"]+props+calcgrid]


def compute_C(parcel, grid, ggg, cp, year, parcel_id, output_interval="M", output_map="Results", ffull_output=False):
    """
    Computes C factor based on formula's Verbist, K. (2004). Computermodel RUSLE C-factor.

    Parameters
    ----------
        'parcel' (pd df): considered parcels, see parameter ``parcel_list`` in :func:`ComputeCFactor`.
        'grid' (pd df): see parameter ``grid`` in :func:`ComputeCFactor`.
        'ggg' (pd df): see parameter ``ggg`` in :func:`ComputeCFactor`.
        'gts' (pd df): see parameters 'gts in :func:`ComputeCFactor`
        'cp' (pd df): see parameters 'cp' in :func:`ComputeCFactor`
        'year' (int): see parameters 'year' in :func:`ComputeCFactor`
        'parcel_id' (int): id of parcel
        'output_interval' (str,opt): see parameter ``output_interval`` in :func:`ComputeCFactor`.
        'output_map' (str,opt): see parameters 'output_map' in :func:`ComputeCFactor`
        'fful_output (bool,opt): see parameters 'fful_outpit' in :func:`ComputeCFactor`

    Returns
    -------
         'C' (pd df): see factor, aggregrated according to output_interval
    """

    create_dir("", [output_map])

    # (SG) prepare grid
    grid = prepare_grid(parcel, grid, ggg, cp, year, parcel_id,output_map, ffull_output=ffull_output)

    grid["f1_N"], grid["f2_EI"], grid["Ru"] = compute_Ru(grid.loc[:, "Ri_tag"].values.flatten(),
                                                         grid.loc[:, "Ri"],
                                                         grid.loc[:, "rain"],
                                                         grid.loc[:, "Rhm"])

    grid["SR"] = compute_SR(grid.loc[:, "Ru"])

    grid["a"], grid["Bsb"], grid["Sp"], grid["W"], grid["F"], grid["SC"] = compute_SC(grid.loc[:, "har_tag"],
                                                                                      grid.loc[:, "bdate"],
                                                                                      grid.loc[:, "edate"],
                                                                                      grid.loc[:, "rain"],
                                                                                      grid.loc[:, "temp"],
                                                                                      grid.loc[:, "p"],
                                                                                      grid.loc[:, "Bsi"],
                                                                                      grid.loc[:, "alpha"],
                                                                                      grid.loc[:, "Ru"])

    grid["CC"] = compute_CC(grid.loc[:, "H"], grid.loc[:, "Fc"])

    grid["SM"] = compute_SM(grid.loc[:, "SM"])

    grid["PLU"] = compute_PLU(grid.loc[:, "PLU"])

    grid["SLR"] = compute_SLR(grid.loc[:, "SC"],
                              grid.loc[:, "CC"],
                              grid.loc[:, "SR"],
                              grid.loc[:, "SM"],
                              grid.loc[:, "PLU"])

    if ffull_output:
        grid.to_csv(os.path.join(output_map, "%i_calculations.csv" % parcel_id))

    # (SG) only consider year i
    grid = grid.loc[grid["bdate"].dt.year == year]

    C = weight_SLR(grid["SLR"], grid["Rhm"], grid["bdate"], output_interval)

    return C, grid["SLR"].values.flatten(), grid["H"].values.flatten(), \
           grid["Fc"].values.flatten(), grid["Rhm"].values.flatten()


def prepare_grid(parcel, grid, ggg, cp, year, parcel_id,output_map, ffull_output=False):
    """
    Prepare grid for array-like calculations by assigning crop properties found in parcel to grid

    Parameters
    ----------
        'parcel': considered parcels, see parameter ``parcel_list`` in :func:`ComputeCFactor`.
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
            'Ri_tag'(int): number/id of roughness class (sequential in time, != Ri_id!!!).
            'H'(float): see see parameter ``ggg'` in :func:`ComputeCFactor`.
            'Fc'(int):  see see parameter ``ggg'` in :func:`ComputeCFactor`.
            'alpha'(float):  see see parameter ``cp'` in :func:`ComputeCFactor`.
            'beta'(float):  see see parameter ``cp'` in :func:`ComputeCFactor`.
            'p'(float):  see see parameter ``cp'` in :func:`ComputeCFactor`.
            'Ri'(float):  see see parameter ``cp'` in :func:`ComputeCFactor`.

    """

    # (SG) simplify crop scheme with a number of rules
    parcel, year = adjust_rotation_scheme(parcel, year, output_map, parcel_id, ffull_output=ffull_output)

    # (SG) create crop objects
    teelten = create_crop_objects(parcel, cp, year)

    # (SG) map parcel to grid
    grid = map_crops_to_grid(teelten, grid)

    # (SG) assign zero to no crops on field
    grid["H"] = 0.
    grid["Fc"] = 0.

    # (SG) assign properties
    cp = cp[["groep_id", "subgroep_id", "alpha", "Bsi", "p", 'Ri']].drop_duplicates()
    grid = grid.merge(cp, on=["groep_id", "subgroep_id"], how="left")
    grid.loc[np.isnan(grid["Ri"]), "Ri"] = Rii

    # (SG) assign growth curves
    grid = assign_growth_curvs(grid, ggg)
    return grid


def assign_growth_curvs(grid, ggg):
    for i in grid["Ri_tag"].unique():
        if i != 0:
            cond_i = (grid["Ri_tag"] == i)
            subgroep_id = grid.loc[cond_i, "subgroep_id"].values[0]
            # (SG) begin and endate
            bdate = grid.loc[(grid["subgroep_id"] == subgroep_id) & (cond_i), "bdate"].values[0]
            edate = grid.loc[(grid["subgroep_id"] == subgroep_id) & (cond_i), "edate"].values[-1]
            cond = (ggg["subgroep_id"] == subgroep_id) & (ggg["dagen_na"] < (edate-bdate) / np.timedelta64(1, 'D') + 30)
            ggg_i = deepcopy(ggg.loc[cond])
            # (SG) set datetime series format
            if (edate-bdate).astype('timedelta64[D]')/np.timedelta64(1, 'D') > ggg_i.loc[ggg_i.index[-2], "dagen_na"]:
                ggg_i.loc[ggg_i.index[-1], "dagen_na"] = (edate-bdate).astype('timedelta64[D]')/np.timedelta64(1, 'D')

            # (SG) set index and round on day
            ggg_i.index = pd.DatetimeIndex([bdate+np.timedelta64(int(ggg_i.loc[j, "dagen_na"]), 'D') for j in ggg_i.index])
            ggg_i.index = ggg_i.index.round('D')

            # (SG) append dates on which should be interpolated
            dates = grid.loc[grid["Ri_tag"] == i, "bdate"]+(grid.loc[grid["Ri_tag"] == i, "edate"] - grid.loc[grid["Ri_tag"] == i, "bdate"])/2
            dates = dates.dt.round('D')
            ggg_i = ggg_i.reindex(ggg_i.index.tolist()+[i for i in dates if i not in ggg_i.index]).sort_index()

            # (SG) Resample too slow
            ggg_i = ggg_i.interpolate()

            # (SG) Assign to grid
            grid.loc[grid["Ri_tag"] == i, ["H", "Fc"]] = ggg_i.loc[dates, ["H", "Fc"]].values
            cond = grid["meerjarig"] == True
            if np.sum(cond) != 0:
                grid.loc[cond, "H"] = np.max(ggg.loc[(ggg["subgroep_id"] == subgroep_id), "H"])
                grid.loc[cond, "Fc"] = np.max(ggg.loc[(ggg["subgroep_id"] == subgroep_id), "Fc"])
    return grid


def adjust_rotation_scheme(parcel, year, output_map, parcel_id, ffull_output=False):
    """
    Hard-coded simplifications of crop rotation scheme
    Implemented to simplify build-up grid in in :func:`map_parcel_to_grid`.
    NOTA: alle uitzonderingen mbt wintergranen etc worden hier geimplementeerd!

    Parameters
    ----------
        'parcel': (pd df)  considered parcel, see parameter ``parcel_list`` in :func:`ComputeCFactor`.
        'year' (int): see parameters 'year' in :func:`ComputeCFactor`

    Returns
    -------
        'parcel' (pd df): considered parcel, see parameter ``parcel_list`` in :func:`ComputeCFactor`.
                        updated by removed rows/records (simplication scheme)

    """
    # max_year!=jaar!!!!!!"
    # (SG) Only consider crop with highest type id for year prior

    max_type = np.max(parcel.loc[(parcel["jaar"] == year-1), "type"])
    cond = ((parcel["jaar"] == year-1) & (parcel["type"] == max_type))
    parcel = parcel.loc[cond].append(parcel.loc[(parcel["jaar"] == year) | (parcel["jaar"] == year+1)])

    # (SG) Filter parcel list based on whether a crop can be a specfic type
    parcel = filter_types(parcel)

    # (SG) Als er geen nateelt is in jaar i, beschouw dan voorteelt i+1 als nateelt i
    # (SG) Als er een nateelt is in jaar i, verwijder dan voorteelt i+1
    parcel = exceptions_voorteelt_nateelt(parcel, year)

    # (SG) hard gecodeerde uitzonderingen voor wintergraan
    parcel = exceptions_winterteelten(parcel, year)

    # (SG) hard gecodeerde uitzonderingen voor meerjarige teelten
    parcel, year = exceptions_meerjarigeteelten(parcel, year)

    # (SG) wanneer conflict is tussen hoofdteelt jaar j en nateelt jaar j-1, vertrouw hoofdteelt jaar j
    if np.sum((parcel["jaar"] == year-1) & (parcel["type"] == 3)) == 2:
        parcel = parcel.iloc[1:]

    # (SG) filter teelten die in jaar j-2 gepositioneerd zijn (vb van hoofdteelt j-1 naar nateelt jaar j-2 geplaatst)
    parcel = parcel[parcel["jaar"] != year-2]

    # (SG) als er conflicten afgeleid zijn uit vereenvoudigingen (twee of drie types gedefinieerd in eenzelfde jaar),
    # neem dan de laatste teelt per koppel (jaar,type): dat wordt beschouwd als het meest betrouwbaar
    parcel = parcel.drop_duplicates(subset=["jaar", "type"], keep="last")

    # (SG) Only consider crop with highest type id for year prior
    max_type = np.max(parcel.loc[(parcel["jaar"] == year-1), "type"])
    parcel = parcel.loc[(parcel["jaar"] == year-1) & (parcel["type"] == max_type)].append(parcel.loc[(parcel["jaar"] == year) | (parcel["jaar"] == year+1)])

    # (SG) print parcel to disk
    if ffull_output:
        parcel.to_csv(os.path.join(output_map, "%i_rotation_scheme_simplified.csv" % int(parcel_id)))

    return parcel, year


def filter_types(parcel):
    # (SG) filter hoofdteelt, nateelten en voorteelten
    cond = (parcel["type"] == 1) & (parcel["voorteelt"] == 0) | (parcel["type"] == 2) & (parcel["hoofdteelt"] == 0) | (parcel["type"] == 3) & (parcel["nateelt"] == 0)
    return parcel[~cond]


def exceptions_voorteelt_nateelt(parcel, year):
    # (SG) als er conflict is tussen de voorteelt jaar j (j+1) en nateelt j-1 (j) geloof de voorteelt jaar (j+1)
    for i in [year, year+1]:
        cond_voorteelt = (parcel["type"] == 1) & (parcel["jaar"] == i)
        cond_nateelt = (parcel["type"] == 3) & (parcel["jaar"] == i-1)
        cond_groente = (parcel["groente"] == 1) & (parcel["jaar"] == i)

        GWSCOD_voorteelt = parcel.loc[cond_voorteelt, "GWSCOD"]
        GWSCOD_nateelt = parcel.loc[cond_nateelt, "GWSCOD"]

        if len(GWSCOD_voorteelt) > 0:
            if len(GWSCOD_nateelt) > 0:
                # (SG) als GWSCOD nateelt gelijk is aan GWSCOD voorteelt: behoud enkel nateelt
                if GWSCOD_nateelt.values[0] == GWSCOD_voorteelt.values[0]:
                    parcel = parcel[~cond_voorteelt]
                # (SG) als GWSCOD nateelt niet gelijk is aan GWSCOD voorteelt:
                # verwijder nateelt en maak voorteelt nateelt ALS voorteelt geen groente is!
                # (SG) de voorteelt in de perceelskaart wordt altijd geloofd!
                else:
                    if parcel.loc[(parcel["GWSCOD"] == GWSCOD_voorteelt.values[0]) & (parcel["jaar"] == i), "groente"].values[0] == 0:
                        parcel = deepcopy(parcel[~cond_nateelt])
                        parcel.loc[cond_voorteelt, ["type", "jaar"]] = [3, i-1]
            # (SG) als er geen nateelt is zet dan de voorteelt gelijk aan nateelt
            else:
                parcel.loc[cond_voorteelt, ["type", "jaar"]] = [3, i-1]
        # (SG) als er geen voorteelt is: doe niets :)
    return parcel


def exceptions_winterteelten(parcel, year):
    groep_id_wintercrops = [8, 9]

    if np.sum(parcel["groep_id"].isin(groep_id_wintercrops)) > 0:
        for i in groep_id_wintercrops:
            # (SG) Als de hoofdteelt gelijk is aan een wintergewas, maak het dan een nateelt:
            cond = (parcel["groep_id"] == i) & (parcel["type"] == 2)
            parcel.loc[cond, "type"] = 3
            parcel.loc[cond, "jaar"] = parcel.loc[cond, "jaar"] - 1

    return parcel


def exceptions_meerjarigeteelten(parcel, year):

    # (SG) Als een teeltschema enkel één type gewas bevat, en het is meerjarig,
    # beschouw dan enkel hoofdteelten en reken enkel jaar-1 door!
    # (SG) als er enkel gegevens zijn over teelt jaar i beschouw het dan als permanent grasland
    # (SG) beschouw enkel hoofdgewassen van meerjarige teelten
    parcel["meerjarig"] = 0.

    # (SG) verwijder twee opeenvolgende meerjarige teelten, behoud de eerste!
    if np.sum(parcel["meerjarige_teelt"] == 1) > 1:

        cond = [True] + [False if ((parcel["groep_id"].iloc[i-1] == parcel["groep_id"].iloc[i]) & (parcel["meerjarige_teelt"].iloc[i] == 1)) else True for i in range(1, len(parcel), 1)]
        parcel = parcel[cond]

    # if np.sum(parcel["meerjarige_teelt"]==1)==len(parcel):
    #     # (SG) enkel hoofdgewassen (tenzij er geen hoofgewassen zijn, doe dan een aanpassing aan type)
    #     temp = deepcopy(parcel)
    #     parcel = parcel[parcel["type"] == 2]
    #     #(SG) voeg meerjaar gewas toe
    #     if len(parcel) == 1:
    #         year_ = parcel["jaar"].unique()[0]
    #         parcel = parcel.append(deepcopy(parcel.iloc[0])).reset_index()
    #         assign_year = year_ - 1 if year_ == year else year_ + 1
    #         parcel.loc[parcel.index[0], "jaar"] = assign_year
    #         parcel.loc[parcel.index[0],"meerjarig"] = 1
    #     #(SG) als de parcel_list empty was dan betekent dit dat gras enkel als voor en nateelt beschreven was,
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
    Function to apply string on parcel dataframe which implements the simplication of the rotatop, scheme

    Parameters
    ----------
        'parcel' (pd df): considered parcels, see parameter ``parcel_list`` in :func:`ComputeCFactor`.
        'statement' (string): condition which should be applied to dataframe parcel
        'max_year' (int): maximum year found in specific parcel (not per se equal to year of simulation (
        e.g. not crops reported for specific year))

    Returns
    -------
        'cond' (list): list of bool stating wether to 'keep' (true) or remove (false) record/row of df

    """
    indices = parcel.index
    cond = []
    if "i+1" in statement:
        for i in range(0, len(indices)-1, 1):
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
    # (SG) get 'hoofdteelt' i
    groep_id = parcel.loc[(parcel["type"] == 2) & (parcel["jaar"] == year) & (parcel["meerjarig"] == 0), "groep_id"]
    GWSCOD = parcel.loc[(parcel["type"] == 2) & (parcel["jaar"] == year) & (parcel["meerjarig"] == 0), "GWSCOD"]
    groenbedekker = parcel.loc[(parcel["type"] == 2) & (parcel["jaar"] == year), "groenbedekker"]

    if len(groep_id) > 0:
        hoofdteelt = Crop("hoofdteelt", year, cp, groep_id.values[0],
                          GWSCOD.values[0], groenbedekker=groenbedekker.values[0])
    else:
        hoofdteelt = None

    # (SG) get 'hoofdteelt' i-1
    groep_id = parcel.loc[(parcel["type"] == 2) & (parcel["jaar"] == year-1), "groep_id"]
    GWSCOD = parcel.loc[(parcel["type"] == 2) & (parcel["jaar"] == year-1), "GWSCOD"]
    meerjarig = parcel.loc[(parcel["type"] == 2) & (parcel["jaar"] == year-1), "meerjarige_teelt"]
    groenbedekker = parcel.loc[(parcel["type"] == 2) & (parcel["jaar"] == year-1), "groenbedekker"]

    if len(groep_id) > 0:
        hoofdteelt_prior = Crop("hoofdteelt", year-1, cp, groep_id.values[0], GWSCOD.values[0],
                                meerjarig=meerjarig.values[0], groenbedekker=groenbedekker.values[0])
        # (SG) pas oogstdatum voor hoofdteelt_prior aan naar laatste jaar
        # (SG) als hoofdteelt prior groendbedekker en hoofdteelt niet, pas oogstdatum hoofdteelt prior aan
        if hoofdteelt is None:
            # (SG) hoofdteelt bestaat niet, alsook  voorteelt/nateelt jaar j niet
            cond = (hoofdteelt_prior.meerjarig == True) & (np.sum((parcel["type"] == 1) & (parcel["jaar"] == year)) == 0) & (np.sum((parcel["type"] == 3) & (parcel["jaar"] == year)) == 0)
            if cond:
                hoofdteelt_prior.harvest_date = generate_datetime_instance("01", "01", year + 1)
        else:
            if (hoofdteelt_prior.groenbedekker == 1):
                hoofdteelt_prior.harvest_date = hoofdteelt.sowing_date
            cond = (hoofdteelt_prior.meerjarig == True) & (np.sum((parcel["type"] == 1) & (parcel["jaar"] == year)) == 0)
            if cond:
                hoofdteelt_prior.harvest_date = hoofdteelt.sowing_date
    else:
        hoofdteelt_prior = None

    # (SG) get 'nateelt' i
    groep_id = parcel.loc[(parcel["type"] == 3) & (parcel["jaar"] == year), "groep_id"]
    GWSCOD = parcel.loc[(parcel["type"] == 3) & (parcel["jaar"] == year), "GWSCOD"]
    groenbedekker = parcel.loc[(parcel["type"] == 3) & (parcel["jaar"] == year), "groenbedekker"]

    if len(groep_id) > 0:
        nateelt = Crop("nateelt", year, cp, groep_id.values[0],
                       GWSCOD.values[0], groenbedekker=groenbedekker.values[0])
        if hoofdteelt is not None:
            hoofdteelt, nateelt = fit_dates_hoofd_nateelt(hoofdteelt, nateelt)
    else:
        nateelt = None

    # (SG) get 'voorteelt' i
    groep_id = parcel.loc[(parcel["type"] == 1) & (parcel["jaar"] == year), "groep_id"]
    GWSCOD = parcel.loc[(parcel["type"] == 1) & (parcel["jaar"] == year), "GWSCOD"]

    if len(groep_id) > 0:
        voorteelt = Crop("voorteelt", year, cp, groep_id.values[0], GWSCOD.values[0])
        if hoofdteelt is not None:
            hoofdteelt, voorteelt = fit_harvest_date(hoofdteelt, voorteelt)
    else:
        voorteelt = None

    # (SG) get 'nateelt_prior' i
    groep_id = parcel.loc[(parcel["type"] == 3) & (parcel["jaar"] == year-1), "groep_id"]
    GWSCOD = parcel.loc[(parcel["type"] == 3) & (parcel["jaar"] == year - 1), "GWSCOD"]
    groenbedekker = parcel.loc[(parcel["type"] == 3) & (parcel["jaar"] == year-1), "groenbedekker"]

    if len(groep_id) > 0:
        nateelt_prior = Crop("nateelt", year-1, cp, groep_id.values[0],
                             GWSCOD.values[0], groenbedekker=groenbedekker.values[0])
        if hoofdteelt_prior is not None:
            hoofdteelt_prior, nateelt_prior = fit_dates_hoofd_nateelt(hoofdteelt_prior, nateelt_prior)

        if hoofdteelt is not None:
            if voorteelt is None:
                hoofdteelt, nateelt_prior = fit_harvest_date(hoofdteelt, nateelt_prior)
            else:
                voorteelt, nateelt_prior = fit_harvest_date(voorteelt, nateelt_prior)
        else:
            if (voorteelt is None) & (nateelt is not None):
                nateelt, nateelt_prior = fit_harvest_date(nateelt, nateelt_prior)
    else:
        nateelt_prior = None

    # (SG) if harvest date hoofdteelt is not filled, set it equal to 01/01 next year
    if hoofdteelt is not None:
        if hoofdteelt.harvest_date is None:
            hoofdteelt.harvest_date = generate_datetime_instance("01", "01", hoofdteelt.year + 2)

    # (SG) if harvest date nateelt is not filled, set it equal to 01/01 next year
    if nateelt is not None:
        if nateelt.harvest_date is None:
            nateelt.harvest_date = generate_datetime_instance("01", "01", nateelt.year + 2)

    # (SG) if harvest date nateelt_prior is not filled, set it equal to 01/01 next year
    if nateelt_prior is not None:
        if nateelt_prior.harvest_date is None:
            nateelt_prior.harvest_date = generate_datetime_instance("01", "01", nateelt_prior.year + 3)

    if hoofdteelt_prior is not None:
        if hoofdteelt_prior.harvest_date is None:
            if (nateelt is not None) & (hoofdteelt is None) & (voorteelt is None):
                hoofdteelt_prior.harvest_date = nateelt.sowing_date

    teelten = {}
    teelten["hoofdteelt"] = hoofdteelt
    teelten["hoofdteelt_prior"] = hoofdteelt_prior
    teelten["nateelt"] = nateelt
    teelten["nateelt_prior"] = nateelt_prior
    teelten["voorteelt"] = voorteelt
    return teelten

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
    #        hoofdteelt_prior, nateelt_prior = fit_dates_hoofd_nateelt(hoofdteelt_prior, nateelt_prior)
    #    if voorteelt!=None:
    #        nateelt_prior.fit_sowing_date(voorteelt.sowing_date)
    #    else:
    #        if hoofdteelt!=None:
    #            nateelt_prior.fit_sowing_date(hoofdteelt.sowing_date)
    # else:
    #    nateelt_prior= None


class Crop:
    def __init__(self, type, year, cp, groep_id, GWSCOD, meerjarig=False, groenbedekker=0):

        self.type = type
        self.year = year
        self.groep_id = groep_id
        self.source = cp.loc[cp["groep_id"] == self.groep_id]
        self.GWSCOD = GWSCOD
        self.meerjarig = meerjarig
        self.groenbedekker = groenbedekker

        self.check_conditions()

        # (SG) uitzondering brouwersgerst, deze worden pas toegekent na vergelijken nateelt_prior en hoofdteelt
        self.set_default_values()

    def check_conditions(self):
        # (SG) filter subgroeps based on condition on type (vb. gras is ingeschreven als hoofdteelt)
        if np.sum(self.source["teeltwisseling"].isin(["hoofdteelt", "nateelt", "voorteelt"])) > 0:
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
        self.default_sow_harvest_date()
        self.default_growth_curve()

    def default_sow_harvest_date(self):
        cond = self.source["default"] == 1
        self.sowing_date_int = int(self.source.loc[cond, "zaaidatum"].values[0])
        self.harvest_date_int = int(self.source.loc[cond, "oogstdatum"].values[0])
        # (SG) convert to datetime object
        self.create_datetime_objects_sowingharvest()

    def default_growth_curve(self):
        cond = self.source["default"] == 1
        self.subgroep_id = self.source.loc[cond, "subgroep_id"].values[0]

    def create_datetime_objects_sowingharvest(self):
        # (SG) als er een nateelt is en deze is een groenbedekker, kap dan de oogstdatum van het hoofdgewas a
        self.sowing_date = self.create_datetime_objects(self.sowing_date_int)
        if self.harvest_date_int != -9999.:
            self.harvest_date = self.create_datetime_objects(self.harvest_date_int)
            # (SG)  if harvest date before sowing date: correct
            if self.harvest_date < self.sowing_date:
                self.harvest_date = self.harvest_date + timedelta(days=365)
        else:
            self.harvest_date = None

    def update_sowhardate(self, sowingdate, harvestdate):
        self.update_sowingdate(sowingdate)
        self.update_harvestdate(harvestdate)

    def update_sowingdate(self, inputdate):
        """
        Function to change sowing date

        Parameters
        ----------
            'inputdate' (datimetime object or int): datetime/int object that should be used to update sowing date.
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
            'inputdate' (datimetime object or int): datetime/int object that should be used to update harvest date.
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
        date = str(int(date))
        dd = date[-2:]
        mm = date[4:6]
        yyyy = self.year
        return generate_datetime_instance(dd, mm, yyyy)

    def extract_non_default_growth_curve(self, indices):
        # (SG) loop over sowing dates
        for i in indices:
            string = self.source.loc[i, "voorwaarde"]
            [lower_bound, upper_bound] = self.extract_bounds_from_string(string)
            if (int(str(self.sowing_date_int)[4:]) >= lower_bound) & \
                    (int(str(self.sowing_date_int)[4:]) <= upper_bound):
                self.subgroep_id = self.source.loc[i, "subgroep_id"]

    def extract_bounds_from_string(self, string):
        lower_bound = int(string[string.index("[") + 1:string.index(",")])
        upper_bound = int(string[string.index(",") + 1:string.index("]")])
        return lower_bound, upper_bound


def fit_dates_hoofd_nateelt(hoofdteelt, nateelt):
    WINTERGRAAN = [8, 9, 10]
    # (SG) check of de hoofdteelt wel een oogstdatum gedefinieerd heeft
    # (SG) zoniet, stel die gelijk aan de default zaaidatum van de nateelt

    if hoofdteelt.harvest_date is None:
        hoofdteelt.harvest_date = nateelt.sowing_date

    # (SG) Als de oogstdatum van de hoofdteelt na de zaaidatum ligt van de nateelt,
    # plaats dan de zaaidatum van de nateelt 15 dagen na de oogstdatum van de hoofdteelt
    threshold_date = generate_datetime_instance(15, 10, hoofdteelt.year)
    if (nateelt.sowing_date <= hoofdteelt.harvest_date) & (hoofdteelt.groenbedekker == 0):
        nateelt.update_sowingdate(hoofdteelt.harvest_date + timedelta(days=15))
    # (SG) Als de nateelt een wintergraan is en de hoofdteelt oogstdatum na 15 oktober ligt
    # pas dan de oogst en zaaidatum aan = 15 oktober (threshold_date)
    elif (nateelt.sowing_date <= hoofdteelt.harvest_date) & (hoofdteelt.groenbedekker != 0):
        hoofdteelt.update_harvestdate(nateelt.sowing_date)
        nateelt.update_sowingdate(hoofdteelt.harvest_date)
    elif (nateelt.groep_id in WINTERGRAAN) & (hoofdteelt.harvest_date >= threshold_date):
        nateelt.update_sowingdate(threshold_date)
        hoofdteelt.update_harvestdate(threshold_date)

    # (SG) selecteer de gepaste gewasgroeicurve via subgroup_id
    indices = [i for i in nateelt.source["voorwaarde"] if "inzaaidatum" in i]
    indices = nateelt.source[nateelt.source["voorwaarde"].isin(indices)].index
    if len(indices) > 0:
        nateelt.extract_non_default_growth_curve(indices)
    return hoofdteelt, nateelt


def fit_harvest_date(hoofdteelt, nateelt_prior):

    GRASSEN = [16]
    MAIS = [1, 2]

    # (SG) implenteer uitzondering voor brouwersgerst
    # if (hoofdteelt.groep_id == 10):
    #    if (nateelt_prior.groep_id == 10):
    #        nateelt_prior.source = nateelt_prior.source[nateelt_prior.source["teeltwisseling"] == "voorwaarde1"]
    #        nateelt_prior.set_default_values()
    #    else:
    #        hoofdteelt.source = hoofdteelt.source[hoofdteelt.source["teeltwisseling"] == "voorwaarde2"]
    #        hoofdteelt.set_default_values()

    # (SG) in geval dat de inzaaidatum van de groenbdekker afhangt van de hoofdteelt
    if nateelt_prior.harvest_date is None:
        nateelt_prior.harvest_date = hoofdteelt.sowing_date

    # (SG) Als de zaaidatum van de hoofdteelt voor de oogstdatum van de voorteelt ligt,
    if nateelt_prior.harvest_date >= hoofdteelt.sowing_date:
        # (SG) En de groep_id duidt op en gras
        # Dan moet het gras geoogst worden 15 dagen voor de zaaidatum van het hoofdgewas (behalve bij mais)
        if nateelt_prior.groep_id in GRASSEN:
            nateelt_prior.harvest_date = hoofdteelt.sowing_date if hoofdteelt.groep_id in MAIS else hoofdteelt.sowing_date - timedelta(days=15)
        else:
            nateelt_prior.harvest_date = hoofdteelt.sowing_date

    return hoofdteelt, nateelt_prior


def map_crops_to_grid(teelten, grid):
    """
    Prepare grid for array-like calculations by assigning crop properties found in parcel to grid

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
        bcrop, ecrop = allocate_grid(grid, hoofdteelt_prior.sowing_date, hoofdteelt_prior.harvest_date)
        cond = (bcrop <= grid["bdate"]) & (ecrop > grid["edate"])
        Ri_tag += 1
        grid.loc[cond, ["groep_id", "subgroep_id", "GWSCOD", 'Ri_tag', 'har_tag', "meerjarig"]] \
            = [hoofdteelt_prior.groep_id, hoofdteelt_prior.subgroep_id, hoofdteelt_prior.GWSCOD, Ri_tag, 0, hoofdteelt_prior.meerjarig]

    # (SG) map nateelt prior to grid
    # ken een Ri_id toe, en de ggg_id voor gewasgroei
    # NOTA: als er geen nateelt is, laat gewastresten hoofdteelt prior op terrein liggen!
    if nateelt_prior is not None:
        bcrop, ecrop = allocate_grid(grid, nateelt_prior.sowing_date, nateelt_prior.harvest_date)
        cond = (bcrop <= grid["bdate"]) & (ecrop > grid["edate"])
        Ri_tag += 1
        grid.loc[cond, ["groep_id", "subgroep_id", "GWSCOD", 'Ri_tag', 'har_tag', "meerjarig"]] \
            = [nateelt_prior.groep_id, nateelt_prior.subgroep_id, nateelt_prior.GWSCOD, Ri_tag, 0, nateelt_prior.meerjarig]
    else:
        # (SG) map harvest remains of hoofdteelt proir to grid
        if hoofdteelt_prior is not None:
            cond = (ecrop <= grid["edate"])
            har_tag += 1
            # (SG) Ri van hoofdteelt laten doorgaan!!!
            grid.loc[cond, ["groep_id", "subgroep_id", "GWSCOD", 'har_tag', 'Ri_tag', "meerjarig"]] \
                = [hoofdteelt_prior.groep_id, hoofdteelt_prior.subgroep_id, hoofdteelt_prior.GWSCOD, har_tag, 0, hoofdteelt_prior.meerjarig]

    # (SG) map hoofdteelt naar het grid
    # ken een Ri_id toe, en de ggg_id voor gewasgroei
    if hoofdteelt is not None:
        bcrop, ecrop = allocate_grid(grid, hoofdteelt.sowing_date, hoofdteelt.harvest_date)
        cond = (bcrop <= grid["bdate"]) & (ecrop > grid["edate"])
        Ri_tag += 1
        grid.loc[cond, ["groep_id", "subgroep_id", "GWSCOD", 'Ri_tag', 'har_tag', "meerjarig"]] \
            = [hoofdteelt.groep_id, hoofdteelt.subgroep_id, hoofdteelt.GWSCOD, Ri_tag, 0, hoofdteelt.meerjarig]
    else:
        # (SG) voeg harvest remains van nateelt vorig jaar toe
        if nateelt_prior is not None:
            cond = (ecrop <= grid["edate"])
            har_tag += 1
            grid.loc[cond, ["groep_id", "subgroep_id", "GWSCOD", 'har_tag', 'Ri_tag', "meerjarig"]] \
                = [nateelt_prior.groep_id, nateelt_prior.subgroep_id, nateelt_prior.GWSCOD, har_tag, 0, nateelt_prior.meerjarig]

    # (SG) map nateelt to grid
    # ken een Ri_id toe, en de ggg_id voor gewasgroei
    # NOTA: als er geen nateelt is, laat gewastresten hoodteelt op terrein liggen!
    if nateelt is not None:
        bcrop, ecrop = allocate_grid(grid, nateelt.sowing_date, nateelt.harvest_date)
        cond = (bcrop <= grid["bdate"]) & (ecrop >= grid["edate"])
        Ri_tag += 1
        grid.loc[cond, ["groep_id", "subgroep_id", "GWSCOD", 'Ri_tag', 'har_tag', "meerjarig"]] \
            = [nateelt.groep_id, nateelt.subgroep_id, nateelt.GWSCOD, Ri_tag, 0, nateelt.meerjarig]
    else:
        if hoofdteelt is not None:
            cond = (ecrop <= grid["edate"])
            har_tag += 1
            grid.loc[cond, ["groep_id", "subgroep_id", "GWSCOD", 'har_tag', 'Ri_tag', "meerjarig"]] \
                = [hoofdteelt.groep_id, hoofdteelt.subgroep_id, hoofdteelt.GWSCOD, har_tag, 0, hoofdteelt.meerjarig]

    # (SG) map voorteelt to grid
    # ken een Ri_id toe, en de ggg_id voor gewasgroei
    #  if zaaidatumV is not None:
    #      bcrop, ecrop = allocate_grid(grid, zaaidatumV, oogstdatumV)
    #      cond = (bcrop <= grid["bdate"]) & (ecrop > grid["edate"])
    #      Ri_tag += 1
    #      grid.loc[cond, ["subgroep_id", "GWSCOD", 'Ri_tag', 'har_tag']] = [ggg_idV, 0, Ri_tag, 0]
    # else:
    #    bcrop, ecrop = allocate_grid(grid, oogstdatumN_, zaaidatumH)
    #    cond = (bcrop <= grid["bdate"]) & (ecrop > grid["edate"])
    #    har_tag += 1
    #    grid.loc[cond, ["ggg_id", "GWSCOD", 'har_tag', 'groep_id']] = [ggg_idH, 0, har_tag, groep_idV]

    return deepcopy(grid.iloc[:-1])


def get_dates_crop(gts, cond, parcel, year, zaaidatum=None):
    """
    Get sowing and harvest dates crop rotation scheme (gts) from parcel on record/row conditioned by 'cond'

    Parameters
    ----------
        'grid': see parameter ``grid`` (Returns) in :func:`prepare_grid`.
        'gts' (pd df): see parameters 'gts in :func:`ComputeCFactor`
        'cond' (string): condition which states which rows/records  parcel should be considered.
        'year' (int): see parameters 'year' in :func:`ComputeCFactor`
        'zaaidatum' (int): sowing date which is based on harvest date of another crop (when used: only groep_id is important to know (see return!)

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
        [zaaidatum, dagen_tot_oogst, ggg_id] = gts.loc[index, ["zaaidatum1", "dagen_tot_oogst", "ggg_id"]].values[0]
        zaaidatum = string_to_date(zaaidatum, year)
        oogstdatum = zaaidatum + timedelta(days=int(dagen_tot_oogst))
        return zaaidatum, oogstdatum, ggg_id, groep_id
    else:
        zaaidatum_int = int("1900"+zaaidatum.strftime("%m%d"))
        cond = (gts["zaaidatum1"] <= zaaidatum_int) & (gts["zaaidatum2"] > zaaidatum_int) & (gts["groep_id"] == groep_id)
        # (SG) als de 'cond' series allemaal False is: you're in trouble.
        # Neenee, dit betekent dat de laatste zaai-instantie gebruikt moet worden, i.e. niet gebonden door einddatum!
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
    output = datetime.strptime(str(int(year))+str(input)[4:], '%Y%m%d')

    return output


def compute_Ru(Ri_tag, Ri, rain, Rhm):
    """
    Computes parcel roughness  Ru = 6.096+[Dr*(Ri-6.096)]

    Parameters
    ----------
        Ri_id (pd series, float): see series ``grid`` in :func:`ComputeCFactor` and :func:`prepare_grid`
        Ri (pd series, float): see series ``grid`` in :func:`ComputeCFactor` and :func:`prepare_grid`
        Rain (pd series, float): see series ``grid`` in :func:`ComputeCFactor`
        EI30 (pd series, float): see series ``grid`` in :func:`ComputeCFactor`

    Returns
    -------
        'Ru' (series, float): soil roughness
        'f1_N' (series, float): first part formula, cumulative rainfall, weighted with factor
        'f2_EI' (series, float): second part formula, cumulative rainfall erosivity, weighted with factor

    (SG) Koen Verbist( 2004)
    Dr = exp(0.5*(-0.0055*Pt)+0.5*(-0.0705*EIt))
    met EIt in MJ mm/(ha jaar)
    Koen Verbist, factor 0.0705 in documentatie is fout (zie hierboven), gezien in codering: 0.012/17.02=0.000705
    Dr[date] = Math.exp(0.5*(-0.14*(cumulOperationP/25.4))+0.5*(-0.012*(cumulOperationEI/17.02)));
    met EIt in (MJ m)/(mm jaar) (volgt uit code, maar deze eenheid klopt niet?)
    Inputwaarden in code zijn erosiviteitswaarden (jaargemiddelden of jaar), en deze kloppen volgens grootte-orde
    (eenheid: (MJ mm)/(ha h) zoals in  Verstraeten, G., Poesen, J., Demarée, G., & Salles, C. (2006). Long-term (105 years) variability " \
                in rain erosivity as derived from 10-min rainfall depth data for Ukkel (Brussels, Belgium): Implications for assessing " \
                soil erosion rates. Journal of Geophysical Research Atmospheres, 111(22), 1–11.)"""
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
    return np.exp(0.5 * f1_N + 0.5 * f2_EI)


def compute_SR(Ru):
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
        bdate_id (pd series, datetime): see series ``grid`` in :func:`ComputeCFactor`
        edate_id (pd series, datetime): see series ``grid`` in :func:`ComputeCFactor`
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
        'W' and 'F' (pd series, float): coefficients weighing rainfall and temperature
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
        W[cond], F[cond], a_val[cond] = compute_a(rain.loc[cond], temp.loc[cond], p.loc[cond])
        # (SG) compute Bsi
        Bsb[cond] = compute_Bsb(bdate.loc[cond], edate.loc[cond], a_val[cond], Bsi.loc[cond])
        # (SG) compute Sp (divide by 10 000 Bsi (kg/ha -> kg/m2)
        Sp[cond] = 100 * (1 - np.exp(-alpha[cond] * Bsb[cond] / (100**2)))
        # (SG) compute SC
        SC[cond] = np.exp(-b * Sp[cond] * ((6.096 / (Ru[cond]))**0.08))
        # SC[cond] = [math.exp(-b*Sp[i]*((6.069 / (Ru[i]**0.08)))) for i in range(len(Sp))]
    return a_val, Bsb, Sp, W, F, SC


def compute_CC(H, Fc):
    """
    Computes crop cover factor based on formula's Verbist, K. (2004). Computermodel RUSLE C-factor.

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
    Computes soil moisture factor based on formula's Verbist, K. (2004). Computermodel RUSLE C-factor.

    Parameters
    ----------
        'grid' (pd df): see parameter ``grid`` in :func:`ComputeCFactor`.
    Returns
    -------
        'SM' (pd series, float): soil moisture
    """
    """
    Verbist et al. (2004): Deze factor brengt de invloed in rekening van het bodemvochtgehalte op de watererosie.
    Deze parameter moet enkel veranderd worden ingeval de bodemvochtsituatie gedurende het jaar significant verschilt
    van een situatie waarbij het perceel het hele jaar braak wordt gelaten en jaarlijks wordt geploegd.
    Voor een normaal akkerperceel wordt de waarde één voorgesteld.
    Deze waarde wordt dan ook gebruikt bij de berekening van de gewasfactoren voor akkerbouwpercelen in Vlaanderen,
    zodat deze subfactor geen rol speelt in de berekeningen
    """

    return [1]*len(grid)


def compute_PLU(grid):
    """
    Computes prior land use factor based on formula's Verbist, K. (2004). Computermodel RUSLE C-factor.

    Parameters
    ----------
        'grid' (pd df): see parameter ``grid`` in :func:`ComputeCFactor`.
    Returns
    -------
        'PLU' (pd series, float): past land use
    """

    """
    Verbist et al. (2004): Volgens Verstraeten et al. (2002) kan deze subfactor geschat worden tussen 0,9 en 1 voor
    een jaarlijks geploegde bodem. In de berekening van de gewasfactor wordt dan ook verder een rekening gehouden met
    deze subfactor (de subfactor PLU wordt gelijkgesteld aan 1).
    """

    return [1]*len(grid)


def compute_SLR(SC, CC, SR, SM, PLU):
    """
    Computes soil loss ration factor based on formula's Verbist, K. (2004). Computermodel RUSLE C-factor.

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


def weight_SLR(SLR, EI30, bdate, output_interval):
    """
    Weight SLR factor based on formula's Verbist, K. (2004). Computermodel RUSLE C-factor.

    Parameters
    ----------
        'SLR' (pd series, float): soil loss ratio
        'EI30' (pd series, float): see series ``grid`` in :func:`ComputeCFactor`
        'bdate' (pd series, float): see series ``grid`` in :func:`ComputeCFactor
        'output_interval' (str): output interval for C-factor (!= interval computation grid), either monthly 'M',
                                semi-monthly'SMS' or yearly 'Y'
    Returns
    -------
         'grid' (pd df): see parameter ``grid`` in :func:`ComputeCFactor`.
    """
    # (SG) compute sum of average EI30 values
    sumR = np.sum(EI30)
    product = (SLR * EI30)
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
        'W' and 'F' (pd series, float): coefficients weighing rainfall and temperature
    """
    R0 = 25.76  # minimum gemiddelde halfmaandelijks neerslag nodig voor afbraak (opm. pagina 6?)
    T0 = 32.22  # celsius!
    A = 7.78  # celsius!

    T0 = celc_to_fahr(T0)
    A = celc_to_fahr(A)
    # (SG) compute W
    W = rain / R0
    # (SG) compute F (in fahr!)
    temp = celc_to_fahr(temp)
    F = (2 * ((temp + A) ** 2) * ((T0 + A) ** 2) - (temp + A) ** 4) / ((T0 + A) ** 4)
    # (SG) compute degradation speed (page 6 Verbist, K. (2004). Computermodel RUSLE C-factor.)
    # (SG) special case if only one record
    a = p * np.min([W, F], axis=0) if len(W) > 1 else p * np.min([W.values[0], F.values[0]])
    return W, F, a


def compute_Bsb(bdate, edate, a, Bsi):
    """
    Computes harvest remains per unit of area over nodes

    Parameters
    ----------
        'edate' (pd series,timedate): see series ``grid`` in :func:`ComputeCFactor` and :func:`prepare_grid`
        'a' (pd series,float): see series ``grid`` in :func:`ComputeCFactor` and :func:`prepare_grid`
        'Bsi' (pd series,float): see series ``grid`` in :func:`ComputeCFactor` and :func:`prepare_grid`
        'bd' (datetime): begin date of series harvest remains with index har_id

    Returns
    -------
        'Bsi' (pd series, float): harvest remains per unit area on time step i (kg/m2)
    """
    Bsb = np.zeros(len(a))
    # (SG) compute remains on middle of computational node
    D = [7]+[(bdate.iloc[i] + (edate.iloc[i] - bdate.iloc[i]) /
              2 - (bdate.iloc[i-1] + (edate.iloc[i-1] - bdate.iloc[i-1]) / 2)).days for i in range(1, len(edate), 1)]
    exp = np.exp(-a * D)
    # (SG) compute harvest remains
    for i in range(len(exp)):
        Bsb[i] = Bsb[i-1] * exp[i] if i != 0 else Bsi.iloc[i] * exp[i]

    return Bsb


def celc_to_fahr(C):
    """
    Computes degree Celsisus

    Parameters
    ----------
        'C' (float, or pd series, array): temperature in degree celsius

    Returns
    -------
        'F'(float, or pd series, array): temperature in degree fahrenheight
    """
    return 9 / 5 * C + 32


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


def ComputeCFactor(parcel_list, grid, ggg, cp, year, output_interval="M",
                   ffull_output=False, output_map="Results", multiprocessing=False, fid="perceel_id"):
    """
    Script to computes C factor per parcel, based on formula's Verbist, K. (2004). Computermodel RUSLE C-factor.

    Parameters
    ----------
        'parcel_list' (pd df): list of parcels, with coupled crops and crop properties
            'REF_ID' (int): id of parcel according to 'perceelsregistratie'
            'perceel_id' (int): generic added id
            'jaar' (int): years considered
            'type' (string): type of crop, voorteelt, hoofdteelt, nateelt
            'GWSCOD' (int): code of the prop, see file 'teelt_eigenschappen.csv'
            'groep_id' (int/float): id of crop group
        'grid' (pd df): computation grid with columns
            'bdate' (datetime): begindate node
            'edate' (datetime): enddate node
            'bmonth' (datetime): month node begin
            'bday' (datetime): day node begin
            'rain' (float): summed rainfall node (mm)
            'temp' (float): average temperature node (degree Celsius)
            'Rhm' (float): erosion index ((MJ mm)/(ha h) as defined in see Verstraeten, G., Poesen, J., Demarée, G.,
             & Salles, C. (2006). Long-term (105 years) variability " \
             In rain erosivity as derived from 10-min rainfall depth data for Ukkel (Brussels, Belgium):
             Implications for assessing soil erosion rates. Journal of Geophysical Research Atmospheres, 111(22), 1–11.)
        'year' (int): year for which simulation is run
        'ggg' (pd df):
            'ggg_id' (int): id coupling crop growth properties
            'dagen_na' (int): number of days after sowing
            'Fc' (float): bodembedekking (m2/m2)
            'H' (float): effectieve valhoogte (m)
        'gts' (pd df):  teelt zaai en oogstdata + data groeicurves
            'groep_id' (int/float): id of crop group
            'zaaidatum1' (int): 'average' sowing date of crop or begin range sowing date
            'zaaidatum2' (int): end range sowing data
            'dagen_tot_oogst' (int): average number of days untill harvest
            'ggg_id' (int): id of crop growth curve (gewasgroeicurve)
        'cp' (pd df): crop properties (Bsi, alpha, p, Ri, see definitions
            1. Verbist, K. (2004). Computermodel RUSLE C-factor.
            2. Verstraeten, G., Van Oost, K., Van Rompaey, A., Poesen, J., & Govers, G. (2001). Integraal land-
                en waterbeheer in landelijke gebieden met het oog op het beperken van bodemverlies en modderoverlast
                ( proefproject gemeente Gingelom ) Integraal land- en waterbeheer in landelijke gebieden met het oog
                op het beperken van bodemverlies en m. Laboratory for Experimental Geomorphology (2014), 67.
            'groep_id' (int/float): id of crop group
            'groep_name' (string): name of crop group
        'output_interval' (str): output interval for C-factor (!= interval computation grid), either monthly 'M',
                                semi-monthly'SMS' or yearly 'Y'
        'full_ouput' (bool): write grid to disk for every parcel (default False)
        'output_map' (string): map to write outputs to
        'multiprocessing' (boolean):  use multiple cores (True= yes)
        "perceel_id" (str): flag for perceels_id

    Returns
    -------
         XXXXX

    """

    # (SG) Only compute parcels for which groep_id is known, and for which a main crop is defined for year i
    parcel_list_comp = parcel_list[parcel_list["compute"] == 1]

    # compute C
    if not multiprocessing:
        out = single(parcel_list_comp, grid, ggg,  cp, year, output_interval=output_interval,
                     ffull_output=ffull_output, output_map=output_map, fid=fid)
    else:
        out = multi(parcel_list_comp, grid, ggg,  cp, year, output_interval=output_interval,
                    ffull_output=ffull_output, output_map=output_map, fid=fid)

    return out


def multi(parcel_list, grid, ggg, cp, year, output_interval="M",
          ffull_output=False, output_map="Results", fid="perceel_id"):

    import multiprocessing

    ncpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(ncpu)
    un_parcels = parcel_list[fid].unique()
    length_job = int(np.ceil(len(un_parcels)/ncpu))
    ind = [np.min([len(un_parcels), i*length_job]) for i in np.arange(ncpu+1)]
    jobs = [0.] * ncpu
    output = []

    for i in range(len(jobs)):
        parcels_i = un_parcels[ind[i]:ind[i+1]]
        jobs[i] = pool.apply_async(eval('single'),
                                   (parcel_list.loc[parcel_list[fid].isin(parcels_i)],
                                    grid, ggg, cp, year, output_interval, ffull_output, output_map, fid, i))
    pool.close()

    for i in range(len(jobs)):
        temp = jobs[i].get()
        output.append(temp)

    return pd.concat(output)


def single(parcel_list, grid, ggg, cp, year, output_interval="M", ffull_output=False,
           output_map="Results", fid="perceel_id", processor=1):

    # note: calculations are computed on a forward grid,
    # e.g. the record with timetag 01/01 are calculations for coming X days
    # initialize columns
    SLRcol = ["SLR_%i" % i for i in range(24)]
    Hcol = ["H_%i" % i for i in range(24)]
    Fccol = ["Fc_%i" % i for i in range(24)]
    Rcol = ["R_%i" % i for i in range(24)]

    # (SG) filter duplicates
    # parcel_list = parcel_list[["GWSCOD","GWSNAM","NR","type","jaar","perceel_id","groep_id","voorteelt",
    #                          "hoofdteelt","nateelt"  groente  groenbedekker  meerjarige_teeltonvolledig
    #                          teeltgegevens_beschikbaar  compute]].drop_duplicats(columns=[[]])

    # ids = parcel_list[parcel_list["perceel_id"]==14432].index[0]
    # parcel_list = parcel_list.loc[ids:]

    un_par = parcel_list[fid].unique()
    out = []

    ind = 0
    # condn = 5
    # starttime = time.time()

    for parcel_id in un_par:
        print("[ComputeC] Running parcel_id %i on processor %i (%.2f %% complete)"
              % (parcel_id, processor, (ind/len(un_par)*100)))
        cond = parcel_list[fid] == parcel_id
        C, SLR, H, Fc, R = compute_C(parcel_list[cond], grid, ggg, cp, year, parcel_id,
                                     output_interval=output_interval, ffull_output=ffull_output, output_map=output_map)

        # (SG) print parcel to disk
        if ffull_output:
            parcel_list[cond].to_csv(os.path.join(output_map, "%i_rotation_scheme.csv" % int(parcel_id)))

        out.append(np.array([parcel_id]+[C]+SLR.tolist()+Fc.tolist()+H.tolist()))
        ind += 1

    out = pd.DataFrame(np.stack(out), columns=[fid, "C"] + SLRcol + Fccol + Hcol)
    # ind += 1

    # cond = np.round(100 * ind / len(parcel_list))

    # if cond >= condn:
    #    print("%i runs of %i done (%i %%), time elapsed (min): %.2f" % (
    #    ind, len(un_par), condn, (time.time() - starttime) / 60))
    #    condn += 5
    return out
