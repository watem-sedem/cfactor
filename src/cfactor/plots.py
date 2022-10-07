from general import load_db, create_dir
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import subprocess
import pdb
from distutils.spawn import find_executable
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)


def generate_report(fname_inputs, resmap, fname_output, GWSCOD=-1):

    # (SG) check if latex is installed
    runLatex = True
    if find_executable("latex"):
        print("latex installed")
    else:
        print(
            "Warning: Latex is not installed on this computer, report of inputdata is not generated! Please install latex before running this script again!"
        )
        runLatex = False

    if runLatex:

        ggg, te, gwscod, _, _, _ = load_db(fname_inputs)
        ggg = ggg[ggg["dagen_na"] != 2000]  # (SG) delete synth. dagen na of 2000

        lo = latexobject(fname_output, os.path.join(resmap, "report"))
        lo.writeheader()

        # selecteer één Specifieke gewascodes
        if GWSCOD != -1:
            gwscod = gwscod[gwscod["GWSCOD"] == GWSCOD]

        create_dir(resmap, ["report/temp"])

        for i in gwscod["groep_id"].unique():
            lo = initialiseer_fiches_teelt(lo, gwscod, i)
            lo = compileer_teeltwisseling(
                lo, te, ggg, i, os.path.join(resmap, "report")
            )

        lo.close()
        lo.compile(os.path.join(resmap, "report"))


def initialiseer_fiches_teelt(lo, gwscod, i):
    cond = gwscod["groep_id"] == i
    GROEP_ID = gwscod.loc[cond, "groep_id"].values[0]
    GROEP_NAME = gwscod.loc[cond, "groep_naam"].values[0]
    GWSCOD = gwscod.loc[cond, "GWSCOD"].values.flatten().tolist()
    GWSNAM = gwscod.loc[cond, "GWSNAM"].values.flatten().tolist()
    MEERJARIG = gwscod.loc[cond, "meerjarige_teelt"].values[0]
    GROENTE = gwscod.loc[cond, "groente"].values[0]
    GROENBEDEKKER = gwscod.loc[cond, "groenbedekker"].values[0]

    GWSNAM = [fix_string(i) for i in GWSNAM]
    GROEP_NAME = fix_string(GROEP_NAME)

    # (SG) write title
    lo.init_crop(GROEP_ID, GROEP_NAME)

    # (SG) write GWSNAM and GWSCOD that are coupled
    lo.write_gws(GWSNAM, GWSCOD, MEERJARIG, GROENBEDEKKER, GROENTE)

    return lo


def compileer_teeltwisseling(lo, te, ggg, i, resmap):
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
    string = string.replace("_", " ")
    return string


class latexobject:
    def __init__(self, fname, resmap):

        self.fname_ = fname + ".tex"
        self.fname = os.path.join(resmap, self.fname_)

    def writeheader(self):
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
        for i in packages:
            f.write(r"" + i)
            f.write("\n")
        return f

    def add_figure(self, fname):
        cmd = (
            r"\begin{center} \begin{figure}[H] \includegraphics[width=12.5cm]{%s.png} \end{figure} \end{center}"
            % fname.replace("\\", "/")
        )
        self.command(cmd)

    def writesubsection(self, title):
        self.command(r"\subsection{%s}" % title.capitalize())

    def newpage(self):
        self.command(r"\newpage")

    def init_crop(self, groep_id, groep_name):
        cmd = r"\section{%s (groep\_id %i)}" % (groep_name.capitalize(), groep_id)
        self.command(cmd)

    def write_gws(self, gwsnam, gwscod, meerjarig, groenbedekker, groente):
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
        # beetje vuile truck, switchen van directory voor compilatie pdf en dan terug
        os.chdir(fmap)
        proc = subprocess.Popen(["pdflatex", self.fname_])
        proc.communicate()
        os.chdir(cwd)


def plot_growth_curve(subgroep_id, data, resmap):
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
