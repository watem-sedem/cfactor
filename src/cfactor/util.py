import os


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
    """create directory for output to which results are written to

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
