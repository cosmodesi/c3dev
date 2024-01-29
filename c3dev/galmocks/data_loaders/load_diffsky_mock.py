"""
"""
import os

from astropy.table import Table

NERSC_DRN = "/global/cfs/cdirs/desi/users/gbeltzmo/C3EMC/UNIT"
LATEST = "v0.3"


def read_diffsky_mock(fn=None, drn=NERSC_DRN):
    """Read the diffsky mock from disk.

    Parameters
    ----------
    fn : string
        Mock file name

    drn : string, optional
        Mock directory. Default set by NERSC_DRN at top of module.

    """
    if fn is None:
        raise ValueError("No filename given")

    f = os.path.join(drn, fn)

    return Table.read(f, path="data")
