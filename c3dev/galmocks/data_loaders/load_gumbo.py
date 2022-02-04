"""
"""
import os
from astropy.table import Table


NERSC_DRN = "/global/cfs/cdirs/desi/users/aphearin/C3GMC/gumbo"
LATEST = "v0.0"


def read_gumbo_mock(fn=None, drn=NERSC_DRN, version=LATEST):
    """Read the gumbo mock from disk. Default is to load the latest version.

    Parameters
    ----------
    fn : string, optional
        Absolute path to a specific version of gumbo

    version : string, optional
        Version of gumbo to read. Default set by LATEST at top of module.

    """
    if fn is None:
        bn = _get_gumbo_basename(version)
        fn = os.path.join(drn, bn)

    return Table.read(fn, path="data")


def _get_gumbo_basename(version):
    if version == "v0.0":
        bn = "gumbo_v0.0.h5"
    else:
        raise ValueError("No other available versions of the gumbo mock")
    return bn
