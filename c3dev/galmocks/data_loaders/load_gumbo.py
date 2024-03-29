"""
"""
import os
from astropy.table import Table


NERSC_DRN = "/global/cfs/cdirs/desi/users/aphearin/C3EMC/gumbo"
LATEST = "v0.3"


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
    elif version == "v0.1":
        bn = "gumbo_v0.1.h5"
    elif version == "v0.2":
        bn = "gumbo_v0.2.h5"
    elif version == "v0.3":
        bn = "gumbo_v0.3.h5"
    else:
        raise ValueError("gumbo mock version `{}` is unavailable".format(version))
    return bn
