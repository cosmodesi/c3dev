"""
"""
import os
from astropy.table import Table
from ..utils.galmatch import compute_uber_host_indx


TASSO = "/Users/aphearin/work/DATA/DESI/C3EMC/UNIT"
BEBOP = "/lcrc/project/halotools/C3EMC/UNIT/v0.2"
NERSC = "/global/cfs/cdirs/desi/users/aphearin/C3EMC/UNIT"
BN = "out_107p.list.hdf5"
UNIT_LBOX = 1000.0  # Mpc/h


def read_unit_sim(fn):
    return Table.read(os.path.join(fn), path="data")


def load_value_added_unit_sim(fn):
    unit = read_unit_sim(fn)
    unit["uber_host_haloid"] = compute_uber_host_indx(
        unit["halo_upid"], unit["halo_id"]
    )
    return unit
