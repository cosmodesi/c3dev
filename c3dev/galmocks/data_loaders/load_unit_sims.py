"""
"""
import os
from astropy.table import Table


TASSO = "/Users/aphearin/work/DATA/DESI/C3GMC/UNIT"
BEBOP = "/lcrc/project/halotools/C3GMC/UNIT"
EXAMPLE_BN = "out_107p.list.hdf5"
UNIT_SIM_LBOX = 1000.0  # Mpc/h


def read_unit_sim(fn):
    return Table.read(os.path.join(fn), path="data")
