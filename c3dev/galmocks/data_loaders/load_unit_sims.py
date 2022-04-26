"""
"""
import os
from astropy.table import Table


TASSO = "/Users/aphearin/work/DATA/DESI/C3EMC/UNIT"
BEBOP = "/lcrc/project/halotools/C3EMC/UNIT"
EXAMPLE_BN = "out_107p.list.hdf5"
UNIT_LBOX = 1000.0  # Mpc/h


def read_unit_sim(fn):
    return Table.read(os.path.join(fn), path="data")
