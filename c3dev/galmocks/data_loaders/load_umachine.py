"""
"""
import numpy as np

BEBOP_DRN = "/lcrc/project/halotools/C3GMC/UM/SMDPL/SFR_snapshot_binaries"
BASENAME = "sfr_catalog_0.550400.bin"
SMDPL_LBOX = 400.0  # Mpc/h

DTYPE = np.dtype(
    dtype=[
        ("id", "i8"),
        ("descid", "i8"),
        ("upid", "i8"),
        ("flags", "i4"),
        ("uparent_dist", "f4"),
        ("pos", "f4", (6)),
        ("vmp", "f4"),
        ("lvmp", "f4"),
        ("mp", "f4"),
        ("m", "f4"),
        ("v", "f4"),
        ("r", "f4"),
        ("rank1", "f4"),
        ("rank2", "f4"),
        ("ra", "f4"),
        ("rarank", "f4"),
        ("A_UV", "f4"),
        ("sm", "f4"),
        ("icl", "f4"),
        ("sfr", "i4"),
        ("obs_sm", "f4"),
        ("obs_sfr", "f4"),
        ("obs_uv", "f4"),
        ("empty", "f4"),
    ],
    align=True,
)


def read_sfr_snapshot(fn):
    return np.fromfile(fn, dtype=DTYPE)
