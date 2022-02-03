"""
"""


SANDY_SCRATCH_PATH = "/global/cscratch1/sd/sihany/TNG300-1/output"
BEBOP = "/lcrc/project/halotools/C3GMC/TNG300-1"


def load_tng_subhalos(drn=SANDY_SCRATCH_PATH):
    import illustris_python as il

    subhalos = il.groupcat.loadSubhalos(drn, 55)
    return subhalos
