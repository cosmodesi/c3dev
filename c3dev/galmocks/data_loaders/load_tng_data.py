"""
"""
from collections import OrderedDict
import numpy as np
from halotools.utils import sliding_conditional_percentile


SANDY_SCRATCH_PATH = "/global/cscratch1/sd/sihany/TNG300-1/output"
BEBOP = "/lcrc/project/halotools/C3EMC/TNG300-1"


def load_tng_subhalos(drn=SANDY_SCRATCH_PATH, snapNum=55):
    import illustris_python as il

    subhalos = il.groupcat.loadSubhalos(drn, snapNum)
    return subhalos


def load_tng_host_halos(drn=SANDY_SCRATCH_PATH, snapNum=55):
    import illustris_python as il

    host_halos = il.groupcat.loadHalos(drn, snapNum)
    return host_halos


def get_value_added_tng_data(subs, hosts):
    tng = OrderedDict()
    tng["host_halo_mass"] = hosts["GroupMass"][subs["SubhaloGrNr"]] * 1e10
    tng["host_halo_pos"] = hosts["GroupPos"][subs["SubhaloGrNr"]]
    tng["host_halo_vel"] = hosts["GroupVel"][subs["SubhaloGrNr"]]

    tng["subhalo_pos"] = subs["SubhaloPos"]
    tng["subhalo_vel"] = subs["SubhaloVel"]
    tng["subhalo_mass"] = subs["SubhaloMass"] * 1e10
    tng["subhalo_vmax"] = subs["SubhaloVmax"]
    tng["subhalo_vdisp"] = subs["SubhaloVelDisp"]

    tng["stellar_metallicity"] = subs["SubhaloStarMetallicity"]
    tng["subhalo_mgas"] = subs["SubhaloMassType"][:, 0] * 1e10
    tng["subhalo_dm"] = subs["SubhaloMassType"][:, 1] * 1e10
    tng["mstar"] = subs["SubhaloMassType"][:, 4] * 1e10
    tng["griz"] = subs["SubhaloStellarPhotometrics"][:, 4:]

    tng["host_halo_index"] = subs["SubhaloGrNr"]

    # Broadcast properties of the central subhalo to each group member
    uvals, indices, counts = np.unique(
        subs["SubhaloGrNr"], return_counts=True, return_index=True
    )
    hosts["central_subhalo_vmax"] = subs["SubhaloVmax"][indices]
    hosts["central_subhalo_vdisp"] = subs["SubhaloVelDisp"][indices]
    tng["host_halo_vmax"] = hosts["central_subhalo_vmax"][subs["SubhaloGrNr"]]
    tng["host_halo_vdisp"] = hosts["central_subhalo_vdisp"][subs["SubhaloGrNr"]]

    hosts["p_vmax"] = sliding_conditional_percentile(
        hosts["GroupMass"], hosts["central_subhalo_vmax"], 101
    )
    hosts["p_vdisp"] = sliding_conditional_percentile(
        hosts["GroupMass"], hosts["central_subhalo_vdisp"], 101
    )
    tng["host_halo_p_vmax"] = hosts["p_vmax"][subs["SubhaloGrNr"]]
    tng["host_halo_p_vdisp"] = hosts["p_vdisp"][subs["SubhaloGrNr"]]

    return tng
