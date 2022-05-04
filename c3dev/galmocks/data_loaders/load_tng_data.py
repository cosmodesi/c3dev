"""
"""
from collections import OrderedDict
import numpy as np
from halotools.utils import sliding_conditional_percentile
from astropy.table import Table


SANDY_SCRATCH_PATH = "/global/cscratch1/sd/sihany/TNG300-1/output"
BEBOP = "/lcrc/project/halotools/C3EMC/TNG300-1"
NERSC = "/global/cfs/cdirs/desi/users/aphearin/C3EMC/TNG300-1"
TNG_LBOX = 205.0


def load_tng_subhalos(drn=NERSC, snapNum=55):
    import illustris_python as il

    subhalos = il.groupcat.loadSubhalos(drn, snapNum)
    return subhalos


def load_tng_host_halos(drn=NERSC, snapNum=55):
    import illustris_python as il

    host_halos = il.groupcat.loadHalos(drn, snapNum)
    return host_halos


def get_value_added_tng_data(subs, hosts):
    hosts["halo_id"] = np.arange(len(hosts["GroupMass"])).astype(int)

    host_keys_to_keep = ["halo_id", "GroupFirstSub", "GroupPos", "GroupVel"]
    tng_hosts = Table(OrderedDict([(key, hosts[key]) for key in host_keys_to_keep]))
    tng_hosts.rename_column("GroupPos", "pos")
    tng_hosts.rename_column("GroupVel", "vel")
    tng_hosts["logmh"] = np.log10(hosts["GroupMass"]) + 10
    tng_hosts["pos"] = tng_hosts["pos"] / 1000

    tng = Table()
    tng["host_halo_logmh"] = tng_hosts["logmh"][subs["SubhaloGrNr"]]
    tng["host_halo_pos"] = tng_hosts["pos"][subs["SubhaloGrNr"]]
    tng["host_halo_vel"] = tng_hosts["vel"][subs["SubhaloGrNr"]]

    tng["subhalo_pos"] = subs["SubhaloPos"] / 1000
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

    subhalo_id = np.arange(len(subs["SubhaloGrNr"])).astype(int)
    subhalo_cen_id = subhalo_id[tng_hosts["GroupFirstSub"]]
    tng["is_central"] = subhalo_cen_id == subhalo_id

    # Broadcast properties of the central subhalo to each host
    tng_hosts["central_subhalo_vmax"] = subs["SubhaloVmax"][tng_hosts["GroupFirstSub"]]
    tng_hosts["central_subhalo_vdisp"] = subs["SubhaloVelDisp"][
        tng_hosts["GroupFirstSub"]
    ]

    # Broadcast properties of the central subhalo to each group member
    tng["host_halo_vmax"] = tng_hosts["central_subhalo_vmax"][subs["SubhaloGrNr"]]
    tng["host_halo_vdisp"] = tng_hosts["central_subhalo_vdisp"][subs["SubhaloGrNr"]]

    tng_hosts["p_vmax"] = sliding_conditional_percentile(
        tng_hosts["logmh"], tng_hosts["central_subhalo_vmax"], 101
    )
    tng_hosts["p_vdisp"] = sliding_conditional_percentile(
        tng_hosts["logmh"], tng_hosts["central_subhalo_vdisp"], 101
    )
    tng["host_halo_p_vmax"] = tng_hosts["p_vmax"][subs["SubhaloGrNr"]]
    tng["host_halo_p_vdisp"] = tng_hosts["p_vdisp"][subs["SubhaloGrNr"]]

    return tng, tng_hosts
